import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbEvalCallback
import argparse
import random
import sys
import time
from copy import deepcopy
import Augmentor
import PIL
import imageio
from Augmentor.Operations import Operation
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (TensorBoard, Callback,ModelCheckpoint)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from model.unet import unet
from metrics import CustomMetricCallback
from utils.img_processing import *


class GaussianNoiseAugmentor(Operation):
    """Gaussian Noise in Augmentor format."""

    def __init__(self, probability, mean, sigma):
        Operation.__init__(self, probability)
        self.mean = mean
        self.sigma = sigma

    def __gaussian_noise__(self, image):
        img = np.array(image).astype(np.int16)
        tmp = np.zeros(img.shape, np.int16)
        img = img + cv2.randn(tmp, self.mean, self.sigma)
        img[img < 0] = 0
        img[img > 255] = 255
        img = img.astype(np.uint8)
        image = PIL.Image.fromarray(img)

        return image

    def perform_operation(self, images):
        images = [self.__gaussian_noise__(image) for image in images]
        return images


class SaltPepperNoiseAugmentor(Operation):
    """Gaussian Noise in Augmentor format."""

    def __init__(self, probability, prop):
        Operation.__init__(self, probability)
        self.prop = prop

    def __salt_pepper_noise__(self, image):
        img = np.array(image).astype(np.uint8)
        h = img.shape[0]
        w = img.shape[1]
        n = int(h * w * self.prop)
        for i in range(n // 2):
            # Salt.
            curr_y = int(np.random.randint(0, h))
            curr_x = int(np.random.randint(0, w))
            img[curr_y, curr_x] = 255
        for i in range(n // 2):
            # Pepper.
            curr_y = int(np.random.randint(0, h))
            curr_x = int(np.random.randint(0, w))
            img[curr_y, curr_x] = 0
        image = PIL.Image.fromarray(img)

        return image

    def perform_operation(self, images):
        images = [self.__salt_pepper_noise__(image) for image in images]
        return images


class InvertPartAugmentor(Operation):
    """Invert colors in Augmentor formant."""

    def __init__(self, probability):
        Operation.__init__(self, probability)

    def __invert__(self, image):
        img = np.array(image).astype(np.uint8)
        h = img.shape[0]
        w = img.shape[1]
        y_begin = int(np.random.randint(0, h))
        x_begin = int(np.random.randint(0, w))
        y_add = int(np.random.randint(0, h - y_begin))
        x_add = int(np.random.randint(0, w - x_begin))
        for i in range(y_begin, y_begin + y_add):
            for j in range(x_begin, x_begin + x_add):
                img[i][j] = 255 - img[i][j]
        image = PIL.Image.fromarray(img)

        return image

    def perform_operation(self, images):
        images = [self.__invert__(image) for image in images]
        return images


def __apply_augmentation__(p):
    batch = []
    for i in range(0, len(p.augmentor_images)):
        images_to_return = [Image.fromarray(x) for x in p.augmentor_images[i]]

        for operation in p.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                images_to_return = operation.perform_operation(images_to_return)

        images_to_return = [np.asarray(x) for x in images_to_return]
        batch.append(images_to_return)
    return batch


def augmentate_batch(imgs_in, imgs_gt):
    """Generate ordered augmented batch of images, using Augmentor"""

    # Non-Linear transformations.
    imgs = [[imgs_in[i], imgs_gt[i]] for i in range(len(imgs_in))]
    p = Augmentor.DataPipeline(imgs)
    p.random_distortion(0.5, 6, 6, 4)
    # Linear transformations.
    # p.rotate(0.75, 15, 15)
    p.shear(0.75, 10.0, 10.0)
    p.zoom(0.75, 1.0, 1.2)
    p.skew(0.75, 0.75)
    imgs = __apply_augmentation__(p)
    imgs_in = [p[0] for p in imgs]
    imgs_gt = [p[1] for p in imgs]

    # Noise transformations.
    p = Augmentor.DataPipeline([[img] for img in imgs_in])
    gaussian_noise = GaussianNoiseAugmentor(0.25, 0, 10)
    p.add_operation(gaussian_noise)
    salt_pepper_noise = SaltPepperNoiseAugmentor(0.25, 0.005)
    p.add_operation(salt_pepper_noise)
    # Brightness transformation.
    p.random_brightness(0.75, 0.5, 1.5)
    p.random_contrast(0.75, 0.5, 1.5)
    # Colors invertion.
    invert = InvertPartAugmentor(0.25)
    p.add_operation(invert)
    p.invert(0.5)
    imgs_in = __apply_augmentation__(p)
    imgs_in = [p[0] for p in imgs_in]

    return imgs_in, imgs_gt


class ParallelDataGenerator(Sequence):
    """Generate images for training/validation/testing (parallel version)."""

    def __init__(self, fnames_in, fnames_gt, batch_size: int, augmentate: bool, return_filenames: bool = False,
                 shuffle: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.fnames_in = deepcopy(fnames_in)
        self.fnames_gt = deepcopy(fnames_gt)
        self.batch_size = batch_size
        self.augmentate = augmentate
        self.return_filenames = return_filenames
        self.shuffle = shuffle
        self.idxs = np.arange(len(self.fnames_in))
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def __len__(self):
        return int(np.ceil(len(self.idxs) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def __getitem__(self, idx):
        # Determine batch indices
        start = idx * self.batch_size
        stop = start + self.batch_size
        stop = min(stop, len(self.idxs))
        batch_indices = self.idxs[start:stop]

        imgs_in = []
        imgs_gt = []

        for i in batch_indices:
            img_in = cv2.imread(self.fnames_in[i], cv2.IMREAD_GRAYSCALE)
            img_gt = cv2.imread(self.fnames_gt[i], cv2.IMREAD_GRAYSCALE)
            imgs_in.append(img_in)
            imgs_gt.append(img_gt)

        # Apply augmentations if needed
        if self.augmentate:
            imgs_in, imgs_gt = augmentate_batch(imgs_in, imgs_gt)

        # Normalize images
        imgs_in = np.array([normalize_in(img) for img in imgs_in])
        imgs_in = imgs_in[..., np.newaxis]  # Add channel dimension
        imgs_gt = np.array([normalize_gt(img) for img in imgs_gt])
        imgs_gt = imgs_gt[..., np.newaxis]  # Add channel dimension


        return imgs_in, imgs_gt

# class Visualisation(Callback):
#     """Custom Keras callback for visualising training through GIFs."""
#
#     def __init__(self, dir_name: str = 'visualisation', batchsize: int = 20,
#                  monitor: str = 'val_loss', save_best_epochs_only: bool = False, mode: str = 'min'):
#         super(Visualisation, self).__init__()
#         self.dir_name = dir_name
#         self.batchsize = batchsize
#         self.epoch_number = 0
#         self.fnames = os.listdir(self.dir_name)
#         for fname in self.fnames:
#             mkdir_s(os.path.join(self.dir_name, fname[:fname.rfind('.')] + '_frames'))
#         self.monitor = monitor
#         self.save_best_epochs_only = save_best_epochs_only
#         self.mode = mode
#         self.curr_metric = None
#
#     def on_train_end(self, logs=None):
#         for fname in self.fnames:
#             frames = []
#             for frame_name in sorted(os.listdir(os.path.join(self.dir_name, fname[:fname.rfind('.')] + '_frames'))):
#                 frames.append(imageio.imread(os.path.join(self.dir_name,
#                                                           fname[:fname.rfind('.')] + '_frames',
#                                                           frame_name)))
#             imageio.mimsave(os.path.join(self.dir_name, fname[:fname.rfind('.')] + '.gif'),
#                             frames, format='GIF', duration=0.5)
#             # rmtree(os.path.join(self.dir_name, fname[:fname.rfind('.')] + '_frames'))
#
#     def on_epoch_end(self, epoch, logs):
#         self.epoch_number += 1
#         if (not self.save_best_epochs_only) or \
#                 ((self.curr_metric is None) or
#                  (self.mode == 'min' and logs[self.monitor] < self.curr_metric) or
#                  (self.mode == 'max' and logs[self.monitor] > self.curr_metric)):
#             self.curr_metric = logs[self.monitor]
#             for fname in self.fnames:
#                 img = cv2.imread(os.path.join(self.dir_name, fname), cv2.IMREAD_GRAYSCALE).astype(np.float32)
#                 img = binarize_img(img, self.model, self.batchsize)
#                 cv2.imwrite(os.path.join(self.dir_name, fname[:fname.rfind('.')] + '_frames',
#                                          str(self.epoch_number) + '_out.png'), img)


def create_callbacks(args):
    """Create Keras callbacks for training."""
    callbacks = []


    model_checkpoint = ModelCheckpoint(filepath=os.path.join(args.weights,
                                                            'weights-improvement-{epoch:02d}.weights.h5'),
                                         monitor='val_dice_coef', mode='max', verbose=1,
                                          save_best_only=True, save_weights_only=True)

    callbacks.append(model_checkpoint)

    # Training visualisation.
    # if args.vis != '':
    #     model_visualisation = Visualisation(dir_name=args.vis, batchsize=args.batchsize, monitor='val_dice_coef',
    #                                         save_best_epochs_only=True, mode='max')
    #     callbacks.append(model_visualisation)

    callbacks.append(WandbMetricsLogger())

    return callbacks


def dice_coef(y_true, y_pred):
    """Count Sorensen-Dice coefficient for output and ground-truth image."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def dice_coef_loss(y_true, y_pred):
    """Count loss of Sorensen-Dice coefficient for output and ground-truth image."""
    return 1 - dice_coef(y_true, y_pred)

def jacard_coef(y_true, y_pred):
    """Count Jaccard coefficient for output and ground-truth image."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    """Count loss of Jaccard coefficient for output and ground-truth image."""
    return 1 - jacard_coef(y_true, y_pred)


def mkdir_s(path: str):
    """Create directory in specified path, if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)


desc_str = r"""Train U-net with pairs of train and ground-truth images.

All train images should be in "in" directory.
All ground-truth images should be in "gt" directory.

"""


def parse_args():
    parser = argparse.ArgumentParser(prog='train',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=desc_str)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v0.1')

    # Main training settings.
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help=r'number of training epochs (default: %(default)s)')
    parser.add_argument('-b', '--batchsize', type=int, default=20,
                        help=r'number of images, simultaneously sent to the GPU (default: %(default)s)')
    parser.add_argument('-a', '--augmentate', action='store_true',
                        help=r'use Keras data augmentation')

    # paths.
    parser.add_argument('-i', '--input', type=str, default="./dataset/train",
                        help=r'directory with input train and ground-truth images (default: "%(default)s")')
    parser.add_argument('-w', '--weights', type=str, default="./weights",
                        help=r'output U-net weights file (default: "%(default)s")')

    parser.add_argument('--vis', type=str, default='',
                        help=r'directory with images for training visualisation')

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)

    # Hardware.
    parser.add_argument('-p', '--extraprocesses', type=int, default=0,
                        help=r'number of extra processes for data augmentation (default: %(default)s)')
    parser.add_argument('-q', '--queuesize', type=int, default=10,
                        help=r'max size of training queue (default: %(default)s)')

    args = parser.parse_args()

    assert (args.epochs > 0)
    assert (args.batchsize > 0)
    assert (args.extraprocesses >= 0)
    assert (args.queuesize >= 0)

    return args


def main():
    start_time = time.time()
    args = parse_args()
    np.random.seed()

    wandb.init(
        project="robin",
        config={
            "dataset": args.input,
            "extra_processes": args.extraprocesses,
            "queue_size": args.queuesize,
            "learning_rate": args.learning_rate,
            "batch_size": args.batchsize,
            "epochs": args.epochs,
            "augmentate": args.augmentate,
        }
    )

    # Creating data for training, validation and testing.
    fnames_in = [os.path.join(args.input, 'in', str(i) + '_in.png')
                 for i in range(len(os.listdir(os.path.join(args.input, 'in'))))]
    fnames_gt = [os.path.join(args.input, 'gt', str(i) + '_gt.png')
                 for i in range(len(os.listdir(os.path.join(args.input, 'gt'))))]
    assert (len(fnames_in) == len(fnames_gt))

    train_generator = ParallelDataGenerator(
        fnames_in=fnames_in,
        fnames_gt=fnames_gt,
        batch_size=args.batchsize,
        augmentate=args.augmentate,
        workers=args.extraprocesses,
        max_queue_size=args.queuesize,
        use_multiprocessing=True
    )

    # validation_in = [os.path.join("validation_patches", 'in', str(i) + '_in.png')
    #              for i in range(len(os.listdir(os.path.join("validation_patches", 'in'))))]
    # validation_gt = [os.path.join("validation_patches", 'in', str(i) + '_in.png')
    #              for i in range(len(os.listdir(os.path.join("validation_patches", 'in'))))]

    # validation_generator = ParallelDataGenerator(
    #     fnames_in=validation_in,
    #     fnames_gt=validation_gt,
    #     batch_size=args.batchsize,
    #     augmentate=False,
    #     workers=args.extraprocesses,
    #     max_queue_size=args.queuesize,
    #     use_multiprocessing=True,
    #     return_filenames=True
    # )

    model = unet()
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_coef_loss, metrics=[dice_coef, jacard_coef, 'accuracy'])
    callbacks = create_callbacks(args)
    callbacks.append(CustomMetricCallback())

    model.fit(
        x=train_generator,
        steps_per_epoch= len(train_generator),
        # validation_data=validation_generator,
        # validation_steps=len(validation_generator),
        epochs = args.epochs,
        shuffle=True,
        callbacks=callbacks,
    )


    print("finished in {0:.2f} seconds".format(time.time() - start_time))
    wandb.finish()
    # Sometimes script freezes.
    sys.exit(0)


if __name__ == "__main__":
    main()
