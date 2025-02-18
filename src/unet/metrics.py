import argparse
import os

import cv2
import numpy as np
from numba import njit
from tensorflow import keras

def create_weight_matrix(mask_size=5):
    weight_matrix = np.zeros((mask_size, mask_size), dtype=np.float64)
    center = mask_size // 2

    for i in range(mask_size):
        for j in range(mask_size):
            if i == center and j == center:
                weight_matrix[i, j] = 0.0
            else:
                distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                weight_matrix[i, j] = 1.0 / distance

    # Normalize the weight matrix
    sum_weights = np.sum(weight_matrix)

    return weight_matrix / sum_weights


@njit
def dr_dcalc(i, j, image_gt, image, normalized_weight_matrix, mask_size):
    height, width = image_gt.shape

    value_gk = image[i, j]

    Bk = np.zeros((mask_size, mask_size), dtype=np.float64)
    Dk = np.zeros((mask_size, mask_size), dtype=np.float64)
    h = 2
    for x in range(mask_size):
        for y in range(mask_size):
            if i - h + x < 0 or j - h + y < 0 or i - h + x >= height or j - h + y >= width:
                Bk[x, y] = value_gk
            else:
                Bk[x, y] = image_gt[i - h + x, j - h + y];
            Dk[x, y] = abs(Bk[x, y] - value_gk);

        DRDk = Dk * normalized_weight_matrix

    res = np.sum(DRDk)

    return res


@njit
def NUBNcalc(f, ii, jj, blck):
    startx = (ii - 1) * blck
    endx = ii * blck
    starty = (jj - 1) * blck
    endy = jj * blck

    check_prv = -2
    retb = 0.0

    for xx in range(startx, endx):
        for yy in range(starty, endy):
            check = f[xx, yy]
            if check_prv < 0:
                check_prv = check
            else:
                if check != check_prv:
                    retb = 1
                    break
        if retb != 0:
            break

    return retb


def get_drd(im, im_gt, normalized_weight_matrix):
    img_height, img_width = im.shape

    block_size = 8
    n = 2
    mask_size = 2 * n + 1

    total_nubn = 0.0
    xb = img_height // block_size
    yb = img_width // block_size

    for i in range(1, xb + 1):
        for j in range(1, yb + 1):
            nubn_b = NUBNcalc(im_gt, i, j, block_size)
            total_nubn += nubn_b

    np.set_printoptions(precision=20, suppress=True, threshold=np.inf)
    total_drd = 0.0

    difference_image = np.where(np.abs(im - im_gt) > 0.5)

    for i, j in zip(difference_image[0], difference_image[1]):
        total_drd += dr_dcalc(i, j, im_gt, im, normalized_weight_matrix, mask_size)

    return total_drd / total_nubn


def load_image_as_binary(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    binary_array = (image > 0).astype(np.uint8)
    return binary_array


def calculate_metrics(im, im_gt, r_weight, p_weight):
    height, width = im_gt.shape

    # Compute DRD metric
    normalized_weight_matrix = create_weight_matrix()
    drd = get_drd(im, im_gt, normalized_weight_matrix)

    # Create masks
    TP_mask = (im == 0) & (im_gt == 0)
    FP_mask = (im == 0) & (im_gt == 1)
    FN_mask = (im == 1) & (im_gt == 0)

    # Compute weighted weights
    weighted_p_weight = 1.0 + p_weight

    # Sum weighted weights using masks
    TPwp = weighted_p_weight[TP_mask].sum()
    FPwp = weighted_p_weight[FP_mask].sum()
    TPwr = r_weight[TP_mask].sum()
    FNwr = r_weight[FN_mask].sum()

    # Compute weighted precision, recall, and F-measure
    w_precision = TPwp / (TPwp + FPwp) if (TPwp + FPwp) > 0 else 0.0
    w_recall = TPwr / (TPwr + FNwr) if (TPwr + FNwr) > 0 else 0.0
    w_f_measure = (2 * w_precision * w_recall) / (w_precision + w_recall) if (w_precision + w_recall) > 0 else 0.0

    # Compute standard precision, recall, and F-measure
    TP = TP_mask.sum()
    FP = FP_mask.sum()
    FN = FN_mask.sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute MSE and PSNR
    npixel = height * width
    mse = (FP + FN) / npixel
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else float('inf')

    f_measure *= 100
    w_f_measure *= 100

    return f_measure, w_f_measure, psnr, drd


def calculate_for_all(gts, imgs, filenames):
    total_fmeasure = 0.0
    total_pf_measure = 0.0
    total_psnr = 0.0
    total_drd = 0.0

    for i in range(len(gts)):

        filename = filenames[i]
        print(filename)
        width, height = gts[i].shape
        r_weight = np.loadtxt(os.path.join("./dataset/validation/r_weights", filename + "_GT_RWeights.dat"),
                              dtype=np.float64).flatten()[:height * width].reshape(
            (height, width))
        p_weight = np.loadtxt(os.path.join("./dataset/validation/p_weights", filename + "_GT_PWeights.dat"),
                              dtype=np.float64).flatten()[:height * width].reshape(
            (height, width))

        fmeasure, pfmeasure, psnr, drd = calculate_metrics(imgs[i], gts[i],r_weight, p_weight)
        total_fmeasure += fmeasure
        total_pf_measure += pfmeasure
        total_psnr += psnr
        total_drd += drd

    total_fmeasure /= len(gts)
    total_pf_measure /= len(gts)
    total_psnr /= len(gts)
    total_drd /= len(gts)

    return total_fmeasure, total_pf_measure, total_psnr, total_drd


class CustomMetricCallback(keras.callbacks.Callback):
    def __init__(self, validation_generator):
        super().__init__()
        self.validation_generator = validation_generator

    def on_epoch_end(self, epoch, logs=None):

        print("Epoch ended. Now calculating metrics")
        logs = logs or {}
        all_preds = []
        all_trues = []
        all_filenames = []

        for i in range(len(self.validation_generator)):
            batch = self.validation_generator[i]
            if len(batch) == 3:
                x, y, filenames = batch
            else:
                raise ValueError("Validation generator must return inputs, targets, and filenames.")

            preds = self.model.predict(x)
            all_preds.append(preds)
            all_trues.append(y)
            all_filenames.extend(filenames)

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)

        # Compute the custom metric using filenames
        fmeasure, pfmeasure, psnr, drd = calculate_for_all(all_trues, all_preds, all_filenames)
        logs["fmeasure"] = fmeasure
        logs["pfmeasure"] = pfmeasure
        logs["psnr"] = psnr
        logs["drd"] = drd