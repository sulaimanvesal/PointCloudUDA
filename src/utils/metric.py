import numpy as np
from medpy.metric.binary import hd, dc, asd


#
# Functions to process files, directories and metrics
#

def metrics(img_gt, img_pred, ifhd=True, ifasd=True):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.

    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = []
    # Loop on each classes of the input images
    # endo, rv, myo
    cat = {500:'endo', 600:'rv', 200:'myo'}
    for c in [500, 600, 200]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)
        h_d = hd(gt_c_i, pred_c_i) if ifhd else -1
        a_sd = asd(gt_c_i, pred_c_i) if ifasd else -1
        # try:
        #     h_d = hd(gt_c_i, pred_c_i)
        # except RuntimeError:
        #     print("{} no h_d".format(cat[c]))
        #     h_d = -1
        # try:
        #     a_sd = asd (gt_c_i, pred_c_i)
        # except RuntimeError:
        #     print("{} no a_sd".format(cat[c]))
        #     a_sd = -1

        # Compute volume
        res += [dice, h_d, a_sd]

    return res


def compute_metrics_on_files(gt, pred, ifhd=True, ifasd=True):
    """
    Function to give the metrics for two files

    Parameters
    ----------

    path_gt: string
    Path of the ground truth image.

    path_pred: string
    Path of the predicted image.
    """
    res = metrics(gt, pred, ifhd=ifhd, ifasd=ifasd)
    res_str = ["{:.3f}".format(r) for r in res]
    formatting = "Endo {:>8} , {:>8} , {:>8} , RV {:>8} , {:>8} , {:>8} , Myo {:>8} , {:>8} , {:>8}"
    # print(formatting.format(*HEADER))
    print(formatting.format(*res_str))
    # to_save ="Endo: {}".format(np.around((res[0] + res[3] + res[6]) / 3., 3))
    # f = open('output.txt', 'a')
    # # print(formatting.format(*res), file=f)  # Python 3.x
    # print(to_save, file=f)  # Python 3.x

    return res


def dice_coef(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + 1.0) / (np.sum(y_true) + np.sum(y_pred) + 1.0)


def dice_coef_multilabel(y_true, y_pred, numLabels=4, channel='channel_first'):
    """
    :param y_true:
    :param y_pred:
    :param numLabels:
    :return:
    """
    assert channel=='channel_first' or channel=='channel_last', r"channel has to be either 'channel_first' or 'channel_last'"
    dice = 0
    if channel == 'channel_first':
        y_true = np.moveaxis(y_true, 1, -1)
        y_pred = np.moveaxis(y_pred, 1, -1)
    for index in range(1, numLabels):
        temp = dice_coef(y_true[:, :, :, index], y_pred[:, :, :, index])
        dice += temp

    dice = dice / 3
    return dice
