import numpy as np
from medpy.metric.binary import hd, dc, asd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from math import log


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
        temp = dice_coef(y_true[..., index], y_pred[..., index])
        dice += temp

    dice = dice / (numLabels - 1)
    return dice

def hausdorff_multilabel(y_true, y_pred, numLabels=4, channel='channel_first'):
    """
    :param y_true:
    :param y_pred:
    :param numLabels:
    :return:
    """
    assert channel=='channel_first' or channel=='channel_last', r"channel has to be either 'channel_first' or 'channel_last'"
    hd_score = 0
    if channel == 'channel_first':
        y_true = np.moveaxis(y_true, 1, -1)
        y_pred = np.moveaxis(y_pred, 1, -1)
    for index in range(1, numLabels):
        temp = hd(reference=y_true[:, :, :, index], result=y_pred[:, :, :, index])
        hd_score += temp

    hd_score = hd_score / (numLabels - 1)
    return hd_score

def metrics(img_gt, img_pred, apply_hd=False, apply_asd=False):
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

    res = {}
    class_name = ["myo", "lv", "rv"]
    # Loop on each classes of the input images
    for c, cls_name in zip([1, 2, 3], class_name) :
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
        h_d, a_sd = 0, 0
        if apply_hd:
            h_d = hd(gt_c_i, pred_c_i)
        if apply_asd:
            a_sd = asd (gt_c_i, pred_c_i)

        # Compute volume
        res[cls_name] = [dice, h_d, a_sd]

    return res


def metrics2(img_gt, img_pred, apply_hd=False, apply_asd=False):
    """
    the metrics function for CT/MR dataset
    """

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = {}
    class_name = ["myo", "la", "lv", "aa"]
    # Loop on each classes of the input images
    for c, cls_name in zip([1, 2, 3, 4], class_name) :

        gt_c_i = np.where(img_gt == c, 1, 0)
        pred_c_i = np.where(img_pred == c, 1, 0)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)
        h_d, a_sd = 0, 0
        if apply_hd:
            h_d = hd(gt_c_i, pred_c_i)
        if apply_asd:
            a_sd = asd (gt_c_i, pred_c_i)

        # Compute volume
        res[cls_name] = [dice, h_d, a_sd]

    return res



def batch_pairwise_dist(x, y):
    # 32, 2500, 3
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P

def batch_NN_loss(x, y):
    bs, num_points, points_dim = x.size()
    dist1 = torch.sqrt(batch_pairwise_dist(x, y))
    values1, indices1 = dist1.min(dim=2)

    dist2 = torch.sqrt(batch_pairwise_dist(y, x))
    values2, indices2 = dist2.min(dim=2)
    a = torch.div(torch.sum(values1,1), num_points)
    b = torch.div(torch.sum(values2,1), num_points)
    sum = torch.div(torch.sum(a), bs) + torch.div(torch.sum(b), bs)

    return sum

