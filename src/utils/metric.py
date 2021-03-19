import numpy as np
from medpy.metric.binary import hd, dc, asd


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
    calculate channel-wise dice similarity coefficient
    :param y_true: the ground truth
    :param y_pred: the prediction
    :param numLabels: the number of classes
    :param channel: 'channel_first' or 'channel_last'
    :return: the dice score
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


def evaluate(img_gt, img_pred, apply_hd=False, apply_asd=False):
    """
    Function to compute the metrics between two segmentation maps given as input.
    :param img_gt: Array of the ground truth segmentation map.
    :param img_pred: Array of the predicted segmentation map.
    :param apply_hd: whether to compute Hausdorff Distance.
    :param apply_asd: Whether to compute Average Surface Distance.
    :return: A list of metrics in this order, [dice myo, hd myo, asd myo, dice lv, hd lv asd lv, dice rv, hd rv, asd rv]
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
    evaluate the models on mmwhs data in batches
    :param img_gt: the ground truth
    :param img_pred: the prediction
    :param apply_hd: whether to evaluate Hausdorff Distance
    :param apply_asd: whether to evaluate Average Surface Distance
    :return:
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
        res[cls_name] = [dice, h_d, a_sd]
    return res


def compute_metrics_on_files(gt, pred, ifhd=True, ifasd=True):
    """
    Function to give the metrics for two files
    :param gt: The ground truth image.
    :param pred: The predicted image.
    :param ifhd: whether to calculate HD.
    :param ifasd: whether to calculate ASD
    :return:
    """

    def metrics(img_gt, img_pred, ifhd=True, ifasd=True):
        """
        Function to compute the metrics between two segmentation maps given as input.

        img_gt: Array of the ground truth segmentation map.

        img_pred: Array of the predicted segmentation map.
        Return: A list of metrics in this order, [Dice endo, HD endo, ASD endo, Dice RV, HD RV, ASD RV, Dice MYO, HD MYO, ASD MYO]
        """

        if img_gt.ndim != img_pred.ndim:
            raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                             "same dimension, {} against {}".format(img_gt.ndim,
                                                                    img_pred.ndim))
        res = []
        # cat = {500: 'endo', 600: 'rv', 200: 'myo'}
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

            h_d, a_sd = -1, -1
            if ifhd or ifasd:
                if np.sum(gt_c_i) == 0 or np.sum(pred_c_i) == 0:
                    dice = -1
                    h_d = -1
                    a_sd = -1
                else:
                    h_d = hd(gt_c_i, pred_c_i) if ifhd else h_d
                    a_sd = asd(gt_c_i, pred_c_i) if ifasd else a_sd
            res += [dice, h_d, a_sd]

        return res
    res = metrics(gt, pred, ifhd=ifhd, ifasd=ifasd)
    res_str = ["{:.3f}".format(r) for r in res]
    formatting = "Endo {:>8} , {:>8} , {:>8} , RV {:>8} , {:>8} , {:>8} , Myo {:>8} , {:>8} , {:>8}"
    print(formatting.format(*res_str))

    return res


if __name__ == '__main__':
    pass
