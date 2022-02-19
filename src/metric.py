import numpy as np
from medpy.metric.binary import hd, dc, asd
import torch


def metrics(img_gt, img_pred, pat_id, modality, apply_hd=False, apply_asd=False, class_name=("myo", "lv", "rv")):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.ndarray
    Array of the ground truth segmentation map.
    img_pred: np.ndarray
    Array of the predicted segmentation map.
    pat_id: int
    The pat which is to be evaluated on (only used for printing error message).
    modality: str
    Modality of the patient (only used for printing error message).
    apply_hd: bool
    Whether to calculate HD.
    apply_asd: bool
    Whether to calculate ASD.
    class_name: array like
    A list/tuple/array of class names

    Return a dictionary of the results for each class, example: {"myo": [], "lv": [], "rv": []}
    ------

    """
    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))
    res = {}
    # Loop on each classes of the input images
    for c, cls_name in zip(np.array(range(len(class_name))) + 1, class_name):
        # Copy the gt image to not alternate the input
        gt_c_i = np.where(img_gt == c, 1, 0)
        # Copy the pred image to not alternate the input
        pred_c_i = np.where(img_pred == c, 1, 0)
        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)
        h_d, a_sd = img_gt.shape[-1], img_gt.shape[-1]
        if apply_hd or apply_asd:
            if np.sum(gt_c_i) == 0:
                h_d = 0
                a_sd = 0
            elif np.sum(pred_c_i) == 0:
                h_d = -1
                a_sd = -1
                print("Prediction empty for {} {}".format(modality, pat_id))
            else:
                h_d = hd(gt_c_i, pred_c_i) if apply_hd else h_d
                a_sd = asd(gt_c_i, pred_c_i) if apply_asd else a_sd
        res[cls_name] = [dice, h_d, a_sd]
    return res


def metrics_torch(img_gt, img_pred, pat_id, modality, apply_hd=False, apply_asd=False, class_name=("myo", "lv", "rv")):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: torch.tensor
    Array of the ground truth segmentation map.
    img_pred: torch.tensor
    Array of the predicted segmentation map.
    pat_id: int
    The pat which is to be evaluated on (only used for printing error message).
    modality: str
    Modality of the patient (only used for printing error message).
    apply_hd: bool
    Whether to calculate HD.
    apply_asd: bool
    Whether to calculate ASD.
    class_name: array like
    A list/tuple/array of class names

    Return a dictionary of the results for each class, example: {"myo": [], "lv": [], "rv": []}
    ------

    """
    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))
    with torch.no_grad():
        res = {}
        # Loop on each classes of the input images
        for c, cls_name in zip(np.array(range(len(class_name))) + 1, class_name):
            # Copy the gt image to not alternate the input
            gt_c_i = torch.where(img_gt == c, 1, 0)
            # Copy the pred image to not alternate the input
            pred_c_i = torch.where(img_pred == c, 1, 0)
            # Compute the Dice
            dice = torch.sum(2 * gt_c_i * pred_c_i) / torch.sum(pred_c_i + gt_c_i)
            dice = dice.detach().cpu().numpy()
            h_d, a_sd = img_gt.size()[-1], img_gt.size()[-1]
            if apply_hd or apply_asd:
                if torch.sum(gt_c_i) == 0:
                    h_d = 0
                    a_sd = 0
                elif torch.sum(pred_c_i) == 0:
                    print("Prediction empty for {} {}".format(modality, pat_id))
                else:
                    gt_c_i = gt_c_i.detach().cpu().numpy()
                    pred_c_i = pred_c_i.detach().cpu().numpy()
                    h_d = hd(gt_c_i, pred_c_i) if apply_hd else h_d
                    a_sd = asd(gt_c_i, pred_c_i) if apply_asd else a_sd
            res[cls_name] = [dice, h_d, a_sd]
    return res


if __name__ == '__main__':
    pass
