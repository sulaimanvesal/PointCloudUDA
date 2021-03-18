import numpy as np
from skimage import measure
import nibabel as nib
import cv2


def to_categorical(mask, num_classes, channel='channel_first'):
    """
    convert ground truth mask to categorical
    :param mask: the ground truth mask
    :param num_classes: the number of classes
    :param channel: 'channel_first' or 'channel_last'
    :return: the categorical mask
    """
    if channel != 'channel_first' and channel != 'channel_last':
        assert False, r"channel should be either 'channel_first' or 'channel_last'"
    assert num_classes > 1, "num_classes should be greater than 1"
    unique = np.unique(mask)
    assert len(unique) <= num_classes, "number of unique values should be smaller or equal to the num_classes"
    assert np.max(unique) < num_classes, "maximum value in the mask should be smaller than the num_classes"
    if mask.shape[1] == 1:
        mask = np.squeeze(mask, axis=1)
    if mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)
    eye = np.eye(num_classes, dtype='uint8')
    output = eye[mask]
    if channel == 'channel_first':
        output = np.moveaxis(output, -1, 1)
    return output


def soft_to_hard_pred(pred, channel_axis=1):
    """
    convert soft prediction to either 1 or 0.
    :param pred: the prediction
    :param channel_axis: the channel axis. For 'channel_first', it should be 1.
    :return: the 'hard' prediction
    """
    max_value = np.max(pred, axis=channel_axis, keepdims=True)
    return np.where(pred == max_value, 1, 0)


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    num_channel = mask.shape[1]
    out_img = np.zeros(mask.shape, dtype=np.uint8)
    for struc_id in range(1, num_channel + 1):

        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)

        props = measure.regionprops(blobs)

        if not props:
            continue

        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label

        out_img[blobs == largest_blob_label] = struc_id

    return out_img


def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everything needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header
    :param img_path: String with the path of the 'nii' or 'nii.gz' image file name.
    :return:Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header


def resize_volume(img_volume, w=256, h=256):
    """
    :param img_volume:
    :return:
    """
    img_res = []
    for im in img_volume:
        img_res.append(cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_AREA))

    return np.array(img_res)
