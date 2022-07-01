import cv2
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from scipy import ndimage


def to_categorical(mask, num_classes, channel='channel_first'):
    assert mask.ndim == 4, "mask should have 4 dims"
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


def crop_volume(vol, crop_size=112):
    """
    :param vol:
    :return:
    """

    return np.array(vol[:,
                    int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                    int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size, ])


def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header


def preprocess_volume(img_volume):

    """
    :param img_volume: A patient volume
    :return: applying CLAHE and Bilateral filter for contrast enhacnmeent and denoising

    """
    prepross_imgs = []
    for i in range(len(img_volume)):
        img = img_volume[i]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        cl1 = clahe.apply(img)
        prepross_imgs.append(cl1)

    return np.array(prepross_imgs)


def resize_volume(img_volume, w=288, h=288):
    """
    :param img_volume:
    :return:
    """
    img_res = []
    for im in img_volume:
        img_res.append(cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_NEAREST))

    return np.array(img_res)


def read_lge_nii_save_png(crop_size=224):

    for pat_id in range(1,46):
        print("saving the {}st lge subject".format(pat_id))
        path = "../../input/raw_data/dataset/patient{}_LGE.nii.gz".format(pat_id)
        vol = sitk.ReadImage(path)
        vol = sitk.Cast(sitk.RescaleIntensity(vol), sitk.sitkUInt8)
        vol = sitk.GetArrayFromImage(vol)
        if vol.shape[1] != 256 or vol.shape[2] != 256:
            vol = resize_volume(vol, w=256, h=256)
        vol = crop_volume(vol, crop_size//2)
        vol = preprocess_volume(vol)
        l =0
        for m in vol:
            cv2.imwrite(filename='../../input/processed/lge_img/pat_{}_lge_{}.png'.format(pat_id,l), img=m)
            l += 1
    print("finish")


def read_lge_nii_label_save_png(crop_size=224):
    for pat_id in range(1,46):
        print("saving the {}st lge subject".format(pat_id))
        path = "../../input/raw_data/labels/patient{}_LGE_manual.nii.gz".format(pat_id)
        vol = sitk.ReadImage(path)
        vol = sitk.Cast(sitk.RescaleIntensity(vol), sitk.sitkUInt8)
        vol = sitk.GetArrayFromImage(vol)
        if vol.shape[1]!=256 or vol.shape[2]!=256:
            vol = resize_volume(vol, w=256, h=256)
        vol = crop_volume(vol, crop_size // 2)
        # vol = preprocess_volume(vol)
        l =0
        for m in vol:
            cv2.imwrite(filename='../../input/processed/lge_label/pat_{}_lge_{}.png'.format(pat_id,l), img=m)
            l +=1
    print("finish")


def read_t2_nii_save_png(crop_size=224):

    for pat_id in range(1,46):
        print("saving the {}st t2 subject".format(pat_id))
        path = "../../input/raw_data/dataset/patient{}_T2.nii.gz".format(pat_id)
        vol = sitk.ReadImage(path)
        vol = sitk.Cast(sitk.RescaleIntensity(vol), sitk.sitkUInt8)
        vol = sitk.GetArrayFromImage(vol)
        if vol.shape[1]!=256 or vol.shape[2]!=256:
            vol = resize_volume(vol, w=256, h=256)
        vol = crop_volume(vol, crop_size // 2)
        vol = preprocess_volume(vol)
        l =0
        for m in vol:
            cv2.imwrite(filename='../../input/processed/t2_img/pat_{}_T2_{}.png'.format(pat_id,l), img=m)
            l +=1
    print("finish")


def read_t2_nii_label_save_png(crop_size=224):

    for pat_id in range(1,46):
        print("saving the {}st t2 subject".format(pat_id))
        path = "../../input/raw_data/labels/t2gt/patient{}_T2_manual.nii.gz".format(pat_id)
        vol = sitk.ReadImage(path)
        vol = sitk.Cast(sitk.RescaleIntensity(vol), sitk.sitkUInt8)
        vol = sitk.GetArrayFromImage(vol)
        if vol.shape[1] != 256 or vol.shape[2] != 256:
            vol = resize_volume(vol, w=256, h=256)
        vol = crop_volume(vol, crop_size // 2)
        # vol = preprocess_volume(vol)
        # vol_unique = np.unique(vol)
        l =0
        for m in vol:
            cv2.imwrite(filename='../../input/processed/t2_label/pat_{}_T2_{}.png'.format(pat_id,l), img=m)
            l +=1
    print("finish")


def read_bssfp_nii_save_png(crop_size=224):

    for pat_id in range(1,46):
        print("saving the {}st bssfp subject".format(pat_id))
        path = "../../input/raw_data/dataset/patient{}_C0.nii.gz".format(pat_id)
        vol = sitk.ReadImage(path)
        vol = sitk.Cast(sitk.RescaleIntensity(vol), sitk.sitkUInt8)
        vol = sitk.GetArrayFromImage(vol)
        if vol.shape[1]!=256 or vol.shape[2]!=256:
            vol = resize_volume(vol, w=256, h=256)
        vol = crop_volume(vol, crop_size // 2)
        vol = preprocess_volume(vol)
        l =0
        for m in vol:
            cv2.imwrite(filename='../../input/processed/bssfp_img/pat_{}_bSSFP_{}.png'.format(pat_id,l), img=m)
            l +=1
    print("finish")


def read_bssfp_nii_label_save_png(crop_size=224):

    for pat_id in range(1,46):
        print("saving the {}st bssfp subject".format(pat_id))
        path = "../../input/raw_data/labels/patient{}_C0_manual.nii.gz".format(pat_id)
        vol = sitk.ReadImage(path)
        vol = sitk.Cast(sitk.RescaleIntensity(vol), sitk.sitkUInt8)
        vol = sitk.GetArrayFromImage(vol)
        if vol.shape[1]!=256 or vol.shape[2]!=256:
            vol = resize_volume(vol, w=256, h=256)
        vol = crop_volume(vol, crop_size // 2)
        l =0
        for m in vol:
            cv2.imwrite(filename='../../input/processed/bssfp_label/pat_{}_bSSFP_{}.png'.format(pat_id,l), img=m)
            l +=1
    print("finish")


def read_lge_nii_save_npy(new_spacing=(1.2, 1.2, 5.0), crop_size=224, phase='train'):
    assert phase=='train' or phase=='valid'
    if phase=='train':
        start = 6
        end = 46
        data_path = 'trainB'
    else:
        start = 1
        end = 6
        data_path = 'testB'
    for pat_id in range(start, end):
        print("saving the {}st lge subject".format(pat_id))
        path = "../../input/raw_data/dataset/patient{}_LGE.nii.gz".format(pat_id)
        vol = sitk.ReadImage(path)
        spacing = np.array(vol.GetSpacing())
        shape = vol.GetSize()
        resize_factor = spacing / new_spacing
        new_real_shape = shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / shape
        real_resize_factor = [1, real_resize_factor[0], real_resize_factor[1]]
        vol1 = sitk.GetArrayFromImage(vol)
        image = ndimage.interpolation.zoom(vol1, real_resize_factor, order=1)
        image = crop_volume(image, crop_size // 2)
        image = (image - image.mean()) / image.std()
        l = 0
        for m in image:
            np.save(file='../../input/processed/npy/' + data_path + '/pat_{}_lge_{}.npy'.format(pat_id, l), arr=m)
            l += 1
    print("finish")


def read_lge_nii_label_save_npy(new_spacing = (1.2, 1.2, 5.), crop_size=224, phase='train'):
    assert phase == 'train' or phase == 'valid'
    if phase == 'train':
        start = 6
        end = 46
        data_path = 'trainBmask'
        image_path = 'lge_test_gt'
    else:
        start = 1
        end = 6
        data_path = 'testBmask'
        image_path = 'lgegt'
    for pat_id in range(start, end):
        print("saving the {}st lge subject".format(pat_id))
        path = "../../input/raw_data/labels/" + image_path + "/patient{}_LGE_manual.nii.gz".format(pat_id)
        vol = sitk.ReadImage(path)
        spacing = np.array(vol.GetSpacing())  # [ 1.25 1.25 12.00000286]
        shape = vol.GetSize()
        vol1 = sitk.GetArrayFromImage(vol)
        vol1 = np.where(vol1 == 200, 1, vol1)
        vol1 = np.where(vol1 == 500, 2, vol1)
        vol1 = np.where(vol1 == 600, 3, vol1)
        vol1 = np.expand_dims(vol1, axis=1)
        labels = to_categorical(vol1, 4)  # (8, 4, 256, 256)
        resize_factor = spacing / new_spacing
        new_real_shape = shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / shape
        real_resize_factor = [1, 1, real_resize_factor[0], real_resize_factor[1]]

        masks = ndimage.interpolation.zoom(labels, real_resize_factor, order=1)  # (19 4 267 267)
        masks = np.argmax(masks, axis=1)  # (19, 267, 267)
        masks = crop_volume(masks, crop_size // 2)
        l = 0
        for m in masks:
            np.save(file='../../input/processed/npy/' + data_path + '/pat_{}_lge_{}.npy'.format(pat_id, l), arr=m)
            l += 1
    print("finish")


def read_bssfp_nii_save_npy(new_spacing=(1.2, 1.2, 5.0), crop_size=224, phase='train'):
    assert phase=='train' or phase=='valid'
    if phase=='train':
        start = 6
        end = 46
        data_path = 'trainA'
    else:
        start = 1
        end = 6
        data_path = 'testA'
    for pat_id in range(start, end):
        print("saving the {}st bssfp subject".format(pat_id))
        path = "../../input/raw_data/dataset/patient{}_C0.nii.gz".format(pat_id)
        vol = sitk.ReadImage(path)
        spacing = np.array(vol.GetSpacing())
        shape = vol.GetSize()
        resize_factor = spacing / new_spacing
        new_real_shape = shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / shape
        real_resize_factor = [1, real_resize_factor[0], real_resize_factor[1]]
        vol1 = sitk.GetArrayFromImage(vol)
        image = ndimage.interpolation.zoom(vol1, real_resize_factor, order=1)
        image = crop_volume(image, crop_size // 2)
        image = (image - image.mean()) / image.std()
        l = 0
        for m in image:
            np.save(file='../../input/processed/npy/' + data_path + '/pat_{}_bSSFP_{}.npy'.format(pat_id, l), arr=m)
            l += 1
    print("finish")


def read_bssfp_nii_label_save_npy(new_spacing = (1.2, 1.2, 5.), crop_size=224, phase='train'):
    assert phase == 'train' or phase == 'valid'
    if phase=='train':
        start = 6
        end = 46
        data_path = 'trainAmask'
    else:
        start = 1
        end = 6
        data_path = 'testAmask'
    for pat_id in range(start, end):
        print("saving the {}st bssfp subject".format(pat_id))
        path = "../../input/raw_data/labels/c0gt/patient{}_C0_manual.nii.gz".format(pat_id)
        vol = sitk.ReadImage(path)
        spacing = np.array(vol.GetSpacing()) # [ 1.25 1.25 12.00000286]
        shape = vol.GetSize()
        vol1 = sitk.GetArrayFromImage(vol)
        vol1 = np.where(vol1==200, 1, vol1)
        vol1 = np.where(vol1==500, 2, vol1)
        vol1 = np.where(vol1==600, 3, vol1)
        vol1 = np.expand_dims(vol1, axis=1)
        labels = to_categorical(vol1, 4) # (8, 4, 256, 256)
        resize_factor = spacing / new_spacing
        new_real_shape = shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / shape
        real_resize_factor = [1, 1, real_resize_factor[0], real_resize_factor[1]]

        masks = ndimage.interpolation.zoom(labels, real_resize_factor, order=1) # (19 4 267 267)
        masks = np.argmax(masks, axis=1) # (19, 267, 267)
        masks = crop_volume(masks, crop_size // 2)
        l =0
        for m in masks:
            np.save(file='../../input/processed/npy/' + data_path + '/pat_{}_bSSFP_{}.npy'.format(pat_id,l), arr=m)
            l +=1
    print("finish")


def read_t2_nii_save_npy(new_spacing=(1.2, 1.2, 5.0), crop_size=224, phase='train'):
    assert phase=='train' or phase=='valid'
    if phase=='train':
        start = 6
        end = 46
        data_path = 'trainA'
    else:
        start = 1
        end = 6
        data_path = 'testA'
    for pat_id in range(start, end):
        print("saving the {}st T2 subject".format(pat_id))
        path = "../../input/raw_data/dataset/patient{}_T2.nii.gz".format(pat_id)
        vol = sitk.ReadImage(path)
        spacing = np.array(vol.GetSpacing())
        shape = vol.GetSize()
        resize_factor = spacing / new_spacing
        new_real_shape = shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / shape
        real_resize_factor = [1, real_resize_factor[0], real_resize_factor[1]]
        vol1 = sitk.GetArrayFromImage(vol)
        image = ndimage.interpolation.zoom(vol1, real_resize_factor, order=1)
        image = crop_volume(image, crop_size // 2)
        image = (image - image.mean()) / image.std()
        l = 0
        for m in image:
            np.save(file='../../input/processed/npy/' + data_path + '/pat_{}_T2_{}.npy'.format(pat_id, l), arr=m)
            l += 1
    print("finish")


def read_t2_nii_label_save_npy(new_spacing = (1.2, 1.2, 5.), crop_size=224, phase='train'):
    assert phase == 'train' or phase == 'valid'
    if phase=='train':
        start = 6
        end = 46
        data_path = 'trainAmask'
    else:
        start = 1
        end = 6
        data_path = 'testAmask'
    for pat_id in range(start, end):
        print("saving the {}st T2 subject".format(pat_id))
        path = "../../input/raw_data/labels/t2gt/patient{}_T2_manual.nii.gz".format(pat_id)
        vol = sitk.ReadImage(path)
        spacing = np.array(vol.GetSpacing()) # [ 1.25 1.25 12.00000286]
        shape = vol.GetSize()
        vol1 = sitk.GetArrayFromImage(vol)
        vol1 = np.where(vol1==200, 1, vol1)
        vol1 = np.where(vol1==500, 2, vol1)
        vol1 = np.where(vol1==600, 3, vol1)
        vol1 = np.expand_dims(vol1, axis=1)
        labels = to_categorical(vol1, 4) # (8, 4, 256, 256)
        resize_factor = spacing / new_spacing
        new_real_shape = shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / shape
        real_resize_factor = [1, 1, real_resize_factor[0], real_resize_factor[1]]

        masks = ndimage.interpolation.zoom(labels, real_resize_factor, order=1) # (19 4 267 267)
        masks = np.argmax(masks, axis=1) # (19, 267, 267)
        masks = crop_volume(masks, crop_size // 2)
        l =0
        for m in masks:
            np.save(file='../../input/processed/T2/npy/' + data_path + '/pat_{}_T2_{}.npy'.format(pat_id,l), arr=m)
            l +=1
    print("finish")


if __name__ == "__main__":
    pass
