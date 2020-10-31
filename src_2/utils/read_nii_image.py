import glob, os, cv2
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
import random

from timer import timeit
from utils.utilss import to_categorical


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

def save_itk(image, spacing, filename):
    """
    :param image: An MRI volume
    :param spacing: Spacing of volue
    :param filename: Name to be saved for volume
    :return:
    """
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    sitk.WriteImage(itkimage, filename, True)

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


def read_nii_save_3Dnpy(new_spacing=(1.2, 1.2, 7.0), crop_size=224, phase=0, start=None, end=None, volume_size=12, modality=2, plot=False):
    assert modality == 0 or modality == 1 or modality == 2
    assert phase==0 or phase==1
    # 0 represent train, 1 represent validation
    def generate_sample_n_save(img_path, msk_path):
        img_arr, msk_arr = [], []
        for i in range(volume_size):
            img_arr.append(image[pat_index[i]])
            msk_arr.append(masks[pat_index[i]])
        img_arr = np.array(img_arr)
        msk_arr = np.array(msk_arr)
        if plot:
            from skimage.measure import find_contours
            for img, msk in zip(img_arr, msk_arr):
                contour = find_contours(msk, 0.5)
                if len(contour) > 0:
                    contour = np.array(contour[0], dtype=int)
                    fig, ax = plt.subplots()
                    ax.imshow(img, cmap='gray')
                    ax.plot(contour[:, 1], contour[:, 0], color='yellow', lw=2)
                    plt.show()
                plt.imshow(img, cmap='gray')
                plt.show()
            input()
        assert img_arr.shape == (
        12, crop_size, crop_size), "volume shape has to be (12, 224, 224). {} detected instead".format(img_arr.shape)
        assert msk_arr.shape == (
        12, crop_size, crop_size), "mask shape has to be (12, 224, 224). {} detected instead".format(msk_arr.shape)
        np.save(file=img_path, arr=img_arr)
        np.save(file=msk_path, arr=msk_arr)
        print("{} pat_id: {} saved".format(modality, pat_id))
    if phase==0:
        start = 6 if start is None else start
        end = 46 if end is None else end
        data_path = 'trainB' if modality==1 else 'trainA'
    else:
        start = 1 if start is None else start
        end = 6 if end is None else end
        data_path = 'testB' if modality==1 else 'testA'
    data_set = ['C0', 'LGE', 'T2']
    modality = data_set[modality]

    for pat_id in range(start, end):
        img_root_dir = os.path.join('../../input/processed/3D/', data_path)
        msk_root_dir = os.path.join('../../input/processed/3D/', data_path + 'mask')
        if not os.path.isdir(img_root_dir):
            os.mkdir(img_root_dir)
        if not os.path.isdir(msk_root_dir):
            os.mkdir(msk_root_dir)
        print("saving the {}st {} subject".format(pat_id, modality))
        path = "../../input/raw_data/dataset/patient{}_{}.nii.gz".format(pat_id, modality)
        vol = sitk.ReadImage(path)
        vol = sitk.Cast(sitk.RescaleIntensity(vol), sitk.sitkUInt8)
        spacing = np.array(vol.GetSpacing())
        shape = vol.GetSize() # (256, 256, 5)
        multiplier = volume_size / shape[2]
        to_save = True
        # check if the file exists
        if multiplier >= 1:
            img_path = os.path.join(img_root_dir, 'pat_{}_{}.npy'.format(pat_id, modality))
            msk_path = os.path.join(msk_root_dir, 'pat_{}_{}.npy'.format(pat_id, modality))
            to_save = to_save and not (os.path.exists(img_path) and os.path.exists(msk_path))
        else:
            img_path1 = os.path.join(img_root_dir, 'pat_{}_{}_1.npy'.format(pat_id, modality))
            msk_path1 = os.path.join(msk_root_dir, 'pat_{}_{}_1.npy'.format(pat_id, modality))
            img_path2 = os.path.join(img_root_dir, 'pat_{}_{}_2.npy'.format(pat_id, modality))
            msk_path2 = os.path.join(msk_root_dir, 'pat_{}_{}_2.npy'.format(pat_id, modality))
            to_save = to_save and not (os.path.exists(img_path1) and os.path.exists(msk_path1)
                                       and os.path.exists(img_path2) and os.path.exists(msk_path2))
        if to_save:
            resize_factor = spacing / new_spacing
            new_real_shape = shape * resize_factor
            new_shape = np.round(new_real_shape)
            real_resize_factor = new_shape / shape
            real_resize_factor = [1, real_resize_factor[0], real_resize_factor[1]]
            vol1 = sitk.GetArrayFromImage(vol)
            image = ndimage.interpolation.zoom(vol1, real_resize_factor, order=1)
            image = crop_volume(image, crop_size // 2)
            path = "../../input/raw_data/labels/patient{}_{}_manual.nii.gz".format(pat_id, modality)
            vol = sitk.ReadImage(path)
            spacing = np.array(vol.GetSpacing())  # [ 1.25 1.25 12.00000286]
            shape = vol.GetSize()
            vol1 = sitk.GetArrayFromImage(vol)
            vol1 = np.where(vol1 == 200, 1, vol1)
            vol1 = np.where(vol1 == 500, 2, vol1)
            vol1 = np.where(vol1 == 600, 3, vol1)
            vol1 = np.expand_dims(vol1, axis=1)
            masks = to_categorical(vol1, 4)  # (8, 4, 256, 256)
            resize_factor = spacing / new_spacing
            new_real_shape = shape * resize_factor
            new_shape = np.round(new_real_shape)
            real_resize_factor = new_shape / shape
            real_resize_factor = [1, 1, real_resize_factor[0], real_resize_factor[1]]
            masks = ndimage.interpolation.zoom(masks, real_resize_factor, order=1)  # (19 4 267 267)
            masks = np.argmax(masks, axis=1)  # (19, 267, 267)
            masks = crop_volume(masks, crop_size // 2)
            pat_index_subject = [i for i in range(masks.shape[0])]
            multiplier = volume_size / image.shape[0]
            if multiplier >= 1:
                multiplier = int(np.floor(multiplier))
                residual = volume_size - masks.shape[0] * multiplier
                residual = random.sample(pat_index_subject, residual) if residual > 0 else []
                pat_index = pat_index_subject * multiplier + residual
                pat_index.sort()
                generate_sample_n_save(img_path, msk_path)
            elif multiplier < 1:
                pat_index = random.sample(pat_index_subject, volume_size)
                pat_index.sort()
                generate_sample_n_save(img_path1, msk_path1)
                rest = list(set(pat_index_subject) - set(pat_index))
                pat_index = rest + random.sample(pat_index, volume_size - len(rest))
                pat_index.sort()
                generate_sample_n_save(img_path2, msk_path2)
        else:
            print("{} pat_id: {} exists".format(modality, pat_id))


def get_mean_std(new_spacing = (1.2, 1.2, 5.0), crop_size=224, data_to_calculate=0):
    assert data_to_calculate == 0 or data_to_calculate == 1 or data_to_calculate == 2
    data_set = ['C0', 'LGE', 'T2']
    data_choice = data_set[data_to_calculate]
    store_image = []
    for pat_id in range(6, 46):
        path = "../../input/raw_data/dataset/patient{}_".format(pat_id) + data_choice + ".nii.gz"
        vol = sitk.ReadImage(path)
        spacing = np.array(vol.GetSpacing())
        shape = vol.GetSize()
        resize_factor = spacing / new_spacing
        new_real_shape = shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / shape
        # new_spacing = spacing / real_resize_factor
        # print(new_spacing)
        real_resize_factor = [1., real_resize_factor[0], real_resize_factor[1]]
        vol1 = sitk.GetArrayFromImage(vol)
        image = ndimage.interpolation.zoom(vol1, real_resize_factor, order=1)
        image = crop_volume(image, crop_size // 2)
        store_image.extend(image.flatten())

    store_image = np.array(store_image)
    mean = np.mean(store_image)
    std = np.std(store_image)

    return mean, std

def img_to_csv():
    pathes = glob.glob('../../input/processed/C0/*.png')

    df = pd.read_csv("../../input/train_masks.csv")
    for path in pathes:
        path1 = os.path.splitext(os.path.basename(path))[0]
        df_ = pd.DataFrame({"img": [path1]})
        df = df.append(df_)
    df.to_csv("../../input/train_A.csv", index=False)

def npy_img_to_csv(save_path='trainA'):
    pathes = glob.glob('../../input/processed/T2/npy/' + save_path + '/*.npy')

    df = pd.DataFrame()
    for path in pathes:
        path1 = os.path.splitext(os.path.basename(path))[0]
        df_ = pd.DataFrame({"img": [path1]})
        df = df.append(df_)
    df.to_csv("../../input/processed/T2/npy_" + save_path + ".csv", index=False)
    print("finish")

def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)

def graipher(pts, K, dim=2):
    farthest_pts = np.zeros((K, dim))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts

def save_point_cloud(data_choice=0, to_save='', number_points=300, dim=3):
    import mcubes
    dataset = ['C0', 'T2', 'LGE']
    save_name = ['bSSFP', 'T2', 'lge']
    new_spacing = (1.2, 1.2, 1.2)
    crop_size = 224
    print("processing {}".format(dataset[data_choice]))
    for pat_id in range(1, 46):
        path = "../../input/raw_data/labels/patient{}_".format(pat_id) + dataset[data_choice] + "_manual.nii.gz"
        vol = sitk.ReadImage(path)
        spacing = np.array(vol.GetSpacing())
        shape = vol.GetSize()
        vol1 = sitk.GetArrayFromImage(vol)
        # vol1 = vol1[1]
        # vol1 = np.array([vol1] * 3)
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
        masks = np.argmax(masks, axis=1)
        masks = crop_volume(masks, crop_size // 2)
        masks = np.where(masks > 0, 1, 0)
        index = 0
        for m in masks:
            vertices_path = "../../input/processed/vertices/" + dataset[data_choice] + '/pat_{}_'.format(pat_id) + save_name[
                data_choice] + '_{}.npy'.format(index)
            plot_path = "../../input/processed/plot/" + dataset[data_choice] + '/pat_{}_'.format(pat_id) + save_name[
                data_choice] + '_{}.npy'.format(index)
            point_cloud = np.zeros((crop_size, crop_size))
            vertices_array = np.zeros((number_points, dim))
            if m.sum() != 0:
                vol = np.array([m, m, m])
                vol = mcubes.smooth(vol)
                vertices, triangles = mcubes.marching_cubes(vol, 0)
                vertices = graipher(vertices, number_points, dim=dim)
                vertices_array = np.array(vertices, dtype=np.int)
                # fig = plt.figure()
                # from mpl_toolkits.mplot3d import Axes3D
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=10)
                # plt.show()
                point_cloud[vertices_array[:,1], vertices_array[:,2]] = 1

            if to_save=='v' or to_save=='':
                np.save(vertices_path, vertices_array)
            if to_save=='p' or to_save=='':
                np.save(plot_path, point_cloud)
                # mcubes.export_mesh(vertices, triangles, "heart_single_slice.dae", "MyHeart_s")
            index += 1
        print('patient{} finished'.format(pat_id))
    print(dataset[data_choice] + " finish")

@timeit
def save_point_cloud_from_npy(folder='trainAmask/', to_save='v', number_points=300, dim=3):
    import mcubes
    from tqdm import tqdm
    crop_size = 224

    folder_path = os.path.join('../../input/processed/raugmented/', folder)
    for path in tqdm(glob.glob(folder_path + '*.png')):
        filename = os.path.splitext(os.path.basename(path))[0]
        vertices_path = os.path.join("../../input/processed/raugmented/vertices/", folder, filename + '.npy')
        plot_path = os.path.join("../../input/processed/raugmented/plots/", folder, filename + '.npy')
        if not os.path.exists(vertices_path):
            mask = cv2.imread(path)
            # mask = np.where(mask == 85, 1, mask)
            # mask = np.where(mask == 212, 2, mask)
            # mask = np.where(mask == 255, 3, mask)
            mask = np.where(mask > 0, 1, 0)
            mask = np.moveaxis(mask, -1, 0)
            assert mask.shape[0]==3 and mask.shape[1]==224 and mask.ndim==3

            point_cloud = np.zeros((crop_size, crop_size))
            vertices_array = np.zeros((number_points, dim))
            if mask.sum() > 10:
                vol = mcubes.smooth(mask)
                vertices, triangles = mcubes.marching_cubes(vol, 0)
                vertices = graipher(vertices, number_points, dim=dim)
                vertices_array = np.array(vertices, dtype=np.int)
                # fig = plt.figure()
                # from mpl_toolkits.mplot3d import Axes3D
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=10)
                # plt.show()
                point_cloud[vertices_array[:,1], vertices_array[:,2]] = 1

            if to_save=='v' or to_save=='':
                np.save(vertices_path, vertices_array)
            if to_save=='p' or to_save=='':
                np.save(plot_path, point_cloud)
                # mcubes.export_mesh(vertices, triangles, "heart_single_slice.dae", "MyHeart_s")

def generate_point_cloud_3D(to_save=False, number_points=3600, plot=False):
    from tqdm import tqdm
    import mcubes
    folder = ['trainAmask', 'trainBmask', 'testAmask', 'testBmask']
    for fld in tqdm(folder):
        print('start to proceed ' + fld)
        vertices_dir = os.path.join('../../input/processed/3D/vertices3D/', fld)
        if not os.path.isdir(vertices_dir):
            os.makedirs(vertices_dir)
        pathes = glob.glob('../../input/processed/3D/' + fld + '/*.npy')
        for path in tqdm(pathes):
            basename = os.path.basename(path)
            vertices_path = os.path.join(vertices_dir, basename)
            if not os.path.exists(vertices_path):
                mask = np.load(path)
                mask = np.where(mask > 0, 1, 0)
                vertices, triangles = mcubes.marching_cubes(mask, 0)
                vertices = graipher(vertices, number_points, dim=3)
                if plot:
                    fig = plt.figure()
                    from mpl_toolkits.mplot3d import Axes3D
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=10)
                    plt.show()
                if to_save:
                    np.save(vertices_path, vertices)
        print(fld + " finished")
    print('finish')


# def save_point_cloud_3D(to_save=False, number_points=2048):
#     from tqdm import tqdm
#     import mcubes
#     folder = ['trainAmask', 'trainBmask', 'testAmask', 'testBmask']
#     for fld in tqdm(folder):
#         print('start to proceed ' + fld)
#         if not os.path.isdir('../../input/processed/vertices3D/' + fld):
#             os.makedirs('../../input/processed/vertices3D/' + fld)
#         pathes = glob.glob('../../input/processed/npy3D/' + fld + '/*.npy')
#         for path in tqdm(pathes):
#             basename = os.path.basename(path)
#             mask = np.load(path)
#             mask = np.where(mask > 0, 1, 0)
#             vertices, triangles = mcubes.marching_cubes(mask, 0)
#             vertices = graipher(vertices, number_points, dim=3)
#             if to_save and (not os.path.exists('../../input/processed/vertices3D/' + fld + basename)):
#                 np.save('../../input/processed/vertices3D/' + fld + '/' + basename, vertices)
#         print(fld + " finished")

def crop_folder_image(folder='../../input/processed/raugmented/trainBmask/', ext='png'):
    from tqdm import tqdm
    crop_size = 112
    img_paths = glob.glob(folder + '*.' + ext)
    for path in tqdm(img_paths):
        img = cv2.imread(path)
        if img.shape[1] == 256 and img.shape[0] == 256:
            img = img[int(img.shape[0] / 2) - crop_size: int(img.shape[0] / 2) + crop_size,
                  int(img.shape[1] / 2) - crop_size: int(img.shape[1] / 2) + crop_size]
            cv2.imwrite(filename=path, img=img)


def plot_generated_3Ddata(shape=(12, 224, 224), n_sample=5):
    root_dir = '../../input/processed/3D'
    paths = glob.glob(root_dir + '/trainB/*.npy')
    # image = np.load('../../input/processed/3D/trainB/pat_6_LGE.npy')
    # masks = np.load('../../input/processed/3D/trainBmask/pat_6_LGE.npy')
    assert n_sample <= len(paths), "n_sample should be smaller than length of paths. n_sample={}, length of paths={}".format(n_sample, len(paths))
    paths = random.sample(paths, n_sample)
    for path in paths:
        image = np.load(path)
        basename = os.path.basename(path)
        print(basename)
        masks = np.load(os.path.join(root_dir, 'trainBmask', basename))
        assert image.shape == shape
        assert masks.shape == shape
        from skimage.measure import find_contours
        for img, msk in zip(image, masks):
            contour = find_contours(msk, 0.5)
            if len(contour) > 0:
                contour = np.array(contour[0], dtype=int)
                # contour_map = np.zeros((224, 224))
                # contour_map[contour[:,0], contour[:,1]] = 1
                # fig, ax = plt.subplots()
                # ax.imshow(msk, cmap='gray')
                # ax.plot(contour[:,1], contour[:,0], color='yellow', lw=3)
                # plt.show()
                fig, ax = plt.subplots()
                ax.imshow(img, cmap='gray')
                ax.plot(contour[:,1], contour[:,0], color='yellow', lw=2)
                plt.show()
            plt.imshow(img, cmap='gray')
            plt.show()
    input()


def generate_3D_data():
    for modality in range(3):
        for phase in [0, 1]:
            read_nii_save_3Dnpy(phase=phase, modality=modality, plot=False)
    print("finish")


def generate_ctmr_3D_data():
    from numpy import savez_compressed
    ct_test_ids = [1003, 1008, 1014, 1019]
    mr_test_ids = [1007, 1009, 1018, 1019]
    val_ids = [1010, 1020]
    def folder_clsfr(path, ct=True):

        temp = False
        if ct:
            for id in ct_test_ids:
                if str(id) in path: temp=True
            if temp: return 'ct_test'
            else:
                for id in val_ids:
                    if str(id) in path: temp=True
                if temp: return 'ct_val'
                else: return 'ct_train'
        else:
            for id in mr_test_ids:
                if str(id) in path: temp=True
            if temp: return 'mr_test'
            else:
                for id in val_ids:
                    if str(id) in path: temp=True
                if temp: return 'mr_val'
                else: return 'mr_train'
    def process_img(path):
        nimg = nib.load(path)
        image = nimg.get_fdata()
        image = np.moveaxis(image, -1, 0)
        new_shape = np.array([256, 256, 256])
        real_resize_factor = new_shape / image.shape
        image = ndimage.interpolation.zoom(image, real_resize_factor, order=1)
        img_max = image.max()
        image = np.where(image > img_max * 0.98, img_max * 0.98, image)
        return image
    def process_label(path):
        nimg = nib.load(path)
        mask1 = nimg.get_fdata()
        mask1 = np.moveaxis(mask1, -1, 0)
        label = np.zeros((5, *mask1.shape))
        label_list = [0, 205, 420, 500, 820]
        for i, value in enumerate(label_list):
            label[i] = np.where(mask1 == value, 1, 0)
        new_shape = np.array([5, 256, 256, 256])
        real_resize_factor = new_shape / label.shape
        label = ndimage.interpolation.zoom(label, real_resize_factor, order=1)
        label = np.argmax(label, axis=0)
        return label
    # process ct data
    ct_img_paths = glob.glob('../../input/mmwhs_data/ct_train/*image.nii.gz')
    ct_label_paths = glob.glob('../../input/mmwhs_data/ct_train/*label.nii.gz')

    for img_path, label_path in zip(ct_img_paths, ct_label_paths):
        img_basename = os.path.basename(img_path)
        img_name = os.path.splitext(os.path.splitext(img_basename)[0])[0] + '.npz'
        folder = folder_clsfr(img_name, True)
        img_save_path = os.path.join('../../input/PnpAda_release_data/3D/', folder, img_name)
        if not os.path.exists(img_save_path):
            image = process_img(img_path).astype(np.float32)
            label = process_label(label_path).astype(np.int32)
            savez_compressed(img_save_path, image=image, label=label)
            print(img_name + ' saved.')
        else:
            print(img_name + ' exists.')

    # process mr data
    mr_img_paths = glob.glob('../../input/mmwhs_data/mr_train/*image.nii.gz')
    mr_label_paths = glob.glob('../../input/mmwhs_data/mr_train/*label.nii.gz')

    for img_path, label_path in zip(mr_img_paths, mr_label_paths):
        img_basename = os.path.basename(img_path)
        img_name = os.path.splitext(os.path.splitext(img_basename)[0])[0] + '.npz'
        folder = folder_clsfr(img_name, False)
        img_save_path = os.path.join('../../input/PnpAda_release_data/3D/', folder, img_name)
        if not os.path.exists(img_save_path):
            image = process_img(img_path).astype(np.float32)
            label = process_label(label_path).astype(np.int32)
            savez_compressed(img_save_path, image=image, label=label)
            print(img_name + ' saved.')
        else:
            print(img_name + ' exists.')
    print('finish')

if __name__ == "__main__":
    generate_ctmr_3D_data()
    exit()
    zdepth = []
    volume_size = 12
    crop_size = 224
    new_spacing = (1.2,1.2,7.0)
    modality = 'LGE'
    for pat_id in range(6, 46):
    # pat_id = 7
        path = "../../input/raw_data/dataset/patient{}_{}.nii.gz".format(pat_id, modality)
        vol = sitk.ReadImage(path)
        vol = sitk.Cast(sitk.RescaleIntensity(vol), sitk.sitkUInt8)
        spacing = np.array(vol.GetSpacing())
        shape = vol.GetSize()  # (256, 256, 5)
        multiplier = volume_size / shape[2]
        resize_factor = spacing / new_spacing
        new_real_shape = shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / shape
        real_resize_factor = [real_resize_factor[2], real_resize_factor[0], real_resize_factor[1]]
        vol1 = sitk.GetArrayFromImage(vol)
        image = ndimage.interpolation.zoom(vol1, real_resize_factor, order=1)
        image = crop_volume(image, crop_size // 2) #(14, 224, 224)
        print(image.shape[0])
        zdepth.append(image.shape[0])
    print(min(zdepth), max(zdepth))
    # plt.show()
    input()
    # read_bssfp_nii_save_3Dnpy(start=1, end=2)
    # generate_3D_data()
    # plot_generated_3Ddata()
    generate_point_cloud_3D(to_save=True)
    input()
    # save_point_cloud_3D(to_save=True)
    paths = glob.glob('../../input/processed/3D/vertices3D/testBmask/*.npy')
    paths = random.sample(paths, 5)
    for path in paths:
        print(os.path.basename(path))
        vertices = np.load(path)
        fig = plt.figure()
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=10)
        plt.show()
    # save_point_cloud_from_npy()
