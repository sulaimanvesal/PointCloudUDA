import numpy as np
try:
    np.random.bit_generator = np.random._bit_generator
    print("rename numpy.random._bit_generator")
except:
    print("numpy.random.bit_generator exists")
import cv2
import pandas as pd
import os

import imgaug as ia
import imgaug.augmenters as iaa

from utils.utils import to_categorical
from utils.npy2point import npy2point_datagenerator


class ImageProcessor:
    @staticmethod
    def augmentation(image, mask):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                iaa.SomeOf((0, 5),
                           [
                               sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                               # convert images into their superpixel representation
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                               iaa.SimplexNoiseAlpha(iaa.OneOf([
                                   iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                   iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                               ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               iaa.Invert(0.05, per_channel=True),  # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),
                               iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                               iaa.OneOf([
                                   iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               ]),
                               iaa.Grayscale(alpha=(0.0, 1.0)),
                               sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                               sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                               sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        image_heavy,mask_heavy = seq(images=image, segmentation_maps=mask)
        return image_heavy, mask_heavy

    @staticmethod
    def augmentation2(image, mask):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
            [
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                iaa.SomeOf((0, 5),
                           [
                               sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                               # convert images into their superpixel representation
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                               iaa.SimplexNoiseAlpha(iaa.OneOf([
                                   iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                   iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                               ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               iaa.Invert(0.05, per_channel=True),  # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),
                               iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                               iaa.OneOf([
                                   iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               ]),
                               iaa.Grayscale(alpha=(0.0, 1.0)),
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        image_heavy,mask_heavy = seq(images=image, segmentation_maps=mask)
        return image_heavy, mask_heavy

    @staticmethod
    def simple_aug(image, mask):
        sometimes = lambda aug: iaa.Sometimes(0.45, aug)
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.3),  # horizontally flip 50% of all images
                iaa.Flipud(0.3),  # vertically flip 20% of all images
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.05), "y": (-0.1, 0.1)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-10, 10),  # rotate by -45 to +45 degrees
                    shear=(-12, 12),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode='constant'  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
            ],
            random_order=True
        )
        if mask is None:
            image_light = seq(images=image)
            return image_light
        else:
            segmaps = np.array(mask, dtype=np.int32)
            if image.ndim == 4:
                image_light, masks = seq(images=image, segmentation_maps=segmaps)
            else:
                image_light, masks = seq(images=image[np.newaxis,...], segmentation_maps=segmaps[np.newaxis,...])
                image_light = image_light[0]
                masks = masks[0]
            return image_light, masks

    @staticmethod
    def split_data(img_path):
        """
        Load train csv file and split the data into train and validation!
        :return:
        """
        df_train = pd.read_csv(img_path)
        ids_train = df_train['img']
        return ids_train

    @staticmethod
    def crop_volume(vol, crop_size=112):
        return np.array(vol[:,
                        int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                        int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size, ])


class DataGenerator_PointNet:
    def __init__(self, df, channel="channel_first",
                 apply_noise=False, phase="train",
                 apply_online_aug=True,
                 batch_size=16,
                 source="source",
                 crop_size=0,
                 n_samples=-1,
                 offline_aug=False,
                 toprint=False,
                 aug2=False,
                 data_dir='./../input_aug/processed/'):
        assert phase == "train" or phase == "valid", r"phase has to be either'train' or 'valid'"
        assert source == "source" or source == "target"
        assert isinstance(apply_noise, bool), "apply_noise has to be bool"
        assert isinstance(apply_online_aug, bool), "apply_online_aug has to be bool"
        self._data = df
        self._len = len(df)
        self._shuffle_indices = np.arange(len(df))
        self._shuffle_indices = np.random.permutation(self._shuffle_indices)
        self._source = source
        self._apply_aug = apply_online_aug
        self._apply_noise = apply_noise
        self._crop_size = crop_size
        self._phase = phase
        self._channel = channel
        self._batch_size = batch_size
        self._index = 0
        self._totalcount = 0
        self._aug2 = aug2
        if n_samples == -1:
            self._n_samples = len(df)
        else:
            self._n_samples = n_samples
        self._offline_aug = offline_aug
        self._toprint = toprint
        self._data_dir = data_dir

    def __len__(self):
        return self._len

    @property
    def apply_aug(self):
        return self._apply_aug

    @apply_aug.setter
    def apply_aug(self, aug):
        assert isinstance(aug, bool), "apply_aug has to be bool"
        self._apply_aug = aug

    def get_image_paths(self, id):
        if self._source == "source":
            if self._phase == "train":
                img_path = os.path.join(self._data_dir, 'processed/trainA/{}.png'.format(id))
                mask_path = os.path.join(self._data_dir, 'processed/trainAmask/{}.png'.format(id))
                vertex_path = os.path.join(self._data_dir, 'vertices/trainA/{}.npy'.format(id))
            else:
                img_path = os.path.join(self._data_dir, 'processed/testA/{}.png'.format(id))
                mask_path = os.path.join(self._data_dir, 'processed/testAmask/{}.png'.format(id))
                vertex_path = os.path.join(self._data_dir, 'vertices/testA/{}.npy'.format(id))

        else:
            if self._phase == "train":
                img_path = os.path.join(self._data_dir, 'processed/trainB/{}.png'.format(id))
                mask_path = os.path.join(self._data_dir, 'processed/trainBmask/{}.png'.format(id))
                vertex_path = os.path.join(self._data_dir, 'vertices/trainB/{}.npy'.format(id))
            else:
                img_path = os.path.join(self._data_dir, 'processed/trainB_orig/{}.png'.format(id))
                mask_path = os.path.join(self._data_dir, 'processed/trainBmask_orig/{}.png'.format(id))
                vertex_path = os.path.join(self._data_dir, 'vertices/trainB_orig/{}.npy'.format(id))

        return img_path, mask_path, vertex_path

    def get_images_masks(self, img_path, mask_path, vertex_path):
        img = cv2.imread(img_path)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask == 85, 1, mask)
        mask = np.where(mask == 212, 2, mask)
        mask = np.where(mask == 255, 3, mask)
        vertex = np.load(vertex_path)

        return img, mask, vertex

    def __iter__(self):
        self._totalcount = 0
        return self

    def __next__(self):
        x_batch = []
        y_batch = []
        z_batch = []


        indices = []
        if self._totalcount >= self._n_samples:
            self._totalcount = 0
            raise StopIteration
        for i in range(self._batch_size):
            indices.append(self._index)
            self._index += 1
            self._totalcount += 1
            self._index = self._index % self._len
            if self._totalcount >= self._n_samples:
                break
        ids_train_batch = self._data.iloc[self._shuffle_indices[indices]]

        for _id in ids_train_batch.values:
            img_path, mask_path, vertex_path = self.get_image_paths(id=_id)

            img, mask, vertex = self.get_images_masks(img_path=img_path, mask_path=mask_path, vertex_path=vertex_path)
            mask = np.expand_dims(mask, axis=-1)
            assert mask.ndim == 3

            x_batch.append(img)
            y_batch.append(mask)
            z_batch.append(vertex)

        # min-max batch normalisation
        if self._apply_aug:
            if self._aug2:
                x_batch, y_batch = ImageProcessor.augmentation2(np.array(x_batch), np.array(y_batch))
            else:
                x_batch, y_batch = ImageProcessor.augmentation(np.array(x_batch), np.array(y_batch))
        x_batch = np.array(x_batch, np.float32) / 255.
        if self._crop_size:
            x_batch = ImageProcessor.crop_volume(x_batch, crop_size=self._crop_size // 2)
            y_batch = ImageProcessor.crop_volume(np.array(y_batch), crop_size=self._crop_size // 2)
        if self._channel == "channel_first":
            x_batch = np.moveaxis(x_batch, -1, 1)
        y_batch = to_categorical(np.array(y_batch), num_classes=4, channel=self._channel)
        z_batch = np.array(z_batch, np.float32)/255.

        return x_batch, y_batch, z_batch


def get_images_masks(img_paths, mask_paths, crop_size=224, aug=False):
    imgs, masks = [], []
    for img_path, mask_path in zip(img_paths, mask_paths):
        img = cv2.imread(img_path)
        # img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_AREA)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask == 85, 1, mask)
        mask = np.where(mask == 212, 2, mask)
        mask = np.where(mask == 255, 3, mask)
        # mask = cv2.resize(mask, (self._width, self._height), interpolation=cv2.INTER_AREA)
        imgs.append(img)
        masks.append(mask)
    imgs = np.array(imgs, dtype=np.float32) / 255.
    masks = np.array(masks)
    if crop_size:
        imgs = ImageProcessor.crop_volume(imgs, crop_size=crop_size // 2)
        masks = ImageProcessor.crop_volume(masks, crop_size=crop_size // 2)
    imgs = np.moveaxis(imgs, -1, 1)
    masks = to_categorical(np.array(masks), num_classes=4)

    return imgs, masks


def reconstuct_volume(vol, crop_size=112, origin_size=256, num_class=4):
    """
    :param vol:
    :return:
    """
    recon_vol = np.zeros((len(vol), origin_size, origin_size, num_class), dtype=np.float32)

    recon_vol[:,
    int(recon_vol.shape[1] / 2) - crop_size: int(recon_vol.shape[1] / 2) + crop_size,
    int(recon_vol.shape[2] / 2) - crop_size: int(recon_vol.shape[2] / 2) + crop_size, :] = vol

    return recon_vol


if __name__ == "__main__":
    ids_train = ImageProcessor.split_data("./../input_aug/aug_trainA.csv")
    ids_valid = ImageProcessor.split_data("./../input_aug/testA.csv")
    ids_train_lge = ImageProcessor.split_data('./../input_aug/aug_trainB.csv')
    ids_valid_lge = ImageProcessor.split_data("./../input_aug/testB.csv")
    source_train_generator = DataGenerator_PointNet(df=ids_train, batch_size=8, phase="train", apply_online_aug=True, source="source", crop_size=224, n_samples=100)
    for dataA in source_train_generator:
        imgA, maskA, _ = dataA
        pass

    dataset = []
    temp = 0
    print(len(source_train_generator))
    generator_iter = iter(source_train_generator)
    for img, mask in generator_iter:
        dataset.extend(img)
        d = np.array(dataset).shape
        temp = np.array(dataset).shape
        pass

    print(temp)
