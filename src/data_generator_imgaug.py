import numpy as np
try:
    np.random.bit_generator = np.random._bit_generator
    print("rename numpy.random._bit_generator")
except:
    print("numpy.random.bit_generator exists")
import cv2
import pandas as pd
from skimage.exposure import match_histograms
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Compose,
    RandomRotate90,
    ElasticTransform,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    GaussNoise, ShiftScaleRotate, Rotate,ShiftScaleRotate
)
import imgaug as ia
import imgaug.augmenters as iaa

from utils.utilss import to_categorical


class ImageProcessor:
    @staticmethod
    def augmentation(image, mask, noise=False):
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.

        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
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
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
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
                               # search either for all edges or for directed edges,
                               # blend the result with the original image using a blobby mask
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
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                               # either change the brightness of the whole image (sometimes
                               # per channel) or change the brightness of subareas
                               iaa.OneOf([
                                   iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               ]),
                               #iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                               iaa.Grayscale(alpha=(0.0, 1.0)),
                               sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                               # move pixels locally around (with random strengths)
                               sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                               # sometimes move parts of the image around
                               sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        image_heavy,mask_heavy  = seq(images=image, segmentation_maps=mask)
        return image_heavy, mask_heavy\


    @staticmethod
    def augmentation2(image, mask, noise=False):
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.

        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                # iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                # iaa.Flipud(0.2),  # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                # sometimes(iaa.Affine(
                #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                #     # scale images to 80-120% of their size, individually per axis
                #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                #     # translate by -20 to +20 percent (per axis)
                #     rotate=(-45, 45),  # rotate by -45 to +45 degrees
                #     shear=(-16, 16),  # shear by -16 to +16 degrees
                #     order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                #     cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                #     mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                # )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
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
                               # search either for all edges or for directed edges,
                               # blend the result with the original image using a blobby mask
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
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                               # either change the brightness of the whole image (sometimes
                               # per channel) or change the brightness of subareas
                               iaa.OneOf([
                                   iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               ]),
                               #iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                               iaa.Grayscale(alpha=(0.0, 1.0)),
                               # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                               # # move pixels locally around (with random strengths)
                               # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                               # # sometimes move parts of the image around
                               # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        image_heavy,mask_heavy  = seq(images=image, segmentation_maps=mask)
        return image_heavy, mask_heavy

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

        """
        :param vol:
        :return:
        """

        return np.array(vol[:,
                        int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                        int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size, ])


class ImageProcessor2:
    @staticmethod
    def augmentation(image, mask, noise=False):

        aug_list = [
            VerticalFlip(p=0.8),
            HorizontalFlip(p=0.8),
            RandomRotate90(p=0.8),
            #ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            CLAHE(p=1., always_apply=True)]
            #RandomBrightnessContrast(p=0.8),
            #RandomGamma(p=0.8)
        if noise:
            aug_list += [ CLAHE(p=1., always_apply=True)]
        aug = Compose(aug_list)

        augmented = aug(image=image, mask=mask)
        image_heavy = augmented['image']
        mask_heavy = augmented['mask']
        return image_heavy, mask_heavy

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

        """
        :param vol:
        :return:
        """

        return np.array(vol[:,
                        int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                        int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size, ])


class DataGenerator:
    def __init__(self, df, channel="channel_first",
                 apply_noise=False, phase="train",
                 apply_online_aug=True,
                 batch_size=16,
                 height=256,
                 width=256,
                 source="source",
                 crop_size=0,
                 n_samples=-1,
                 offline_aug=False,
                 toprint=False,
                 match_hist=False):
        assert phase == "train" or phase == "valid", r"phase has to be either'train' or 'valid'"
        assert source == "source" or source == "target"
        assert isinstance(apply_noise, bool), "apply_noise has to be bool"
        assert isinstance(apply_online_aug, bool), "apply_online_aug has to be bool"
        self._data = df
        self._len = len(df)
        self._shuffle_indices = np.arange(len(df))
        self._shuffle_indices = np.random.permutation(self._shuffle_indices)
        self._height, self._width = height, width
        self._source = source
        self._apply_aug = apply_online_aug
        self._apply_noise = apply_noise
        self._crop_size = crop_size
        self._phase = phase
        self._channel = channel
        self._batch_size = batch_size
        self._index = 0
        self._totalcount = 0
        if n_samples == -1:
            self._n_samples = len(df)
        else:
            self._n_samples = n_samples
        self._offline_aug = offline_aug
        self._toprint = toprint

        self._match_hist = match_hist
        if match_hist:
            self._reference_img = Compose([CLAHE(always_apply=True)])(image=cv2.imread('./../input_aug/processed/trainA/pat_12_lge_8.png'))["image"]

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
                img_path = './../input_aug/processed/trainA/{}.png'.format(id)
                mask_path = './../input_aug/processed/trainAmask/{}.png'.format(id)
            else:
                img_path = './../input_aug/processed/testA/{}.png'.format(id)
                mask_path = './../input_aug/processed/testAmask/{}.png'.format(id)
        else:
            if self._phase == "train":
                img_path = './../input_aug/processed/trainB/{}.png'.format(id)
                mask_path = './../input_aug/processed/trainBmask/{}.png'.format(id)
            else:
                img_path = './../input_aug/processed/testB/{}.png'.format(id)
                mask_path = './../input_aug/processed/testBmask/{}.png'.format(id)
        return img_path, mask_path

    def get_images_masks(self, img_path, mask_path):
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask == 85, 1, mask)
        mask = np.where(mask == 212, 2, mask)
        mask = np.where(mask == 255, 3, mask)

        return img, mask

    def __iter__(self):
        self._totalcount = 0
        return self

    def __next__(self):
        x_batch = []
        y_batch = []

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
            img_path, mask_path = self.get_image_paths(id=_id)

            img, mask = self.get_images_masks(img_path=img_path, mask_path=mask_path)
            if self._match_hist:
                img = match_histograms(img, self._reference_img, multichannel=True)
            if self._apply_aug:
                img, mask = ImageProcessor.augmentation(img, mask, noise=self._apply_noise)
            # else:
            #     aug = Compose([CLAHE(always_apply=True)])
            #     augmented = aug(image=img, mask=mask)
            #     img, mask = augmented["image"], augmented["mask"]
            mask = np.expand_dims(mask, axis=-1)
            assert mask.ndim == 3

            x_batch.append(img)
            y_batch.append(mask)

        # min-max batch normalisation
        x_batch = np.array(x_batch, np.float32) / 255.
        if self._crop_size:
            x_batch = ImageProcessor.crop_volume(x_batch, crop_size=self._crop_size // 2)
            y_batch = ImageProcessor.crop_volume(np.array(y_batch), crop_size=self._crop_size // 2)
        if self._channel == "channel_first":
            x_batch = np.moveaxis(x_batch, -1, 1)
        y_batch = to_categorical(np.array(y_batch), num_classes=4, channel=self._channel)
        return x_batch, y_batch


class DataGenerator_PointNet:
    def __init__(self, df, channel="channel_first",
                 apply_noise=False, phase="train",
                 apply_online_aug=True,
                 batch_size=16,
                 source="source",
                 crop_size=0,
                 n_samples=-1,
                 offline_aug=False,
                 match_hist=False,
                 aug2=False,
                 rev_data=False,
                 offclahe = True):
        """

        :param df: Dataframe of the path of the data set
        :param channel: choose to use "channel_first" or "channel_last"
        :param apply_noise:
        :param phase: "train" or "valid
        :param apply_online_aug:
        :param batch_size:
        :param source: "source" or "target"
        :param crop_size: the size of the images after cropping
        :param n_samples: number of sample used to train for each epoch
        :param offline_aug:
        :param match_hist:
        :param aug2: controls the type of augmentation
        :param rev_data:
        :param offclahe:
        """
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
        self._rev_data = rev_data
        if n_samples == -1:
            self._n_samples = len(df)
        else:
            self._n_samples = n_samples
        self._offline_aug = offline_aug

        self._match_hist = match_hist
        if match_hist:
            # take the reference image for histogram matching
            if rev_data:
                self._reference_img = cv2.imread('./../input_aug/processed/trainA/pat_9_bSSFP_0.png')
                if offclahe:
                    self._reference_img = Compose([CLAHE(always_apply=True)])(image=self._reference_img)["image"]
            else:
                self._reference_img = cv2.imread('./../input_aug/processed/trainB/pat_12_lge_8.png')
                if offclahe:
                    self._reference_img = Compose([CLAHE(always_apply=True)])(image=self._reference_img)["image"]

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
        # take the pathes for image, ground truth and point cloud
        if self._rev_data:
            if self._source == "source":
                if self._phase == "train":
                    img_path = './../input_aug/rev_data/processed/trainB/{}.png'.format(id)
                    mask_path = './../input_aug/rev_data/processed/trainBmask/{}.png'.format(id)
                    vertex_path = './../input_aug/rev_data/vertices/trainB/{}.npy'.format(id)
                else:
                    img_path = './../input_aug/rev_data/processed/trainB_orig/{}.png'.format(id)
                    mask_path = './../input_aug/rev_data/processed/trainBmask_orig/{}.png'.format(id)
                    vertex_path = './../input_aug/rev_data/vertices/trainB_orig/{}.npy'.format(id)
            else:
                if self._phase == "train":
                    img_path = './../input_aug/rev_data/processed/trainA/{}.png'.format(id)
                    mask_path = './../input_aug/rev_data/processed/trainAmask/{}.png'.format(id)
                    vertex_path = './../input_aug/rev_data/vertices/trainA/{}.npy'.format(id)
                else:
                    img_path = './../input_aug/rev_data/processed/testA/{}.png'.format(id)
                    mask_path = './../input_aug/rev_data/processed/testAmask/{}.png'.format(id)
                    vertex_path = './../input_aug/rev_data/vertices/testA/{}.npy'.format(id)
        else:
            if self._source == "source":
                if self._phase == "train":
                    img_path = './../input_aug/processed/trainA/{}.png'.format(id)
                    mask_path = './../input_aug/processed/trainAmask/{}.png'.format(id)
                    vertex_path = './../input_aug/vertices/trainA/{}.npy'.format(id)
                else:
                    img_path = './../input_aug/processed/testA/{}.png'.format(id)
                    mask_path = './../input_aug/processed/testAmask/{}.png'.format(id)
                    vertex_path = './../input_aug/vertices/testA/{}.npy'.format(id)

            else:
                if self._phase == "train":
                    img_path = './../input_aug/processed/trainB/{}.png'.format(id)
                    mask_path = './../input_aug/processed/trainBmask/{}.png'.format(id)
                    vertex_path = './../input_aug/vertices/trainB/{}.npy'.format(id)
                else:
                    img_path = './../input_aug/processed/trainB_orig/{}.png'.format(id)
                    mask_path = './../input_aug/processed/trainBmask_orig/{}.png'.format(id)
                    vertex_path = './../input_aug/vertices/trainB_orig/{}.npy'.format(id)

        return img_path, mask_path, vertex_path

    # get image, ground truth and point cloud according to the pathes
    def get_images_masks(self, img_path, mask_path, vertex_path):
        img = cv2.imread(img_path)
        # img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_AREA)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask == 85, 1, mask)
        mask = np.where(mask == 212, 2, mask)
        mask = np.where(mask == 255, 3, mask)
        # mask = cv2.resize(mask, (self._width, self._height), interpolation=cv2.INTER_AREA)
        vertex = np.load(vertex_path)

        return img, mask, vertex

    def __iter__(self):
        # self._index = 0
        self._totalcount = 0
        return self

    def __next__(self):
        x_batch = []
        y_batch = []
        z_batch = []
        indices = []
        if self._totalcount >= self._n_samples:
            # self._index = 0
            self._totalcount = 0
            # self._shuffle_indices = np.random.permutation(self._shuffle_indices)
            raise StopIteration
        # compute and save the indices of the images to load
        for i in range(self._batch_size):
            indices.append(self._index)
            self._index += 1
            self._totalcount += 1
            self._index = self._index % self._len
            if self._totalcount >= self._n_samples:
                break
        # load the names of the images
        ids_train_batch = self._data.iloc[self._shuffle_indices[indices]]

        for _id in ids_train_batch.values:
            img_path, mask_path, vertex_path = self.get_image_paths(id=_id)

            img, mask, vertex = self.get_images_masks(img_path=img_path, mask_path=mask_path, vertex_path=vertex_path)
            if self._match_hist:
                img = match_histograms(img, self._reference_img, multichannel=True)
            mask = np.expand_dims(mask, axis=-1)
            assert mask.ndim == 3

            x_batch.append(img)
            y_batch.append(mask)
            z_batch.append(vertex)

        if self._apply_aug:
            if self._aug2:
                # apply the augmentations which have no affine transformation or elastic transformation which will destroy the spatial structure of the images
                x_batch, y_batch = ImageProcessor.augmentation2(np.array(x_batch), np.array(y_batch),
                                                               noise=self._apply_noise)
            else:
                x_batch, y_batch = ImageProcessor.augmentation(np.array(x_batch), np.array(y_batch), noise=self._apply_noise)
        # map [0, 255] -> [0, 1]
        x_batch = np.array(x_batch, np.float32) / 255.
        if self._crop_size:
            # crop the images from 256 by 256 to 224 by 224
            x_batch = ImageProcessor.crop_volume(x_batch, crop_size=self._crop_size // 2)
            y_batch = ImageProcessor.crop_volume(np.array(y_batch), crop_size=self._crop_size // 2)
        if self._channel == "channel_first":
            x_batch = np.moveaxis(x_batch, -1, 1)
        y_batch = to_categorical(np.array(y_batch), num_classes=4, channel=self._channel)
        # map the vertices of point clouds from [0, 255] to [0, 1]
        z_batch = np.array(z_batch, np.float32)/255.
        return x_batch, y_batch, z_batch


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
