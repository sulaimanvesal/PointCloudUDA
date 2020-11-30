import numpy as np
try:
    np.random.bit_generator = np.random._bit_generator
    print("rename numpy.random._bit_generator")
except:
    print("numpy.random.bit_generator exists")
import cv2
import pandas as pd
from skimage.exposure import match_histograms
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid
from utils.npy2point import npy2point_datagenerator
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    GaussNoise, ShiftScaleRotate, Rotate,ShiftScaleRotate
)
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
def augmentation(image, mask=None, segmap=False):
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
    if mask is None:
        image_heavy = seq(images=image)
        return image_heavy
    else:
        if image.ndim == 4:
            mask = np.array(mask)
            image_heavy,mask_heavy = seq(images=image, segmentation_maps=mask)
        else:
            image_heavy,mask_heavy = seq(images=image[np.newaxis,...], segmentation_maps=mask[np.newaxis,...])
            image_heavy,mask_heavy = image_heavy[0], mask_heavy[0]
        return image_heavy,mask_heavy

def light_aug(images, masks=None, lit_prob=False, segmap=True, simple_aug=False):
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    if lit_prob:
        sometimes = lambda aug: iaa.Sometimes(0.3, aug)
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.2),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(iaa.CropAndPad(
                    percent=(-0.03, 0.03),
                    pad_mode='constant',
                    pad_cval=(0, 255)
                )),
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
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 2),
                           [
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.05), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   iaa.CoarseDropout((0.01, 0.05), size_percent=(0.03, 0.05), per_channel=0.2),
                               ]),
                               sometimes(iaa.ElasticTransformation(alpha=(8, 18), sigma=8)),
                               # move pixels locally around (with random strengths)
                               sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.02))),
                               # sometimes move parts of the image around
                               sometimes(iaa.PerspectiveTransform(scale=(0.02, 0.1)))
                           ],
                           random_order=True
                           ),
            ],
            random_order=True
        )
    elif simple_aug:
        sometimes = lambda aug: iaa.Sometimes(0.3, aug)
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.2),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images
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
    else:
        sometimes = lambda aug: iaa.Sometimes(1, aug)
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(.5),  # horizontally flip 50% of all images
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
                iaa.SomeOf((0, 2),
                           [
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
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
    if masks is None:
        image_light = seq(images=images)
        return image_light
    else:
        if segmap:
            segmaps = []
            for mask in masks:
                segmaps.append(SegmentationMapsOnImage(mask.astype(np.int32), shape=images.shape[-3:]))
        else:
            segmaps = np.array(masks, dtype=np.int32)
        image_light,masks = seq(images=images, segmentation_maps=segmaps)
        if segmap:
            mask_light = []
            for mask in masks:
                mask_light.append(mask.get_arr())
            masks = np.array(mask_light)
        return image_light, masks

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


class ImageProcessor:
    @staticmethod
    def augmentation(image, mask, aug=False, noise=False, gn_prob=.5):
        assert isinstance(gn_prob, float), 'gn_prob must be float. Current type {}'.format(type(gn_prob))
        assert gn_prob > 0 and gn_prob <= 1, 'gn_prob must be > 0 and <= 1. Current value: {}'.format(gn_prob)
        image_heavy = image
        mask_heavy = mask
        if aug:
            aug_list = [
                # VerticalFlip(p=0.8),
                # HorizontalFlip(p=0.8),
                # RandomRotate90(p=0.9),
                ElasticTransform(p=.3, alpha_affine=15)
                ]
            aug = Compose(aug_list)
            augmented = aug(image=image_heavy, mask=mask_heavy)
            image_heavy = augmented['image']
            mask_heavy = augmented['mask']
        if noise:
            aug_list = [GaussNoise(var_limit=(0.001, 0.01), p=gn_prob, mean=0)]
            aug = Compose(aug_list)
            img_min, img_max = image.min(), image.max()
            image_heavy = (image_heavy - img_min) / (img_max - img_min)
            image_heavy = aug(image=image_heavy)['image'] * (img_max - img_min) + img_min
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
            self._reference_img = np.load('../input/PnpAda_release_data/ct_train/img/ct_train_slice0.tfrecords.npy')

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
                if not self._offline_aug:
                    img_path = '../input/PnpAda_release_data/mr_train/img/{}.npy'.format(id)
                    mask_path = '../input/PnpAda_release_data/mr_train/mask/{}.npy'.format(id)
                else:
                    # to set the directory of the offline augmented data
                    img_path = ''
                    mask_path = ''
            else:
                img_path = '../input/PnpAda_release_data/mr_val/img/{}.npy'.format(id)
                mask_path = '../input/PnpAda_release_data/mr_val/mask/{}.npy'.format(id)
        else:
            if self._phase == "train":
                img_path = '../input/PnpAda_release_data/ct_train/img/{}.npy'.format(id)
                mask_path = '../input/PnpAda_release_data/ct_train/mask/{}.npy'.format(id)
            else:
                img_path = '../input/PnpAda_release_data/ct_val/img/{}.npy'.format(id)
                mask_path = '../input/PnpAda_release_data/ct_val/mask/{}.npy'.format(id)
        return img_path, mask_path

    def get_images_masks(self, img_path, mask_path):
        img, mask = np.load(img_path), np.load(mask_path)
        return img, np.array(mask, dtype=int)

    def __iter__(self):
        # self._index = 0
        self._totalcount = 0
        return self

    def __next__(self):
        # shuffle image names
        x_batch = []
        y_batch = []

        indices = []
        if self._totalcount >= self._n_samples:
            # self._index = 0
            self._totalcount = 0
            # self._shuffle_indices = np.random.permutation(self._shuffle_indices)
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
            assert mask.ndim == 3

            x_batch.append(img)
            y_batch.append(mask)

        # min-max batch normalisation
        x_batch = np.array(x_batch)
        if self._crop_size:
            x_batch = ImageProcessor.crop_volume(x_batch, crop_size=self._crop_size // 2)
            y_batch = ImageProcessor.crop_volume(np.array(y_batch), crop_size=self._crop_size // 2)
        if self._channel == "channel_first":
            x_batch = np.moveaxis(x_batch, -1, 1)
        y_batch = to_categorical(np.array(y_batch), num_classes=5, channel=self._channel)

        return x_batch, y_batch, 0


class DataGenerator_PointNet:
    def __init__(self, df, channel="channel_first",
                 # apply_noise=False, gn_prob=.5,
                 phase="train",
                 heavy_aug=False,
                 litaug=False,
                 batch_size=16,
                 source="source",
                 crop_size=0,
                 n_samples=-1,
                 toprint=False,
                 match_hist=False,
                 ifvert=False,
                 lit_prob=False,
                 simple_aug=False,
                 segmap=False):
        assert phase == "train" or phase == "valid", r"phase has to be either'train' or 'valid'"
        assert source == "source" or source == "target"
        # assert isinstance(apply_noise, bool), "apply_noise has to be bool"
        self._data = df
        self._len = len(df)
        self._shuffle_indices = np.arange(len(df))
        self._shuffle_indices = np.random.permutation(self._shuffle_indices)
        self._source = source
        # self._apply_noise = apply_noise
        # self._gn_prob = gn_prob
        self._heavy_aug= heavy_aug
        self._litaug = litaug
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
        self._toprint = toprint

        self._match_hist = match_hist
        if match_hist:
            self._reference_img = np.load('../input/PnpAda_release_data/ct_train/img/ct_train_slice0.tfrecords.npy')
        self._vert = ifvert
        self._lit_prob = lit_prob
        self._simple_aug = simple_aug
        self._segmap = segmap

    def __len__(self):
        return self._len

    def get_image_paths(self, id):
        if self._source == "source":
            if self._phase == "train":
                img_path = '../input/PnpAda_release_data/mr_train/img/{}.npy'.format(id)
                mask_path = '../input/PnpAda_release_data/mr_train/mask/{}.npy'.format(id)
                vertex_path = '../input/PnpAda_release_data/mr_train/vertices/{}.npy'.format(id)
            else:
                img_path = '../input/PnpAda_release_data/mr_val/img/{}.npy'.format(id)
                mask_path = '../input/PnpAda_release_data/mr_val/mask/{}.npy'.format(id)
                vertex_path = '../input/PnpAda_release_data/mr_val/vertices/{}.npy'.format(id)
        else:
            if self._phase == "train":
                img_path = '../input/PnpAda_release_data/ct_train/img/{}.npy'.format(id)
                mask_path = '../input/PnpAda_release_data/ct_train/mask/{}.npy'.format(id)
                vertex_path = '../input/PnpAda_release_data/ct_train/vertices/{}.npy'.format(id)
            else:
                img_path = '../input/PnpAda_release_data/ct_val/img/{}.npy'.format(id)
                mask_path = '../input/PnpAda_release_data/ct_val/mask/{}.npy'.format(id)
                vertex_path = '../input/PnpAda_release_data/ct_val/vertices/{}.npy'.format(id)
        return img_path, mask_path, vertex_path

    def get_images_masks(self, img_path, mask_path, vertex_path):
        img, mask = np.load(img_path), np.array(np.load(mask_path), dtype=int)
        if self._vert:
            if not (self._heavy_aug or self._litaug):
                vertex = np.load(vertex_path)
                return img, mask, vertex
        return img, mask, None

    def __iter__(self):
        # self._index = 0
        self._totalcount = 0
        return self

    def __next__(self):
        # shuffle image names
        x_batch = []
        y_batch = []
        z_batch = []
        images, masks, verts = [],[],[]

        indices = []
        if self._totalcount >= self._n_samples:
            # self._index = 0
            self._totalcount = 0
            # self._shuffle_indices = np.random.permutation(self._shuffle_indices)
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
            # plt.imshow(img, cmap='gray')
            # plt.show()
            if self._match_hist:
                img = match_histograms(img, self._reference_img, multichannel=True)


            assert mask.ndim == 3

            images.append(img)
            masks.append(mask)
            verts.append(vertex)
        images = np.array(images)
        if self._heavy_aug or self._litaug:
            img_min = images.min()
            img_max = images.max()
            images = (images - img_min) * 255. / (img_max - img_min)
            images = np.array(images, dtype=np.uint8)
            if self._heavy_aug:
                images, masks = augmentation(images, masks)
            else:
                images, masks = light_aug(images, masks, lit_prob=self._lit_prob, segmap=self._segmap, simple_aug=self._simple_aug)
                # plt.imshow(img, cmap='gray')
                # plt.show()
            images = img_min + images.astype(np.float32) * (img_max - img_min) / 255.
            masks = np.array(masks)
            if self._vert:
                verts = []
                for mask in masks:
                    try:
                        vertex = npy2point_datagenerator(mask)
                        verts.append(vertex)
                    except:
                        print('error when converting mask to pointcloud')
                        exit()

        if self._crop_size:
            images = ImageProcessor.crop_volume(images, crop_size=self._crop_size // 2)
            masks = ImageProcessor.crop_volume(np.array(masks), crop_size=self._crop_size // 2)
        if self._channel == "channel_first":
            images = np.moveaxis(images, -1, 1)
        masks = to_categorical(np.array(masks), num_classes=5, channel=self._channel)
        verts = np.array(verts, np.float32) / 255.

        return images, masks, verts


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # import nibabel as nib
    # from plyfile import PlyData, PlyElement
    # path = '../input/mmwhs_data/ct_train/ct_train_1001_label.nii.gz'
    # nimg = nib.load(path)
    # msk = nimg.get_data()
    # print(np.unique(msk))
    # msk = np.where(msk > 0, 1, 0)
    # import mcubes
    # vertices, triangles = mcubes.marching_cubes(msk, 0)
    # temp = []
    # for f in triangles:
    #     temp.append((f, 255, 255, 255))
    # triangles = np.array(temp, dtype=[('vertex_indices', 'i4', (3,)),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    # elf = PlyElement.describe(triangles, 'face')
    # temp = []
    # for v in vertices:
    #     temp.append(tuple(v))
    # vertices = np.array(temp, dtype=[('x', 'f4'),('y', 'f4'),('z', 'f4')])
    # elv = PlyElement.describe(vertices, 'vertex')
    # PlyData([elv, elf], text=True).write('hearttest_text.ply')
    # 
    # print('end')

    # import glob
    # for path in glob.glob('../input/PnpAda_release_data/ct_train/img/*.npy'):
    #     img = np.load(path)
    #     img = np.array(255 * (img - img.min()) / (img.max() - img.min()), dtype=np.uint8)[...,1]
    #     plt.imshow(img, cmap='gray')
    #     plt.show()
    np.random.seed(0)
    ct_train = ImageProcessor.split_data('./../input/ct_train_list.csv')
    trainB_generator = DataGenerator_PointNet(df=ct_train, channel="channel_first",
                                              phase="train", batch_size=8, source="target",
                                              n_samples=2000, ifvert=True,
                                              heavy_aug=False, litaug=False, lit_prob=True, segmap=False)
    for images, masks, verts in trainB_generator:
        images = 255 * (images - images.min()) / (images.max() - images.min())
        images = np.array(images[:, 1], dtype=np.uint8)
        masks = np.argmax(masks, axis=1)
        for img, msk, vert in zip(images, masks, verts):
            fig = plt.figure(figsize=(16,8))
            fig.add_subplot(1,2,1)
            plt.axis('off')
            plt.imshow(img, cmap='gray')
            fig.add_subplot(1,2,2)
            plt.axis('off')
            plt.imshow(msk, cmap='gray')
            plt.show()
    #
    # img = np.load('../input/PnpAda_release_data/ct_train/img/ct_train_slice12.tfrecords.npy')[:,:,1]
    # img = np.array(255 * (img - img.min()) / (img.max() - img.min()), dtype=np.uint8)
    # mask = np.load('../input/PnpAda_release_data/ct_train/mask/ct_train_slice12.tfrecords.npy')[...,0]
    # mask = np.array(mask, dtype=np.int32)
    # from imgaug.augmentables.segmaps import SegmentationMapsOnImage
    # segmap = SegmentationMapsOnImage(mask, shape=(256,256))
    # for i in range(10):
    #     seq = iaa.Sequential(
    #         [
    #             iaa.Affine(
    #                 scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #                 translate_percent={"x": (-.2,.2), "y": (-0.2,0.2)},
    #                 rotate=(-25,25),  # rotate by -45 to +45 degrees
    #                 shear=(-16,16),  # shear by -16 to +16 degrees
    #                 order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
    #                 cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
    #                 mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
    #             )
    #             # iaa.CropAndPad(
    #             #     percent=(-0.05, 0.1),
    #             #     pad_mode=ia.ALL,
    #             #     pad_cval=(0, 255))
    #             # iaa.PiecewiseAffine(scale=(0.01, 0.02))
    #
    #         ]
    #     )
    #     img_new, mask_new = seq(image=img, segmentation_maps=segmap)
    #     mask_new = mask_new.get_arr()
    #     fig = plt.figure()
    #     fig.add_subplot(1,2,1)
    #     plt.axis('off')
    #     plt.imshow(img, cmap='gray')
    #     fig.add_subplot(1,2,2)
    #     plt.axis('off')
    #     plt.imshow(img_new, cmap='gray')
    #     plt.show()
    #
    #     fig = plt.figure()
    #     fig.add_subplot(1,2,1)
    #     plt.axis('off')
    #     plt.imshow(mask, cmap='gray')
    #     fig.add_subplot(1,2,2)
    #     plt.axis('off')
    #     plt.imshow(mask_new, cmap='gray')
    #     plt.show()
    #
    # print('finish')
    # input()

