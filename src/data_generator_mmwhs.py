import numpy as np
import os
try:
    np.random.bit_generator = np.random._bit_generator
    print("rename numpy.random._bit_generator")
except:
    print("numpy.random.bit_generator exists")
import pandas as pd
from skimage.exposure import match_histograms
from utils.npy2point import npy2point_datagenerator
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from utils.utils import to_categorical


def augmentation(image, mask=None):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            iaa.SomeOf((0, 5),
                       [
                           sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),
                               iaa.MedianBlur(k=(3, 11)),
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                           iaa.SimplexNoiseAlpha(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0.5, 1.0)),
                               iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           ])),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                           ]),
                           iaa.Invert(0.05, per_channel=True),
                           iaa.Add((-10, 10), per_channel=0.5),
                           iaa.AddToHueAndSaturation((-20, 20)),
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
    if mask is None:
        image_heavy = seq(images=image)
        return image_heavy
    else:
        if image.ndim == 4:
            mask = np.array(mask)
            image_heavy,mask_heavy = seq(images=image, segmentation_maps=mask.astype(np.int32))
        else:
            image_heavy,mask_heavy = seq(images=image[np.newaxis,...], segmentation_maps=mask[np.newaxis,...])
            image_heavy,mask_heavy = image_heavy[0], mask_heavy[0]
        return image_heavy,mask_heavy


def light_aug(images, masks=None, segmap=False):

    sometimes = lambda aug: iaa.Sometimes(0.3, aug)
    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.2),
            iaa.Flipud(0.2),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.1, 0.05), "y": (-0.1, 0.1)},
                rotate=(-10, 10),
                shear=(-12, 12),
                order=[0, 1],
                cval=(0, 255),
                mode='constant',
            )),
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


class ImageProcessor:

    @staticmethod
    def split_data(img_path):
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
                 phase="train",
                 aug='',
                 batch_size=16,
                 source="source",
                 crop_size=0,
                 n_samples=-1,
                 toprint=False,
                 match_hist=False,
                 ifvert=False,
                 segmap=False,
                 data_dir='../input/PnpAda_release_data/'):
        assert phase == "train" or phase == "valid", r"phase has to be either'train' or 'valid'"
        assert source == "source" or source == "target"
        assert aug == '' or aug == 'heavy' or aug == 'light'
        self._data = df
        self._len = len(df)
        self._shuffle_indices = np.arange(len(df))
        self._shuffle_indices = np.random.permutation(self._shuffle_indices)
        self._source = source
        self._aug = aug
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
        self._segmap = segmap
        self._data_dir = data_dir

    def __len__(self):
        return self._len

    def get_image_paths(self, id):
        if self._source == "source":
            if self._phase == "train":
                img_path = os.path.join(self._data_dir, 'PnpAda_release_data/mr_train/img/{}.npy'.format(id))
                mask_path = os.path.join(self._data_dir, 'PnpAda_release_data/mr_train/mask/{}.npy'.format(id))
                vertex_path = os.path.join(self._data_dir, 'PnpAda_release_data/mr_train/vertices/{}.npy'.format(id))
            else:
                img_path = os.path.join(self._data_dir, 'PnpAda_release_data/mr_val/img/{}.npy'.format(id))
                mask_path = os.path.join(self._data_dir, 'PnpAda_release_data/mr_val/mask/{}.npy'.format(id))
                vertex_path = os.path.join(self._data_dir, 'PnpAda_release_data/mr_val/vertices/{}.npy'.format(id))
        else:
            if self._phase == "train":
                img_path = os.path.join(self._data_dir, 'PnpAda_release_data/ct_train/img/{}.npy'.format(id))
                mask_path = os.path.join(self._data_dir, 'PnpAda_release_data/ct_train/mask/{}.npy'.format(id))
                vertex_path = os.path.join(self._data_dir, 'PnpAda_release_data/ct_train/vertices/{}.npy'.format(id))
            else:
                img_path = os.path.join(self._data_dir, 'PnpAda_release_data/ct_val/img/{}.npy'.format(id))
                mask_path = os.path.join(self._data_dir, 'PnpAda_release_data/ct_val/mask/{}.npy'.format(id))
                vertex_path = os.path.join(self._data_dir, 'PnpAda_release_data/ct_val/vertices/{}.npy'.format(id))
        return img_path, mask_path, vertex_path

    def get_images_masks(self, img_path, mask_path, vertex_path):
        img, mask = np.load(img_path), np.array(np.load(mask_path), dtype=int)
        if self._vert:
            if not ((self._aug == 'heavy') or (self._aug == 'light')):
                vertex = np.load(vertex_path)
                return img, mask, vertex
        return img, mask, None

    def __iter__(self):
        self._totalcount = 0
        return self

    def __next__(self):
        images, masks, verts = [],[],[]

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
            if self._match_hist:
                img = match_histograms(img, self._reference_img, multichannel=True)

            assert mask.ndim == 3

            images.append(img)
            masks.append(mask)
            verts.append(vertex)
        images = np.array(images)
        if self._aug == 'heavy' or self._aug == 'light':
            img_min = images.min()
            img_max = images.max()
            images = (images - img_min) * 255. / (img_max - img_min)
            images = np.array(images, dtype=np.uint8)
            if self._aug == 'heavy':
                images, masks = augmentation(images, masks)
            else:
                images, masks = light_aug(images, masks, segmap=self._segmap)
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
    pass
