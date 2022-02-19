import os.path
import numpy as np
try:
    np.random.bit_generator = np.random._bit_generator
    print("rename numpy.random._bit_generator")
except:
    print("numpy.random.bit_generator exists")
import cv2
from glob import glob
import imgaug.augmenters as iaa


def to_categorical(mask, num_classes, channel='channel_first'):
    """
    Convert label into categorical format (one-hot encoded)
    Args:
        mask: The label to be converted
        num_classes: maximum number of classes in the label
        channel: whether the output mask should be 'channel_first' or 'channel_last'

    Returns:
    The one-hot encoded label
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


class ImageProcessor:
    @staticmethod
    def simple_aug(image, mask):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.05), "y": (-0.1, 0.1)},
                    rotate=(-10, 10),  # rotate by -10 to +10 degrees
                    shear=(-12, 12),  # shear by -12 to +12 degrees
                    order=[0, 1],
                    cval=(0, 255),
                    mode='constant'
                )),
            ],
            random_order=True
        )
        if image.ndim == 4:
            mask = np.array(mask)
            image_heavy, mask_heavy = seq(images=image, segmentation_maps=mask.astype(np.int32))
        else:
            image_heavy, mask_heavy = seq(images=image[np.newaxis, ...], segmentation_maps=mask[np.newaxis, ...])
            image_heavy, mask_heavy = image_heavy[0], mask_heavy[0]
        return image_heavy, mask_heavy

    @staticmethod
    def crop_volume(vol, crop_size=112):
        """
        Center crop the input vol into [B, 2 * crop_size, 2 * crop_size, ...]
        :param vol:
        :param crop_size:
        :return:
        """

        return np.array(vol[:,
                        int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                        int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size, ])


class DataGenerator:
    def __init__(self, phase="train", batch_size=8, height=256, width=256, modality="bssfp", crop_size=224,
                 n_samples=-1, toprint=False, augmentation=False, data_dir='../data/mscmrseg'):
        assert modality == "bssfp" or modality == "t2" or modality == 'lge'
        self._height, self._width = height, width
        self._modality = modality
        self._crop_size = crop_size
        self._phase = phase
        self._batch_size = batch_size
        self._index = 0  # start from the 0th sample
        self._totalcount = 0
        self._toprint = toprint
        self._augmentation = augmentation
        if modality == 'bssfp':
            folder = 'bSSFP'
        else:
            folder = modality
        self._image_names = glob(os.path.join(data_dir, 'trainA/*{}*.png'.format(folder)))
        self._mask_names = glob(os.path.join(data_dir, 'trainAmask/*{}*.png'.format(folder)))
        assert len(self._image_names) == len(self._mask_names)
        self._len = len(self._image_names)
        print("{}: {}".format(modality, self._len))
        self._shuffle_indices = np.arange(self._len)
        self._shuffle_indices = np.random.permutation(self._shuffle_indices)
        if n_samples == -1:
            self._n_samples = self._len
        else:
            self._n_samples = n_samples
        self._image_names = np.array(self._image_names)
        self._mask_names = np.array(self._mask_names)

    def __len__(self):
        return self._len

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
        indices_choise = self._shuffle_indices[indices]
        image_name_batch = self._image_names[indices_choise]
        mask_name_batch = self._mask_names[indices_choise]

        for img_name, msk_name in zip(image_name_batch, mask_name_batch):

            img, mask = self.get_images_masks(img_path=img_name, mask_path=msk_name)
            mask = np.expand_dims(mask, axis=-1)
            if self._augmentation:
                img, mask = ImageProcessor.simple_aug(image=img, mask=mask)
            x_batch.append(img)
            y_batch.append(mask)
        # min-max batch normalization
        x_batch = np.array(x_batch, np.float32) / 255.
        if self._crop_size:
            x_batch = ImageProcessor.crop_volume(x_batch, crop_size=self._crop_size // 2)
            y_batch = ImageProcessor.crop_volume(np.array(y_batch), crop_size=self._crop_size // 2)
        x_batch = np.moveaxis(x_batch, -1, 1)
        y_batch = to_categorical(np.array(y_batch), num_classes=4)
        return x_batch, y_batch  # (N, 3, 256, 256) (N, 4, 256, 256)


if __name__ == "__main__":

    def getcolormap():
        from matplotlib.colors import ListedColormap
        colorlist = np.round(
            np.array([[0, 0, 0], [186, 137, 120], [124, 121, 174], [240, 216, 152], [148, 184, 216]]) / 256, decimals=2)
        mycolormap = ListedColormap(colors=colorlist, name='mycolor', N=5)
        return mycolormap


    import matplotlib.pyplot as plt
    bssfp_generator = DataGenerator(phase='train', modality='bssfp', crop_size=224, n_samples=1000, augmentation=True,
                                    data_dir='../../data/mscmrseg')
    for img, msk in bssfp_generator:
        print(img.shape, msk.shape)
        print(img.min(), img.max())
        print(np.argmax(msk,axis=1).min(), np.argmax(msk,axis=1).max())
        for m, gt in zip(img, msk):
            f, plots = plt.subplots(1, 2)
            plots[0].axis('off')
            plots[1].axis('off')
            plots[0].imshow(m[1], cmap='gray')
            plots[1].imshow(np.argmax(gt, axis=0), cmap=getcolormap(), vmin=0, vmax=3)
            plt.show()
        pass
