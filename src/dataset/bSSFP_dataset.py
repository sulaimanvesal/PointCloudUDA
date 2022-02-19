import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
import glob
import os
import imgaug.augmenters as iaa


class ImageProcessor:
    @staticmethod
    def simple_aug(image, mask):
        """
        apply affine transformation to the data as data augmentation
        Args:
            image:
            mask:

        Returns:
        Augmented images and labels
        """
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.05), "y": (-0.1, 0.1)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-10, 10),  # rotate by -10 to +10 degrees
                    shear=(-12, 12),  # shear by -12 to +12 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode='constant'  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
            ],
            random_order=True
        )
        if image.ndim == 4:
            mask = np.array(mask)
            image_heavy, mask_heavy = seq(images=image, segmentation_maps=mask.astype(np.int32))
        else:
            image_heavy, mask_heavy = seq(images=image[np.newaxis, ...],
                                          segmentation_maps=mask[np.newaxis, ...].astype(np.int32))
            image_heavy, mask_heavy = image_heavy[0], mask_heavy[0]
        return image_heavy, mask_heavy


class bSSFPDataSet(data.Dataset):
    def __init__(self, list_path, max_iters=None, crop_size=224, length=None, t2=False):
        self._length = length
        self.list_path = list_path
        self.crop_size = crop_size
        self.img_ids = glob.glob(os.path.join(list_path, "trainA/*bSSFP*.png"))
        self.label_ids = glob.glob(os.path.join(list_path, "trainAmask/*bSSFP*.png"))
        if t2:
            self.img_ids = self.img_ids + glob.glob(os.path.join(list_path, "trainA/*t2*.png"))
            self.label_ids = self.label_ids + glob.glob(os.path.join(list_path, "trainAmask/*t2*.png"))
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))
        self.files = []
        self.id_to_trainid = {0: 0, 85: 1, 212: 2, 255: 3}
        for img_file, label_file in zip(self.img_ids, self.label_ids):
            name = os.path.basename(img_file)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files) if self._length is None else self._length

    def __getitem__(self, index):
        datafiles = self.files[index % len(self.files)]
        image = Image.open(datafiles["img"]).convert('RGB')  # H,W,C
        img_w, img_h = image.size
        label = Image.open(datafiles["label"])
        if img_w != self.crop_size:
            border_size = int((img_w - self.crop_size) // 2)
            image = image.crop((border_size, border_size, img_w - border_size, img_h - border_size))
            label = label.crop((border_size, border_size, img_w - border_size, img_h - border_size))
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        label_copy = np.asarray(label, np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label_copy == k] = v
        label_copy = np.expand_dims(label_copy, axis=-1)
        image, label_copy = ImageProcessor.simple_aug(image=image, mask=label_copy)
        label_copy = label_copy[..., 0]  # H,W
        image = image[:, :, ::-1]
        image = image / 255.0
        image = image.transpose((2, 0, 1))  # C,H,W
        return image.copy(), label_copy.copy(), name


def get_bssfp_dataloader(scratch, args, input_size_source):
    bssfp_dataset = bSSFPDataSet(scratch if args.scratch else args.data_dir, max_iters=None,
                                 crop_size=input_size_source, length=args.ns, t2=args.t2)
    trainloader = data.DataLoader(bssfp_dataset, batch_size=args.bs, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    print("length of bSSFP training dataset: {}".format(len(bssfp_dataset)))
    return trainloader, len(bssfp_dataset)


if __name__ == '__main__':
    dst = bSSFPDataSet("../../../data/mscmrseg/", crop_size=224)
    trainloader = data.DataLoader(dst, batch_size=4, shuffle=False)
    for i, data in enumerate(trainloader):
        imgs, labels, _ = data
        print(imgs.shape, labels.shape, np.unique(labels))
        img = torchvision.utils.make_grid(imgs).numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img[:, :, ::-1]
        labels = labels[:, None, :]
        label = torchvision.utils.make_grid(labels).numpy()
        label = np.transpose(label, (1, 2, 0)) / 3.
        mix = np.concatenate([img, label], axis=0)
        plt.axis('off')
        plt.imshow(mix, cmap='gray')
        plt.show()
