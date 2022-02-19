import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
import glob
import os
import imgaug as ia
import imgaug.augmenters as iaa


def augmentation(image):

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            # sometimes(iaa.CropAndPad(
            #     percent=(-0.05, 0.1),
            #     pad_mode=ia.ALL,
            #     pad_cval=(0, 255)
            # )),
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode='constant'  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            ),
            iaa.SomeOf((0, 3), [
                iaa.ElasticTransformation(alpha=(0.5, 3.0), sigma=0.25),
                iaa.PiecewiseAffine(scale=(0.01, 0.05)),
                iaa.PerspectiveTransform(scale=(0.01, 0.1)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.01, 0.05), size_percent=(0.1, 0.2), per_channel=0.2),
                ]),
                iaa.OneOf([
                    iaa.GaussianBlur((1, 1.75)),  # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 4)),
                    # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 5)),
                    # blur image using local medians with kernel sizes between 2 and 7
                ]),
                # iaa.SimplexNoiseAlpha(iaa.OneOf([
                #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                # ])),
            ]),
        ],
        random_order=True
    )
    image_aug = seq(image=image)
    return image_aug


class LGEDataSet(data.Dataset):
    def __init__(self, list_path, max_iters=None, crop_size=224, pat_id=0, mode='fewshot', shuffle=False, aug=False,
                 length=None):
        self.list_path = list_path
        self.crop_size = crop_size
        self._aug = aug
        search_path = os.path.join(list_path, "trainB/*_{}_lge*.png".format(pat_id))
        self.img_ids = glob.glob(search_path)
        self.label_ids = glob.glob(os.path.join(list_path, "trainBmask/*_{}_lge*.png".format(pat_id)))
        if mode == 'fulldata':
            print('LGE dataset: fulldata')
            self.img_ids = glob.glob(os.path.join(list_path, "trainB/pat*lge*.png"))
            self.label_ids = glob.glob(os.path.join(list_path, "trainBmask/pat*lge*.png"))
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))
        self.files = []
        self.id_to_trainid = {0: 0, 85: 1, 212: 2, 255: 3}
        if shuffle:
            arr = np.random.permutation(len(self.img_ids))
            self.img_ids = np.array(self.img_ids)[arr]
            self.label_ids = np.array(self.label_ids)[arr]
        for img_file, label_file in zip(self.img_ids, self.label_ids):
            name = os.path.basename(img_file)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        print('Lge volume size: {}'.format(len(self.img_ids)))
        if mode == 'oneshot' or mode == 'fewshot':
            self.length = len(self.files)
        elif length is not None:
            self.length = length
        else:
            self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        datafiles = self.files[index % len(self.files)]
        image = Image.open(datafiles["img"]).convert('RGB')  # H,W,C
        img_w, img_h = image.size
        if img_w != self.crop_size:
            border_size = int((img_w - self.crop_size) // 2)
            image = image.crop((border_size, border_size, img_w-border_size, img_h-border_size))
        name = datafiles["name"]
        image = np.asarray(image, np.uint8)
        image = image[:, :, ::-1]
        if self._aug:
            img_aug = augmentation(image)
            img_aug = img_aug / 255.0
            img_aug = img_aug.transpose((2, 0, 1))
        image = image / 255.0
        image = image.transpose((2, 0, 1))
        if self._aug:
            return image.astype(np.float32).copy(), img_aug.astype(np.float32).copy(), name
        else:
            return image.astype(np.float32).copy(), name


def get_lge_dataloader(scratch, args, input_size, aug=False, length=None):
    lge_dataset = LGEDataSet(scratch if args.scratch else args.data_dir, max_iters=None, crop_size=input_size,
                             pat_id=args.pat_id, mode=args.mode,
                             shuffle=False, aug=aug, length=length)
    print('length of LGE training dataset: {}'.format(len(lge_dataset)))
    # target_dataset_iter = enumerate(lge_dataset)
    if args.scratch:
        data_dir = os.path.join(scratch, 'trainB/*{}*lge*.png').format(args.pat_id)
    else:
        data_dir = os.path.join(args.data_dir, 'trainB/*{}*lge*.png').format(args.pat_id)
    print(data_dir)
    print('length: {}'.format(len(glob.glob(data_dir))))
    targetloader = data.DataLoader(
        lge_dataset,
        batch_size=len(glob.glob(os.path.join(scratch if args.scratch else args.data_dir, "trainB/*_{}_*lge*.png".
                                              format(args.pat_id)))) if args.mode == 'oneshot' else args.bs,
        shuffle=True if (args.mode == 'fulldata' or args.mode == 'fewshot') else False,
        num_workers=args.num_workers, pin_memory=True)
    return targetloader


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    dst = LGEDataSet("../../../data/mscmrseg/", mode='fulldata', pat_id=10, crop_size=224, shuffle=False, aug=True)
    trainloader = data.DataLoader(dst, batch_size=2, shuffle=True,
                    num_workers=2, pin_memory=True)
    # for i, data in enumerate(trainloader):
    #     imgs, _ = data
    #     print(imgs.shape)
    #     img = torchvision.utils.make_grid(imgs).numpy()
    #     img = np.transpose(img, (1, 2, 0))
    #     img = img[:, :, ::-1]
    #     plt.axis('off')
    #     plt.imshow(img)
    #     plt.show()
    for i, data in enumerate(trainloader):
        img, img_aug, name = data
        print(i)
        plt.axis('off')
        plt.imshow(img.numpy()[0].transpose((1,2,0)), cmap='gray')
        plt.show()
        plt.imshow(img_aug.numpy()[0].transpose((1,2,0)), cmap='gray')
        plt.show()
    # for i, batch in enumerate(dst):
    #     img, img_aug, name = batch
    #     plt.imshow(img.transpose((1, 2, 0)), cmap='gray')
    #     plt.show()
    #     plt.imshow(img_aug.transpose((1, 2, 0)), cmap='gray')
    #     plt.show()
    #     print(name)
