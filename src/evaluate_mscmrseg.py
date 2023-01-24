import torch as t

import numpy as np
import cv2
import torch
import pandas as pd
from tqdm import tqdm
import os

from utils.timer import timeit
from utils.utils import load_nii, resize_volume, keep_largest_connected_components
from utils.metric import compute_metrics_on_files
from albumentations import (
    Compose,
    CLAHE,
)

def crop_volume(vol, crop_size=112):
    """
    crop the images
    :param vol: the image
    :param crop_size: half size of cropped images
    :return:
    """
    return np.array(vol[:,
                    int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                    int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size])


def reconstuct_volume(vol, crop_size=112, origin_size=256):
    """
    reconstruct the image (reverse process of cropping)
    :param vol: the images
    :param crop_size: half size of cropped images
    :param origin_size: the original size of the images
    :return:
    """
    recon_vol = np.zeros((len(vol), origin_size, origin_size, 4), dtype=np.float32)

    recon_vol[:,
    int(recon_vol.shape[1] / 2) - crop_size: int(recon_vol.shape[1] / 2) + crop_size,
    int(recon_vol.shape[2] / 2) - crop_size: int(recon_vol.shape[2] / 2) + crop_size, :] = vol

    return recon_vol


def read_img(pat_id, img_len, clahe=False):
    """
    read in raw images
    :param pat_id:
    :param img_len:
    :param clahe: whether to apply clahe (False)
    :return:
    """
    images = []
    for im in range(img_len):
        img = cv2.imread(
            os.path.join(
                args.data_dir, f"processed/trainB/pat_{pat_id}_lge_{im}.png"
            )
        )
        if clahe:
            aug = Compose([CLAHE(always_apply=True)])
            augmented = aug(image=img)
            img= augmented["image"]
        images.append(img)
    return np.array(images)


def get_csv_path(model_name, clahe=False):
    """
    generate csv path to save the result
    :param model_name:
    :param clahe:
    :return:
    """
    csv_path = model_name
    if clahe:
        csv_path += '_clahe'
    csv_path += '_evaluation.csv'
    return csv_path


@timeit
def evaluate_segmentation(unet_model, bs=8, clahe=False, save=False, toprint=True, toplot=False, model_name='',
                          ifhd=True, ifasd=True, pat_id_range=(6, 46), weight_dir = '', crop_size=224, klc=True):
    """
    to evaluate the trained model
    :param unet_model: Name of the model to load.
    :param bs: batch size.
    :param clahe: whether to apply clahe.
    :param save: whether to save the evaluation result.
    :param toprint: whether to print the result.
    :param toplot: whether to plot the prediction.
    :param model_name: the model name (only for files to save).
    :param ifhd: whether to calculate HD.
    :param ifasd: whether to calculate ASD.
    :param pat_id_range: the pat_ids should be in (6, 46).
    :param weight_dir: the directory to the weight.
    :param crop_size: the size of the cropped images.
    :param klc: whether to apply 'keep largest connected component'.
    :return:
    """
    assert (pat_id_range[0] <= pat_id_range[1]) and (pat_id_range[0] >= 6) and (pat_id_range[1] <= 46), "pat_id_range error."
    if save:
        csv_path = get_csv_path(model_name=model_name, clahe=clahe)
        print(csv_path)
        if pat_id_range[0] > 6:
            df = pd.read_csv(csv_path)
        else:
            data = {'DSC': [], 'HD': [], 'ASD': [], 'cat': [], 'model': [], 'pad_id': []}
            df = pd.DataFrame(data)
    checkpoint = t.load(weight_dir)
    try:
        unet_model.load_state_dict(checkpoint['model_state_dict'])
        print('model load from dict')
    except:
        unet_model.load_state_dict(checkpoint)
        print('model load from single state')
    unet_model.eval()

    if toprint:
        endo_dc,myo_dc,rv_dc = [],[],[]
        endo_hd,myo_hd,rv_hd = [],[],[]
        endo_asd,myo_asd,rv_asd, = [],[],[]
    for pat_id in tqdm(range(pat_id_range[0], pat_id_range[1])):
        print(f"patient {pat_id}")
        mask_path = os.path.join(
            args.data_dir,
            f"raw_data/labels/lge_test_gt/patient{pat_id}_LGE_manual.nii.gz",
        )

        nimg, affine, header = load_nii(mask_path)
        vol_resize = read_img(pat_id, nimg.shape[2], clahe=clahe)
        vol_resize = crop_volume(vol_resize, crop_size=crop_size // 2)
        x_batch = np.array(vol_resize, np.float32) / 255.
        x_batch = np.moveaxis(x_batch, -1, 1)
        pred = []
        for i in range(0, len(x_batch), bs):
            index = np.arange(i, min(i + bs, len(x_batch)))
            imgs = x_batch[index]
            pred1, _, _= unet_model(torch.tensor(imgs).cuda())
            pred1 = pred1.cpu().detach().numpy()
            pred.append(pred1)
        pred = np.concatenate(pred, axis=0)
        pred = np.moveaxis(pred, 1, -1)
        pred = reconstuct_volume(pred, crop_size=112)
        pred_resize = [
            resize_volume(pred[:, :, :, i], w=nimg.shape[0], h=nimg.shape[1])
            for i in range(4)
        ]
        pred = np.stack(np.array(pred_resize), axis=3)
        pred = np.argmax(pred, axis=3)

        masks = nimg.T
        if klc:
            pred = keep_largest_connected_components(pred)
        pred = np.array(pred).astype(np.uint16)
        pred = np.where(pred == 1, 200, pred)
        pred = np.where(pred == 2, 500, pred)
        pred = np.where(pred == 3, 600, pred)

        if toplot:
            from matplotlib import pyplot as plt
            for x, prediction, mask in zip(x_batch, pred, masks):
                f, ax = plt.subplots(1,3,figsize=(10,6))
                ax[0].axis('off')
                ax[0].imshow(x[1], cmap='gray')
                ax[1].axis('off')
                ax[1].imshow(prediction, cmap='gray', vmin=0, vmax=600)
                ax[-1].axis('off')
                ax[-1].imshow(mask, cmap='gray', vmin=0, vmax=600)
                plt.tight_layout()
                plt.show()
            print('plt finish')
            input()
        res = compute_metrics_on_files(masks, pred, ifhd=ifhd, ifasd=ifasd)
        if save:
            df2 = pd.DataFrame([[res[0], res[1], res[2], 'lv', model_name, pat_id],
                                [res[3], res[4], res[5], 'rv', model_name, pat_id],
                                [res[6], res[7], res[8], 'myo', model_name, pat_id]],
                               columns=['DSC', 'HD', 'ASD', 'cat', 'model', 'pad_id'])
            df = df.append(df2, ignore_index=True)
        if toprint:
            # endo, rv, myo
            endo_dc.append(res[0])
            rv_dc.append(res[3])
            myo_dc.append(res[6])
            if res[1] != -1:
                endo_hd.append(res[1])
            if res[4] != -1:
                rv_hd.append(res[4])
            if res[7] != -1:
                myo_hd.append(res[7])
            if res[2] != -1:
                endo_asd.append(res[2])
            if res[5] != -1:
                rv_asd.append(res[5])
            if res[8] != -1:
                myo_asd.append(res[8])
        if save:
            df.to_csv(csv_path, index=False)

    if toprint:
        mean_endo_dc = np.around(np.mean(np.array(endo_dc)), 3)
        mean_rv_dc = np.around(np.mean(np.array(rv_dc)), 3)
        mean_myo_dc = np.around(np.mean(np.array(myo_dc)), 3)
        std_endo_dc = np.around(np.std(np.array(endo_dc)), 3)
        std_rv_dc = np.around(np.std(np.array(rv_dc)), 3)
        std_myo_dc = np.around(np.std(np.array(myo_dc)), 3)

        print(
            f"Ave endo DC: {mean_endo_dc}, {std_endo_dc}, Ave rv DC: {mean_rv_dc}, {std_rv_dc}, Ave myo DC: {mean_myo_dc}, {std_myo_dc}"
        )
        print("Ave Dice: {:.3f}, {:.3f}".format((mean_endo_dc + mean_rv_dc + mean_myo_dc) / 3., (std_endo_dc + std_rv_dc + std_myo_dc) / 3.))
        if ifhd:
            mean_endo_hd = np.around(np.mean(np.array(endo_hd)), 3)
            mean_rv_hd = np.around(np.mean(np.array(rv_hd)), 3)
            mean_myo_hd = np.around(np.mean(np.array(myo_hd)), 3)
            std_endo_hd = np.around(np.std(np.array(endo_hd)), 3)
            std_rv_hd = np.around(np.std(np.array(rv_hd)), 3)
            std_myo_hd = np.around(np.std(np.array(myo_hd)), 3)
            print(
                f"Ave endo HD: {mean_endo_hd}, {std_endo_hd}, Ave rv HD: {mean_rv_hd}, {std_rv_hd}, Ave myo HD: {mean_myo_hd}, {std_myo_hd}"
            )
            print("Ave HD: {:.3f}, {:.3f}".format((mean_endo_hd + mean_rv_hd + mean_myo_hd) / 3., (std_endo_hd + std_rv_hd + std_myo_hd) / 3.))
        if ifasd:
            mean_endo_asd = np.around(np.mean(np.array(endo_asd)), 3)
            mean_rv_asd = np.around(np.mean(np.array(rv_asd)), 3)
            mean_myo_asd = np.around(np.mean(np.array(myo_asd)), 3)
            std_endo_asd = np.around(np.std(np.array(endo_asd)), 3)
            std_rv_asd = np.around(np.std(np.array(rv_asd)), 3)
            std_myo_asd = np.around(np.std(np.array(myo_asd)), 3)
            print(
                f"Ave endo ASD: {mean_endo_asd}, {std_endo_asd}, Ave rv ASD: {mean_rv_asd}, {std_rv_asd}, Ave myo ASD: {mean_myo_asd}, {std_myo_asd}"
            )
            print("Ave ASD: {:.3f}, {:.3f}".format((mean_endo_asd + mean_rv_asd + mean_myo_asd) / 3., (std_endo_asd + std_rv_asd + std_myo_asd) / 3.))

        print(
            f'{mean_myo_dc}, {std_myo_dc}, {mean_endo_dc}, {std_endo_dc}, {mean_rv_dc}, {std_rv_dc}'
        )
        if ifhd:
            print(
                f'{mean_myo_hd}, {std_myo_hd}, {mean_endo_hd}, {std_endo_hd}, {mean_rv_hd}, {std_rv_hd}'
            )
        if ifasd:
            print(
                f'{mean_myo_asd}, {std_myo_asd}, {mean_endo_asd}, {std_endo_asd}, {mean_rv_asd}, {std_rv_asd}'
            )

    print('Evaluation finished')


if __name__ == '__main__':
    from networks.unet import Segmentation_model_Point
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", help="the batch size of training", type=int, default=16)
    parser.add_argument("-klc", help="whether to apply keep largest connected components", action='store_true')
    parser.add_argument("-toplot", help="whether to plt the prediction and mask", action='store_true')
    parser.add_argument("-model_dir", help="to specify the directory to the model to load", type=str, default='')
    parser.add_argument("-model_chosen", help="the model chosen to be evaluated", type=str, default='unet')
    parser.add_argument("-data_dir", help="the directory to the data", type=str, default='../../project_cross_modality/Working_Model/input/')
    args = parser.parse_args()
    seed = 0
    np.random.seed(seed)
    t.manual_seed(seed)
    if args.model_dir == '':
        model_names = {"unet": "best_unet_model_checkpoint_train_point_unet.resnet.lr0.001.offaug.Scr0.564.pt",
                       "unet_medium_aug": "best_unet_model_checkpoint_train_point_imgaug_unet.resnet.lr0.001.offaug.Scr0.568.pt",
                       "unet_heavy_aug": "best_unet_model_checkpoint_train_point_imgaug_unet.resnet.lr0.001.Scr0.834.pt",
                       "unet_d2": "best_unet_model_checkpoint_train_point_imgaug_concat.resnet.lr0.001.d2lr2.5e-05.softmax.Scr0.845.pt",
                       "unet_d1d2": "best_unet_model_checkpoint_train_point_imgaug_concat.resnet.lr0.001.d1lr2.5e-05.d2lr2.5e-05.softmax.Scr0.849.pt",
                       "unet_d4_aug2": "best_unet_model_checkpoint_train_point_imgaug.resnet.lr0.001.d4lr2.5e-05.aug2.softmax.Scr0.832.pt",
                       "unet_d2d4_aug2": "best_unet_model_checkpoint_train_point_imgaug.resnet.lr0.001.d2lr2.5e-05.d4lr2.5e-05.aug2.softmax.Scr0.816.pt",
                       "unet_d1d2d4_aug2": "best_unet_model_checkpoint_train_point_imgaug.resnet.lr0.001.d1lr2.5e-05.d2lr2.5e-05.d4lr2.5e-05.aug2.softmax.Scr0.822.pt"}

        print("evaluate model: " + args.model_chosen)
        model_gen = Segmentation_model_Point(
            filters=32, pointnet='d4' in model_names[args.model_chosen]
        )
        file_name = model_names[args.model_chosen]
        weight_dir = '../weights/' + file_name
    else:
        weight_dir = args.model_dir
        model_gen = Segmentation_model_Point(filters=32,
                                             pointnet=False)

    ifhd = True
    ifasd = False
    # print_evaluation_result(model_name=model_to_choose, clahe=clahe, ifhd=ifhd, ifasd=ifasd)
    model_gen.cuda()
    model_gen.eval()
    evaluate_segmentation(unet_model=model_gen, bs=args.bs, save=False, toprint=args.toplot, toplot=False, ifhd=True,
                          ifasd=True, pat_id_range=(6, 46), weight_dir=weight_dir, crop_size=224, klc=args.klc)

    # use the following code if you want to evaluate all the models
    # for model_name in model_names:
    #     print(model_name)
    #     weight_path = root_directory + model_name
    #     if "d4lr" in weight_path:
    #         model_gen = Segmentation_model_Point(filters=32, pointnet=True)
    #     else:
    #         model_gen = Segmentation_model_Point(filters=32, pointnet=False)
    #     model_gen.cuda()
    #     model_gen.eval()
    #     evaluate_segmentation(unet_model=model_gen, bs=args.bs, save=False, toprint=args.toplot, toplot=True, ifhd=True,
    #     ifasd=True,pat_id_range=(6, 46), weight_dir=weight_path, crop_size=224, klc=args.klc)
