import numpy as np
import torch as t
import cv2
import torch
import pandas as pd
import os

from utils.utilss import load_nii, resize_volume, keep_largest_connected_components

from utils.metric import compute_metrics_on_files
from albumentations import (
    Compose,
    CLAHE,
)


def crop_volume(vol, crop_size=112):
    """
    :param vol:
    :return:
    """

    return np.array(vol[:,
                    int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                    int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size])


def reconstruct_volume(vol, crop_size=112):
    """
    :param vol:
    :return:
    """
    recon_vol = np.zeros((len(vol), 256, 256, 4), dtype=np.float32)

    recon_vol[:,
    int(recon_vol.shape[1] / 2) - crop_size: int(recon_vol.shape[1] / 2) + crop_size,
    int(recon_vol.shape[2] / 2) - crop_size: int(recon_vol.shape[2] / 2) + crop_size, :] = vol

    return recon_vol


def read_img(pat_id, img_len, clahe=True):

    images = []
    for im in range(img_len):
        img = cv2.imread("../input/processed/trainB/pat_{}_lge_{}.png".format(pat_id, im))
        if clahe:
            aug = Compose([CLAHE(always_apply=True)])
            augmented = aug(image=img)
            img= augmented["image"]
        images.append(img)
    return np.array(images)


def read_vertices(pat_id, img_len):

    vertices = []
    for i in range(img_len):
        vert = np.load("../input/vertices/trainB/pat_{}_lge_{}.npy".format(pat_id, i))
        vertices.append(vert)
    return np.array(vertices)


def evaluate_segmentation(bs=8, clahe=False, save=False, model_name='', ifhd=True, ifasd=True):
    """
    evaluate the model
    :param bs: batch size
    :param clahe: whether to apply clahe
    :param save: whether to save the result
    :param model_name: the name of the model(will appear in log file)
    :param ifhd: whether to evaluate with HausDorff Distance
    :param ifasd: whether to evaluate with Average Surface Distance
    :return:
    """
    if save:
        csv_path = 'evaluation_of_models.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            data = {'DC': [], 'HD': [], 'ASD': [], 'cat': [], 'model': [], 'pad_id': []}
            df = pd.DataFrame(data)
    unet_model.load_state_dict(t.load(root_directory))
    print("model loaded")

    endo_dc,myo_dc,rv_dc = [],[],[]
    endo_hd,myo_hd,rv_hd = [],[],[]
    endo_asd,myo_asd,rv_asd, = [],[],[]
    for pat_id in range(6, 46):
        print("patient {}".format(pat_id))
        # test_path = sorted(glob("../input/raw_data/dataset/patient{}_LGE.nii.gz".format(pat_id)))
        mask_path = "../input/raw_data/labels/lge_test_gt/patient{}_LGE_manual.nii.gz".format(pat_id)

        nimg, affine, header = load_nii(mask_path)
        vol_resize = read_img(pat_id, nimg.shape[2], clahe=clahe)
        vol_resize = crop_volume(vol_resize, crop_size=112)
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
        pred = np.moveaxis(pred, 1, 3)
        pred = reconstruct_volume(pred, crop_size=112)
        pred_resize = []
        for i in range(0, 4):
            pred_resize.append(resize_volume(pred[:, :, :, i], w=nimg.shape[0], h=nimg.shape[1]))
        pred = np.stack(np.array(pred_resize), axis=3)
        pred = np.argmax(pred, axis=3)

        masks = nimg.T
        pred = keep_largest_connected_components(pred)
        pred = np.array(pred).astype(np.uint16)
        pred = np.where(pred == 1, 200, pred)
        pred = np.where(pred == 2, 500, pred)
        pred = np.where(pred == 3, 600, pred)
        res = compute_metrics_on_files(masks, pred, ifhd=ifhd, ifasd=ifasd)
        if save:
            df2 = pd.DataFrame([[res[0], res[1], res[2], 'lv', model_name, pat_id],
                                [res[3], res[4], res[5], 'rv', model_name, pat_id],
                                [res[6], res[7], res[8], 'myo', model_name, pat_id]],
                               columns=['DC', 'HD', 'ASD', 'cat', 'model', 'pad_id'])
            df = df.append(df2, ignore_index=True)
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
    mean_endo_dc = np.around(np.mean(np.array(endo_dc)), 3)
    mean_rv_dc = np.around(np.mean(np.array(rv_dc)), 3)
    mean_myo_dc = np.around(np.mean(np.array(myo_dc)), 3)
    std_endo_dc = np.around(np.std(np.array(endo_dc)), 3)
    std_rv_dc = np.around(np.std(np.array(rv_dc)), 3)
    std_myo_dc = np.around(np.std(np.array(myo_dc)), 3)

    print("Ave endo DC: {}, {}, Ave rv DC: {}, {}, Ave myo DC: {}, {}".format(mean_endo_dc, std_endo_dc, mean_rv_dc, std_rv_dc, mean_myo_dc, std_myo_dc))
    print("Ave Dice: {:.3f}, {:.3f}".format((mean_endo_dc + mean_rv_dc + mean_myo_dc) / 3., (std_endo_dc + std_rv_dc + std_myo_dc) / 3.))
    if ifhd:
        mean_endo_hd = np.around(np.mean(np.array(endo_hd)), 3)
        mean_rv_hd = np.around(np.mean(np.array(rv_hd)), 3)
        mean_myo_hd = np.around(np.mean(np.array(myo_hd)), 3)
        std_endo_hd = np.around(np.std(np.array(endo_hd)), 3)
        std_rv_hd = np.around(np.std(np.array(rv_hd)), 3)
        std_myo_hd = np.around(np.std(np.array(myo_hd)), 3)
        print("Ave endo HD: {}, {}, Ave rv HD: {}, {}, Ave myo HD: {}, {}".format(mean_endo_hd, std_endo_hd, mean_rv_hd, std_rv_hd, mean_myo_hd, std_myo_hd))
        print("Ave HD: {:.3f}, {:.3f}".format((mean_endo_hd + mean_rv_hd + mean_myo_hd) / 3., (std_endo_hd + std_rv_hd + std_myo_hd) / 3.))
    if ifasd:
        mean_endo_asd = np.around(np.mean(np.array(endo_asd)), 3)
        mean_rv_asd = np.around(np.mean(np.array(rv_asd)), 3)
        mean_myo_asd = np.around(np.mean(np.array(myo_asd)), 3)
        std_endo_asd = np.around(np.std(np.array(endo_asd)), 3)
        std_rv_asd = np.around(np.std(np.array(rv_asd)), 3)
        std_myo_asd = np.around(np.std(np.array(myo_asd)), 3)
        print("Ave endo ASD: {}, {}, Ave rv ASD: {}, {}, Ave myo ASD: {}, {}".format(mean_endo_asd, std_endo_asd, mean_rv_asd, std_rv_asd, mean_myo_asd, std_myo_asd))
        print("Ave ASD: {:.3f}, {:.3f}".format((mean_endo_asd + mean_rv_asd + mean_myo_asd) / 3., (std_endo_asd + std_rv_asd + std_myo_asd) / 3.))

    print('{}, {}, {}, {}, {}, {}'.format(mean_myo_dc, std_myo_dc, mean_endo_dc, std_endo_dc, mean_rv_dc, std_rv_dc))
    if ifhd:
        print('{}, {}, {}, {}, {}, {}'.format(mean_myo_hd, std_myo_hd, mean_endo_hd, std_endo_hd, mean_rv_hd, std_rv_hd))
    if ifasd:
        print('{}, {}, {}, {}, {}, {}'.format(mean_myo_asd, std_myo_asd, mean_endo_asd, std_endo_asd, mean_rv_asd, std_rv_asd))

if __name__ == '__main__':
    from networks.unet import Segmentation_model_Point
    model_names = {"unet": "best_unet_model_checkpoint_train_point_unet.resnet.lr0.001.offaug.Scr0.564.pt",
                   "unet_medium_aug": "best_unet_model_checkpoint_train_point_imgaug_unet.resnet.lr0.001.offaug.Scr0.568.pt",
                   "unet_heavy_aug": "best_unet_model_checkpoint_train_point_imgaug_unet.resnet.lr0.001.Scr0.834.pt",
                   "unet_d2": "best_unet_model_checkpoint_train_point_imgaug_concat.resnet.lr0.001.d2lr2.5e-05.softmax.Scr0.845.pt",
                   "unet_d1d2": "best_unet_model_checkpoint_train_point_imgaug_concat.resnet.lr0.001.d1lr2.5e-05.d2lr2.5e-05.softmax.Scr0.849.pt",
                   "unet_d4_aug2": "best_unet_model_checkpoint_train_point_imgaug.resnet.lr0.001.d4lr2.5e-05.aug2.softmax.Scr0.832.pt",
                   "unet_d2d4_aug2": "best_unet_model_checkpoint_train_point_imgaug.resnet.lr0.001.d2lr2.5e-05.d4lr2.5e-05.aug2.softmax.Scr0.816.pt",
                   "unet_d1d2d4_aug2": "best_unet_model_checkpoint_train_point_imgaug.resnet.lr0.001.d1lr2.5e-05.d2lr2.5e-05.d4lr2.5e-05.aug2.softmax.Scr0.822.pt"}

    seed = 0
    # fix the random seed of both numpy and torch
    np.random.seed(seed)
    t.manual_seed(seed)
    # "unet_d1d2d4_aug2": "best_unet_model_checkpoint_train_point_imgaug.resnet.lr0.001.d1lr2.5e-05.d2lr2.5e-05.d4lr2.5e-05.aug2.softmax.Scr0.822.pt"
    model_to_choose = 'unet'
    # file_name = model_names[model_to_choose]

    file_name = "best_unet_model_checkpoint_train_point_imgaug.resnet.lr0.001.d1lr2.5e-05.d2lr2.5e-05.sigmoid.Scr0.853.pt"
    root_directory = '../weights/' + file_name
    toprint = ""
    if "d1lr" in root_directory:
        toprint += "d1"
    if "d2lr" in root_directory:
        toprint += "d2"
    if "d4lr" in root_directory:
        unet_model = Segmentation_model_Point(filters=32, feature_dis=False, pointnet=True, drop=False)
        toprint += "d4"
    else:
        unet_model = Segmentation_model_Point(filters=32, feature_dis=False, pointnet=False, drop=False)
    if toprint != "":
        print(toprint)
    unet_model.cuda()
    unet_model.eval()
    evaluate_segmentation(clahe=False, save=False, model_name=model_to_choose, ifhd=True, ifasd=True)
