import numpy as np
# from glob import glob
import torch as t
# import SimpleITK as sitk
# import cv2
import torch
import pandas as pd
import os

from utils.utilss import load_nii, keep_largest_connected_components
# from networks.unet import Segmentation_model
from utils_ import to_categorical
from timer import timeit

# from utils.metric import dice_coef_multilabel


def read_img(pat_id):
    assert os.path.exists('../input/PnpAda_release_data/test_ct_image_n_labels/image_ct_{}.nii.gz'.format(pat_id)), "The specified patid doesnot exists: {}".format(pat_id)
    assert os.path.exists('../input/PnpAda_release_data/test_ct_image_n_labels/gth_ct_{}.nii.gz'.format(pat_id)), "The specified patid doesnot exists: {}".format(pat_id)
    img, _, _ = load_nii('../input/PnpAda_release_data/test_ct_image_n_labels/image_ct_{}.nii.gz'.format(pat_id))
    mask, _, _ = load_nii('../input/PnpAda_release_data/test_ct_image_n_labels/gth_ct_{}.nii.gz'.format(pat_id))
    mask = np.array(mask, dtype=np.int)
    axis = 2
    img = np.moveaxis(img, axis, 0)[:, ::-1, ::-1]
    mask = np.moveaxis(mask, axis, 0)[:, ::-1, ::-1]
    imgs = []
    for i in range(img.shape[0]):
        imgs.append(img[[i-1, i, (i+1) % img.shape[0]]])
    masks = to_categorical(mask=mask[:,np.newaxis,...], num_classes=5)
    return np.array(imgs, dtype=np.float32), masks


def metrics(img_gt, img_pred, ifhd=True, ifasd=True):
    from medpy.metric.binary import hd, dc, asd
    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = []
    cat = {'Myo', 'LA-blood', 'LV-blood', 'AA'}
    for c in range(len(cat)):
        # Copy the gt image to not alterate the input
        gt_c_i = np.where(img_gt==c+1, 1, 0)
        # Copy the pred image to not alterate the input
        pred_c_i = np.where(img_pred==c+1, 1, 0)
        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)
        try:
            h_d = hd(gt_c_i, pred_c_i) if ifhd else -1
        except:
            h_d = -1
        try:
            a_sd = asd(gt_c_i, pred_c_i) if ifasd else -1
        except:
            a_sd = -1
        res += [dice, h_d, a_sd]

    return res


def compute_metrics_on_files(gt, pred, ifhd=True, ifasd=True):
    res = metrics(gt, pred, ifhd=ifhd, ifasd=ifasd)
    res_str = ["{:.3f}".format(r) for r in res]
    formatting = "Myo {:>8} , {:>8} , {:>8} , LA-blood {:>8} , {:>8} , {:>8} , LV-blood {:>8} , {:>8} , {:>8} , AA {:>8} , {:>8} , {:>8}"
    # print(formatting.format(*HEADER))
    print(formatting.format(*res_str))
    # to_save ="Endo: {}".format(np.around((res[0] + res[3] + res[6]) / 3., 3))
    # f = open('output.txt', 'a')
    # # print(formatting.format(*res), file=f)  # Python 3.x
    # print(to_save, file=f)  # Python 3.x

    return res


@timeit
def evaluate_segmentation(root_directory='', unet_model=None, bs=8, save=False, model_name='', ifhd=True, ifasd=True):
    """
    :param Model_name: Name of the trained model
    """
    if save:
        csv_path = 'evaluation_of_models_tf.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            data = {'DC': [], 'HD': [], 'ASD': [], 'cat': [], 'model': [], 'pad_id': []}
            df = pd.DataFrame(data)
    checkpoint = t.load(root_directory)
    try:
        unet_model.load_state_dict(checkpoint['model_state_dict'])
        print('load from dict')
    except:
        unet_model.load_state_dict(checkpoint)
        print('load from single state')
    print("model loaded")

    AA_dc,LAblood_dc,LVblood_dc,LVmyo_dc = [],[],[],[]
    AA_hd,LAblood_hd,LVblood_hd,LVmyo_hd = [],[],[],[]
    AA_asd,LAblood_asd,LVblood_asd,LVmyo_asd = [],[],[],[]
    pat_ids = [1003, 1008, 1014, 1019]
    for pat_id in pat_ids:
        print("patient {}".format(pat_id))
        x_batch, mask = read_img(pat_id)
        pred = []
        for i in range(0, len(x_batch), bs):
            index = np.arange(i, min(i + bs, len(x_batch)))
            imgs = x_batch[index]
            pred1, _, _= unet_model(torch.tensor(imgs).cuda())
            pred1 = pred1.cpu().detach().numpy()
            pred.append(pred1)
        pred = np.concatenate(pred, axis=0)
        pred = np.argmax(pred, axis=1)

        pred = keep_largest_connected_components(pred)
        pred = np.array(pred).astype(np.uint16)
        res = compute_metrics_on_files(np.argmax(mask, axis=1), pred, ifhd=ifhd, ifasd=ifasd)
        if save:
            df2 = pd.DataFrame([[res[0], res[1], res[2], 'lv', model_name, pat_id],
                                [res[3], res[4], res[5], 'rv', model_name, pat_id],
                                [res[6], res[7], res[8], 'myo', model_name, pat_id]],
                               columns=['DC', 'HD', 'ASD', 'cat', 'model', 'pad_id'])
            df = df.append(df2, ignore_index=True)
        # endo, rv, myo
        LVmyo_dc.append(res[0])
        LAblood_dc.append(res[3])
        LVblood_dc.append(res[6])
        AA_dc.append(res[9])
        if res[1] != -1:
            LVmyo_hd.append(res[1])
        if res[4] != -1:
            LAblood_hd.append(res[4])
        if res[7] != -1:
            LVblood_hd.append(res[7])
        if res[10] != -1:
            AA_hd.append(res[10])
        if res[2] != -1:
            LVmyo_asd.append(res[2])
        if res[5] != -1:
            LAblood_asd.append(res[5])
        if res[8] != -1:
            LVblood_asd.append(res[8])
        if res[11] != -1:
            AA_asd.append(res[11])
    if save:
        df.to_csv(csv_path, index=False)
    mean_AA_dc = np.around(np.mean(np.array(AA_dc)), 3)
    mean_LAblood_dc = np.around(np.mean(np.array(LAblood_dc)), 3)
    mean_LVblood_dc = np.around(np.mean(np.array(LVblood_dc)), 3)
    mean_LVmyo_dc = np.around(np.mean(np.array(LVmyo_dc)), 3)
    std_AA_dc = np.around(np.std(np.array(AA_dc)), 3)
    std_LAblood_dc = np.around(np.std(np.array(LAblood_dc)), 3)
    std_LVblood_dc = np.around(np.std(np.array(LVblood_dc)), 3)
    std_LVmyo_dc = np.around(np.std(np.array(LVmyo_dc)), 3)

    print("Ave AA DC: {}, {}, Ave LAblood DC: {}, {}, Ave LVblood DC: {}, {}, Ave LVmyo DC: {}, {}".format(mean_AA_dc, std_AA_dc, mean_LAblood_dc, std_LAblood_dc, mean_LVblood_dc, std_LVblood_dc, mean_LVmyo_dc, std_LVmyo_dc))
    print("Ave Dice: {:.3f}, {:.3f}".format((mean_AA_dc + mean_LAblood_dc + mean_LVblood_dc + mean_LVmyo_dc) / 4., (std_AA_dc + std_LAblood_dc + std_LVblood_dc + std_LVmyo_dc) / 4.))
    if ifhd:
        mean_AA_hd = np.around(np.mean(np.array(AA_hd)), 3)
        mean_LAblood_hd = np.around(np.mean(np.array(LAblood_hd)), 3)
        mean_LVblood_hd = np.around(np.mean(np.array(LVblood_hd)), 3)
        mean_LVmyo_hd = np.around(np.mean(np.array(LVmyo_hd)), 3)
        std_AA_hd = np.around(np.std(np.array(AA_hd)), 3)
        std_LAblood_hd = np.around(np.std(np.array(LAblood_hd)), 3)
        std_LVblood_hd = np.around(np.std(np.array(LVblood_hd)), 3)
        std_LVmyo_hd = np.around(np.std(np.array(LVmyo_hd)), 3)
        print("Ave AA HD: {}, {}, Ave LAblood HD: {}, {}, Ave LVblood HD: {}, {}, Ave LVmyo HD: {}, {}".format(mean_AA_hd, std_AA_hd, mean_LAblood_hd, std_LAblood_hd, mean_LVblood_hd, std_LVblood_hd, mean_LVmyo_hd, std_LVmyo_hd))
        print("Ave HD: {:.3f}, {:.3f}".format((mean_AA_hd + mean_LAblood_hd + mean_LVblood_hd + mean_LVmyo_hd) / 4., (std_AA_hd + std_LAblood_hd + std_LVblood_hd + std_LVmyo_hd) / 4.))
    if ifasd:
        mean_AA_asd = np.around(np.mean(np.array(AA_asd)), 3)
        mean_LAblood_asd = np.around(np.mean(np.array(LAblood_asd)), 3)
        mean_LVblood_asd = np.around(np.mean(np.array(LVblood_asd)), 3)
        mean_LVmyo_asd = np.around(np.mean(np.array(LVmyo_asd)), 3)
        std_AA_asd = np.around(np.std(np.array(AA_asd)), 3)
        std_LAblood_asd = np.around(np.std(np.array(LAblood_asd)), 3)
        std_LVblood_asd = np.around(np.std(np.array(LVblood_asd)), 3)
        std_LVmyo_asd = np.around(np.std(np.array(LVmyo_asd)), 3)
        print("Ave AA ASD: {}, {}, Ave LAblood ASD: {}, {}, Ave LVblood ASD: {}, {}, Ave LVmyo ASD: {}, {}".format(mean_AA_asd, std_AA_asd, mean_LAblood_asd, std_LAblood_asd, mean_LVblood_asd, std_LVblood_asd, mean_LVmyo_asd, std_LVmyo_asd))
        print("Ave ASD: {:.3f}, {:.3f}".format((mean_AA_asd + mean_LAblood_asd + mean_LVblood_asd + mean_LVmyo_asd) / 4., (std_AA_asd + std_LAblood_asd + std_LVblood_asd + std_LVmyo_asd) / 4.))

    print('{}, {}, {}, {}, {}, {}, {}, {}'.format(mean_AA_dc, std_AA_dc, mean_LAblood_dc, std_LAblood_dc, mean_LVblood_dc, std_LVblood_dc, mean_LVmyo_dc, std_LVmyo_dc))
    if ifhd:
        print('{}, {}, {}, {}, {}, {}, {}, {}'.format(mean_AA_hd, std_AA_hd, mean_LAblood_hd, std_LAblood_hd, mean_LVblood_hd, std_LVblood_hd, mean_LVmyo_hd, std_LVmyo_hd))
    if ifasd:
        print('{}, {}, {}, {}, {}, {}, {}, {}'.format(mean_AA_asd, std_AA_asd, mean_LAblood_asd, std_LAblood_asd, mean_LVblood_asd, std_LVblood_asd, mean_LVmyo_asd, std_LVmyo_asd))

if __name__ == '__main__':
    from networks.unet import Segmentation_model_Point
    seed = 0
    np.random.seed(seed)
    t.manual_seed(seed)
    # "d4: best_unet_model_checkpoint_train_point_tf.resnet.lr0.0002.d4lr1.5e-05.offaug.hvyaug.softmax.Scr0.589.pt"
    # "d2d4: best_unet_model_checkpoint_train_point_tf.resnet.lr0.0002.d2lr1.5e-05.d4lr1.5e-05.offaug.hvyaug.softmax.Scr0.659.pt"
    # "d1d2d4: best_unet_model_checkpoint_train_point_tf.resnet.lr0.0002.d1lr1.5e-05.d2lr1e-05.d4lr2.5e-05.offaug.hvyaug.softmax.Scr0.652.pt"
    file_name = "best_unet_model_checkpoint_train_point_tf.resnet.lr0.001.d1lr1.5e-05.d2lr1.5e-05.offaug.gn.softmax.Scr0.58.pt"
    root_directory = '../weights/' + 'DeepLabv3' + '/' + file_name
    toprint = "model: "
    if "d1lr" in root_directory:
        toprint += "d1"
    if "d2lr" in root_directory:
        toprint += "d2"
    if "d4lr" in root_directory:
        unet_model = Segmentation_model_Point(filters=32, in_channels=3, feature_dis=False, pointnet=True, drop=False, n_class=5, fc_inch=121)
        toprint += "d4"
    else:
        unet_model = Segmentation_model_Point(filters=32, in_channels=3, feature_dis=False, pointnet=False, drop=False, n_class=5, fc_inch=121)
    if 'offaug' in root_directory:
        toprint += ' | offaug'
    if 'offmh' in root_directory:
        toprint += '.offmh'
    if 'gn' in root_directory:
        toprint += '.gn'
    if 'softmax' in root_directory:
        toprint += '.softmax'
    if 'etpls' in root_directory:
        toprint += '.etpls'
    if 'Tetpls' in root_directory:
        toprint += '.Tetpls'
    if toprint != "":
        print(toprint)
    unet_model.cuda()
    unet_model.eval()
    evaluate_segmentation(root_directory=root_directory, unet_model=unet_model, save=False, model_name='', ifhd=True, ifasd=True)
