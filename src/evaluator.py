from datetime import datetime
import os
import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from utils.utils import load_nii, read_img_mscmr, resize_volume, keep_largest_connected_components, crop_volume, \
    reconstruct_volume, reconstruct_volume_torch
from utils.timer import timeit
from metric import metrics, metrics_torch


class Evaluator:
    """
        Evaluate the performance of a segmentation model with the raw data of bSSFP and LGE
    """
    def __init__(self, data_dir='../data/mscmrseg', class_name=('myo', 'lv', 'rv')):
        """
        Parameters
        ----------
        class_name:
        """
        self.class_name = class_name
        self._data_dir = data_dir

    @timeit
    def evaluate_single_dataset(self, seg_model, model_name='best_model', modality='lge', phase='test', ifhd=True,
                                ifasd=True, save=False, weight_dir=None, bs=32, toprint=True, lge_train_test_split=None,
                                cal_unctnty=False, watch_pat=None, klc=True):
        """
        Function to compute the metrics for a single modality of a single dataset.
        Parameters
        ----------
        seg_model: t.nn.Module
        the segmentation module.
        model_name: str
        the model name to be saved.
        modality: str
        choose from "bssfp" and "lge".
        phase: str
        choose from "train", "valid" and "test".
        ifhd: bool
        whether to calculate HD.
        ifasd: bool
        whether to calculate ASD.
        save: bool
        whether to save the resuls as csv file.
        weight_dir: str
        specify the directory to the weight if load weight.
        bs: int
        the batch size for prediction (only for memory saving).
        toprint: bool
        whether to print out the results.
        (following are not used for FUDA)
        lge_train_test_split: int
        specify from where the training data should be splitted into training and testing data.
        cal_unctnty: bool
        whether to calculate and print out the highest uncertainty (entropy) of the prediction.
        watch_pat: int
        specify the pat_id that should be printed out its uncertainty.

        Returns a dictionary of metrics {dc: [], hd: [], asd: []}.
        -------

        """
        with torch.no_grad():
            uncertainty_list, uncertainty_slice_list = [], []
            seg_model.eval()
            if save:
                csv_path = 'evaluation_of_models_on_{}_for_{}_{}.csv'.format(modality, phase, datetime.now().date())
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                else:
                    data = {'DC': [], 'HD': [], 'ASD': [], 'cat': [], 'model': [], 'pad_id': []}
                    df = pd.DataFrame(data)
            if weight_dir is not None:
                try:
                    seg_model.load_state_dict(torch.load(weight_dir)['model_state_dict'])
                except:
                    seg_model.load_state_dict(torch.load(weight_dir))
                print("model loaded")
            if modality == 'lge':
                folder = 'LGE'
            elif modality == 'bssfp':
                folder = 'C0'
            else:
                raise ValueError('modality can only be \'bssfp\' or \'lge\'')
            endo_dc, myo_dc, rv_dc = [], [], []
            endo_hd, myo_hd, rv_hd = [], [], []
            endo_asd, myo_asd, rv_asd, = [], [], []
            if phase == 'valid':
                start_idx = 1
                end_idx = 6
            elif phase == 'test':
                start_idx = 6 if lge_train_test_split is None else lge_train_test_split
                end_idx = 46
            else:
                start_idx = 6
                end_idx = 46 if lge_train_test_split is None else lge_train_test_split
            for pat_id in range(start_idx, end_idx):
                # if klc:
                #     print('Evaluate pat {}'.format(pat_id))
                mask_path = os.path.join(self._data_dir, 'raw_data/labels/patient{}_{}_manual.nii.gz'.format(pat_id, folder))

                nimg, affine, header = load_nii(mask_path)
                vol_resize = read_img_mscmr(pat_id, nimg.shape[2], modality=modality, file_path=self._data_dir)
                vol_resize = crop_volume(vol_resize, crop_size=112)
                x_batch = np.array(vol_resize, np.float32) / 255.
                x_batch = np.moveaxis(x_batch, -1, 1)
                pred = []
                # temp = []
                for i in range(0, len(x_batch), bs):
                    index = np.arange(i, min(i + bs, len(x_batch)))
                    imgs = x_batch[index]
                    pred_temp = seg_model(torch.tensor(imgs).cuda())
                    pred1, pred_norm = pred_temp[0], pred_temp[1]
                    # uncertainty = F.softmax(pred1, dim=1).cpu().detach().numpy()
                    # temp.append(uncertainty)
                    pred1 = pred1.cpu().detach().numpy()
                    pred.append(pred1)
                pred = np.concatenate(pred, axis=0)
                # pred = np.moveaxis(pred, 1, 3)
                pred = reconstruct_volume(pred, crop_size=112)
                pred_resize = []
                for i in range(0, 4):
                    pred_resize.append(resize_volume(pred[:, i, :, :], w=nimg.shape[0], h=nimg.shape[1]))
                pred = np.stack(np.array(pred_resize), axis=3)
                pred = np.argmax(pred, axis=3)
                if klc:
                    pred = keep_largest_connected_components(pred)
                masks = nimg.T
                masks = np.where(masks == 200, 1, masks)
                masks = np.where(masks == 500, 2, masks)
                masks = np.where(masks == 600, 3, masks)
                pred = np.array(pred).astype(np.uint16)
                res = metrics(masks, pred, apply_hd=ifhd, apply_asd=ifasd, pat_id=pat_id, modality=modality,
                              class_name=self.class_name)
                if save:
                    df2 = pd.DataFrame([[res['lv'][0], res['lv'][1], res['lv'][2], 'lv', model_name, pat_id],
                                        [res['rv'][0], res['rv'][1], res['rv'][2], 'rv', model_name, pat_id],
                                        [res['myo'][0], res['myo'][1], res['myo'][2], 'myo', model_name, pat_id]],
                                       columns=['DC', 'HD', 'ASD', 'cat', 'model', 'pad_id'])
                    df = df.append(df2, ignore_index=True)
                # endo, rv, myo
                endo_dc.append(res['lv'][0])
                rv_dc.append(res['rv'][0])
                myo_dc.append(res['myo'][0])
                if res['lv'][1] != -1:
                    endo_hd.append(res['lv'][1])
                if res['rv'][1] != -1:
                    rv_hd.append(res['rv'][1])
                if res['myo'][1] != -1:
                    myo_hd.append(res['myo'][1])
                if res['lv'][2] != -1:
                    endo_asd.append(res['myo'][2])
                if res['rv'][2] != -1:
                    rv_asd.append(res['rv'][2])
                if res['myo'][2] != -1:
                    myo_asd.append(res['myo'][2])
            if cal_unctnty:
                pat_highest_ucty = np.argmax(uncertainty_list) + start_idx
                print("The pat id with the highest uncertainty: {}".format(pat_highest_ucty))
                print("The slice with the highest uncertainty in the pat {}: {}".format(pat_highest_ucty, np.argmax(uncertainty_slice_list[np.argmax(uncertainty_list)])))
                print("The pat id with the lowest uncertainty: {}".format(np.argmin(uncertainty_list) + start_idx))
                if watch_pat:
                    print("The slice with the highest uncertainty in the pat {}: {}".format(watch_pat, np.argmax(
                        uncertainty_slice_list[watch_pat - start_idx])))
                    print("Uncertainty of the slices of pat {}: {}".format(watch_pat, uncertainty_slice_list[watch_pat - start_idx]))
                print("Uncertainty list: {}".format(np.round(uncertainty_list, 5)))
                print("The patient with the highest DC: {}".format(np.argmax(endo_dc) + start_idx))
                print("The patient with the lowest DC: {}".format(np.argmin(endo_dc) + start_idx))
                print("DC list: {}".format(np.round(endo_dc, 3)))
            if save:
                df.to_csv(csv_path, index=False)
            measures = self.calculate_messages(endo_dc, rv_dc, myo_dc, endo_hd, rv_hd, myo_hd, endo_asd, rv_asd,
                                               myo_asd,
                                               toprint, modality, phase, ifhd, ifasd)
        return measures

    @timeit
    def evaluate_single_dataset_torch(self, seg_model, model_name='best_model', modality='lge', phase='test', ifhd=True,
                                ifasd=True, save=False, weight_dir=None, bs=32, toprint=True, lge_train_test_split=None,
                                cal_unctnty=False, watch_pat=None, klc=True):
        """
        Function to compute the metrics for a single modality of a single dataset.
        Parameters
        ----------
        seg_model: t.nn.Module
        the segmentation module.
        model_name: str
        the model name to be saved.
        modality: str
        choose from "bssfp" and "lge".
        phase: str
        choose from "train", "valid" and "test".
        ifhd: bool
        whether to calculate HD.
        ifasd: bool
        whether to calculate ASD.
        save: bool
        whether to save the resuls as csv file.
        weight_dir: str
        specify the directory to the weight if load weight.
        bs: int
        the batch size for prediction (only for memory saving).
        toprint: bool
        whether to print out the results.
        (following are not used for FUDA)
        lge_train_test_split: int
        specify from where the training data should be splitted into training and testing data.
        cal_unctnty: bool
        whether to calculate and print out the highest uncertainty (entropy) of the prediction.
        watch_pat: int
        specify the pat_id that should be printed out its uncertainty.

        Returns a dictionary of metrics {dc: [], hd: [], asd: []}.
        -------

        """
        with torch.no_grad():
            uncertainty_list, uncertainty_slice_list = [], []
            seg_model.eval()
            if save:
                csv_path = 'evaluation_of_models_on_{}_for_{}_{}.csv'.format(modality, phase, datetime.now().date())
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                else:
                    data = {'DC': [], 'HD': [], 'ASD': [], 'cat': [], 'model': [], 'pad_id': []}
                    df = pd.DataFrame(data)
            if weight_dir is not None:
                try:
                    seg_model.load_state_dict(torch.load(weight_dir)['model_state_dict'])
                except:
                    seg_model.load_state_dict(torch.load(weight_dir))
                print("model loaded")
            if modality == 'lge':
                folder = 'LGE'
            elif modality == 'bssfp':
                folder = 'C0'
            else:
                raise ValueError('modality can only be \'bssfp\' or \'lge\'')
            endo_dc, myo_dc, rv_dc = [], [], []
            endo_hd, myo_hd, rv_hd = [], [], []
            endo_asd, myo_asd, rv_asd, = [], [], []
            if phase == 'valid':
                start_idx = 1
                end_idx = 6
            elif phase == 'test':
                start_idx = 6 if lge_train_test_split is None else lge_train_test_split
                end_idx = 46
            else:
                start_idx = 6
                end_idx = 46 if lge_train_test_split is None else lge_train_test_split
            for pat_id in range(start_idx, end_idx):
                mask_path = os.path.join(self._data_dir, 'raw_data/labels/patient{}_{}_manual.nii.gz'.format(pat_id, folder))

                nimg, affine, header = load_nii(mask_path)
                vol_resize = read_img_mscmr(pat_id, nimg.shape[2], modality=modality, file_path=self._data_dir)
                vol_resize = crop_volume(vol_resize, crop_size=112)
                x_batch = np.array(vol_resize, np.float32) / 255.
                x_batch = np.moveaxis(x_batch, -1, 1)
                pred = []
                for i in range(0, len(x_batch), bs):
                    index = np.arange(i, min(i + bs, len(x_batch)))
                    imgs = x_batch[index]
                    pred_temp = seg_model(torch.tensor(imgs).cuda())
                    pred1, pred_norm = pred_temp[0], pred_temp[1]
                    pred1 = pred1.detach()
                    pred.append(pred1)
                pred = torch.concat(pred, dim=0)
                pred = reconstruct_volume_torch(pred, crop_size=112)
                pred = F.interpolate(pred, (nimg.shape[0], nimg.shape[1]), mode='bilinear', align_corners=True)
                pred = torch.argmax(pred, dim=1).cuda()
                if klc:
                    pred = keep_largest_connected_components(pred.cpu().numpy())
                    pred = torch.tensor(pred, dtype=torch.int).cuda()
                masks = nimg.T
                masks = torch.tensor(masks).long().cuda()
                masks = torch.where(masks == 200, 1, masks)
                masks = torch.where(masks == 500, 2, masks)
                masks = torch.where(masks == 600, 3, masks)
                res = metrics_torch(masks, pred, apply_hd=ifhd, apply_asd=ifasd, pat_id=pat_id, modality=modality,
                              class_name=self.class_name)
                if save:
                    df2 = pd.DataFrame([[res['lv'][0], res['lv'][1], res['lv'][2], 'lv', model_name, pat_id],
                                        [res['rv'][0], res['rv'][1], res['rv'][2], 'rv', model_name, pat_id],
                                        [res['myo'][0], res['myo'][1], res['myo'][2], 'myo', model_name, pat_id]],
                                       columns=['DC', 'HD', 'ASD', 'cat', 'model', 'pad_id'])
                    df = df.append(df2, ignore_index=True)
                # endo, rv, myo
                endo_dc.append(res['lv'][0])
                rv_dc.append(res['rv'][0])
                myo_dc.append(res['myo'][0])
                if res['lv'][1] != -1:
                    endo_hd.append(res['lv'][1])
                if res['rv'][1] != -1:
                    rv_hd.append(res['rv'][1])
                if res['myo'][1] != -1:
                    myo_hd.append(res['myo'][1])
                if res['lv'][2] != -1:
                    endo_asd.append(res['myo'][2])
                if res['rv'][2] != -1:
                    rv_asd.append(res['rv'][2])
                if res['myo'][2] != -1:
                    myo_asd.append(res['myo'][2])
            if cal_unctnty:
                pat_highest_ucty = np.argmax(uncertainty_list) + start_idx
                print("The pat id with the highest uncertainty: {}".format(pat_highest_ucty))
                print("The slice with the highest uncertainty in the pat {}: {}".format(pat_highest_ucty, np.argmax(uncertainty_slice_list[np.argmax(uncertainty_list)])))
                print("The pat id with the lowest uncertainty: {}".format(np.argmin(uncertainty_list) + start_idx))
                if watch_pat:
                    print("The slice with the highest uncertainty in the pat {}: {}".format(watch_pat, np.argmax(
                        uncertainty_slice_list[watch_pat - start_idx])))
                    print("Uncertainty of the slices of pat {}: {}".format(watch_pat, uncertainty_slice_list[watch_pat - start_idx]))
                print("Uncertainty list: {}".format(np.round(uncertainty_list, 5)))
                print("The patient with the highest DC: {}".format(np.argmax(endo_dc) + start_idx))
                print("The patient with the lowest DC: {}".format(np.argmin(endo_dc) + start_idx))
                print("DC list: {}".format(np.round(endo_dc, 3)))
            if save:
                df.to_csv(csv_path, index=False)
            measures = self.calculate_messages(endo_dc, rv_dc, myo_dc, endo_hd, rv_hd, myo_hd, endo_asd, rv_asd, myo_asd,
                           toprint, modality, phase, ifhd, ifasd)
        return measures

    def calculate_messages(self, endo_dc, rv_dc, myo_dc, endo_hd, rv_hd, myo_hd, endo_asd, rv_asd, myo_asd,
                           toprint, modality, phase, ifhd, ifasd):
        mean_endo_dc = np.around(np.mean(np.array(endo_dc)), 3)
        mean_rv_dc = np.around(np.mean(np.array(rv_dc)), 3)
        mean_myo_dc = np.around(np.mean(np.array(myo_dc)), 3)
        std_endo_dc = np.around(np.std(np.array(endo_dc)), 3)
        std_rv_dc = np.around(np.std(np.array(rv_dc)), 3)
        std_myo_dc = np.around(np.std(np.array(myo_dc)), 3)
        if toprint:
            print("Modality: {}, Phase: {}".format(modality, phase))
            print("Ave endo DC: {:.3f}, {:.3f}, Ave rv DC: {:.3f}, {:.3f}, Ave myo DC: {:.3f}, {:.3f}".format(mean_endo_dc, std_endo_dc,
                                                                                      mean_rv_dc,
                                                                                      std_rv_dc, mean_myo_dc,
                                                                                      std_myo_dc))
            print("Ave Dice: {:.3f}, {:.3f}".format((mean_endo_dc + mean_rv_dc + mean_myo_dc) / 3.,
                                                    (std_endo_dc + std_rv_dc + std_myo_dc) / 3.))

        if ifhd:
            mean_endo_hd = np.around(np.mean(np.array(endo_hd)), 3)
            mean_rv_hd = np.around(np.mean(np.array(rv_hd)), 3)
            mean_myo_hd = np.around(np.mean(np.array(myo_hd)), 3)
            std_endo_hd = np.around(np.std(np.array(endo_hd)), 3)
            std_rv_hd = np.around(np.std(np.array(rv_hd)), 3)
            std_myo_hd = np.around(np.std(np.array(myo_hd)), 3)
            if toprint:
                print("Ave endo HD: {:.3f}, {:.3f}, Ave rv HD: {:.3f}, {:.3f}, Ave myo HD: {:.3f}, {:.3f}".format(mean_endo_hd, std_endo_hd,
                                                                                          mean_rv_hd, std_rv_hd,
                                                                                          mean_myo_hd, std_myo_hd))
                print("Ave HD: {:.3f}, {:.3f}".format((mean_endo_hd + mean_rv_hd + mean_myo_hd) / 3.,
                                                      (std_endo_hd + std_rv_hd + std_myo_hd) / 3.))
        else:
            mean_myo_hd, std_myo_hd, mean_endo_hd, std_endo_hd, mean_rv_hd, std_rv_hd = 0, 0, 0, 0, 0, 0
        if ifasd:
            mean_endo_asd = np.around(np.mean(np.array(endo_asd)), 3)
            mean_rv_asd = np.around(np.mean(np.array(rv_asd)), 3)
            mean_myo_asd = np.around(np.mean(np.array(myo_asd)), 3)
            std_endo_asd = np.around(np.std(np.array(endo_asd)), 3)
            std_rv_asd = np.around(np.std(np.array(rv_asd)), 3)
            std_myo_asd = np.around(np.std(np.array(myo_asd)), 3)
            if toprint:
                print(
                    "Ave endo ASD: {:.3f}, {:.3f}, Ave rv ASD: {:.3f}, {:.3f}, Ave myo ASD: {:.3f}, {:.3f}".format(mean_endo_asd, std_endo_asd,
                                                                                           mean_rv_asd, std_rv_asd,
                                                                                           mean_myo_asd, std_myo_asd))
                print("Ave ASD: {:.3f}, {:.3f}".format((mean_endo_asd + mean_rv_asd + mean_myo_asd) / 3.,
                                                       (std_endo_asd + std_rv_asd + std_myo_asd) / 3.))
        else:
            mean_myo_asd, std_myo_asd, mean_endo_asd, std_endo_asd, mean_rv_asd, std_rv_asd = 0, 0, 0, 0, 0, 0

        if toprint:
            print(
                'DC: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(mean_myo_dc, std_myo_dc, mean_endo_dc, std_endo_dc, mean_rv_dc,
                                                    std_rv_dc))
            if ifhd:
                print(
                    'HD: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(mean_myo_hd, std_myo_hd, mean_endo_hd, std_endo_hd, mean_rv_hd,
                                                        std_rv_hd))
            if ifasd:
                print('ASD: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(mean_myo_asd, std_myo_asd, mean_endo_asd, std_endo_asd,
                                                           mean_rv_asd,
                                                           std_rv_asd))

        return {'dc': [mean_myo_dc, std_myo_dc, mean_endo_dc, std_endo_dc, mean_rv_dc, std_rv_dc],
                'hd': [mean_myo_hd, std_myo_hd, mean_endo_hd, std_endo_hd, mean_rv_hd, std_rv_hd],
                'asd': [mean_myo_asd, std_myo_asd, mean_endo_asd, std_endo_asd, mean_rv_asd, std_rv_asd]}

    @timeit
    def evaluate(self, seg_model, ifhd=True, ifasd=True, weight_dir=None, bs=16, lge_train_test_split=None):
        bssfp_train = self.evaluate_single_dataset(seg_model=seg_model, modality='bssfp', phase='train', ifhd=ifhd, ifasd=ifasd, save=False, weight_dir=weight_dir, bs=bs, toprint=False)
        bssfp_val = self.evaluate_single_dataset(seg_model=seg_model, modality='bssfp', phase='valid', ifhd=ifhd, ifasd=ifasd, save=False, weight_dir=weight_dir, bs=bs, toprint=False)
        lge_val = self.evaluate_single_dataset(seg_model=seg_model, modality='lge', phase='valid', ifhd=ifhd, ifasd=ifasd, save=False, weight_dir=weight_dir, bs=bs, toprint=False)
        lge_test = self.evaluate_single_dataset(seg_model=seg_model, modality='lge', phase='test', ifhd=ifhd, ifasd=ifasd, save=False, weight_dir=weight_dir, bs=bs, toprint=False,
                                                lge_train_test_split=lge_train_test_split)

        return bssfp_train, bssfp_val, lge_val, lge_test


if __name__ == '__main__':
    start = datetime.now()
    import argparse
    from networks.unet import Segmentation_model_Point
    from torch.cuda import get_device_name
    print("Device name: {}".format(get_device_name(0)))
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--restore_from", type=str,
                        default='pretrained/best_DR_UNet.fewshot.lr0.0003.cw0.002.poly.pat_10_lge.adam.e63.Scr0.674.pt',
                        help="Where restore model parameters from.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default='../../data/mscmrseg')
    parser.add_argument("--modality", type=str, default='lge')
    parser.add_argument("--phase", type=str, default='test')
    parser.add_argument("--klc", action='store_true')
    parser.add_argument("--torch", action='store_true')
    parser.add_argument("--hd", action='store_true')
    parser.add_argument("--asd", action='store_true')
    parser.add_argument("--d4", action='store_true')

    args = parser.parse_args()
    evaluator = Evaluator(data_dir=args.data_dir)
    segmentor = Segmentation_model_Point(filters=32, pointnet=args.d4, n_class=4).cuda()
    if args.torch:
        evaluator.evaluate_single_dataset_torch(segmentor, model_name='best_model', modality=args.modality, phase=args.phase, ifhd=args.hd,
                                          ifasd=args.asd, save=False, weight_dir=args.restore_from, bs=args.batch_size,
                                          toprint=True, lge_train_test_split=None, cal_unctnty=False, watch_pat=None,
                                          klc=args.klc)
    else:
        evaluator.evaluate_single_dataset(segmentor, model_name='best_model', modality=args.modality, phase=args.phase,
                                                ifhd=args.hd,
                                                ifasd=args.asd, save=False, weight_dir=args.restore_from,
                                                bs=args.batch_size,
                                                toprint=True, lge_train_test_split=None, cal_unctnty=False,
                                                watch_pat=None,
                                                klc=args.klc)
    end = datetime.now()
    print('Time elapsed: {}'.format(end - start))
