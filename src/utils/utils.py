import numpy as np
from skimage import measure
import nibabel as nib
import cv2
import torch
import os
from pathlib import Path
from datetime import datetime


def to_categorical(mask, num_classes, channel='channel_first'):
    """
    convert ground truth mask to categorical
    :param mask: the ground truth mask
    :param num_classes: the number of classes
    :param channel: 'channel_first' or 'channel_last'
    :return: the categorical mask
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


def soft_to_hard_pred(pred, channel_axis=1):
    """
    convert soft prediction to either 1 or 0.
    :param pred: the prediction
    :param channel_axis: the channel axis. For 'channel_first', it should be 1.
    :return: the 'hard' prediction
    """
    max_value = np.max(pred, axis=channel_axis, keepdims=True)
    return np.where(pred == max_value, 1, 0)


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    num_channel = mask.shape[1]
    out_img = np.zeros(mask.shape, dtype=np.uint8)
    for struc_id in range(1, num_channel + 1):

        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)

        props = measure.regionprops(blobs)

        if not props:
            continue

        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label

        out_img[blobs == largest_blob_label] = struc_id

    return out_img


def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everything needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header
    :param img_path: String with the path of the 'nii' or 'nii.gz' image file name.
    :return:Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """

    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def resize_volume(img_volume, w=256, h=256):
    """
    :param img_volume:
    :return:
    """
    img_res = []
    for im in img_volume:
        img_res.append(cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_AREA))

    return np.array(img_res)

def reconstruct_volume(vol, crop_size=112, origin_size=256):
    """
    :param vol:
    :return:
    """
    recon_vol = np.zeros((len(vol), 4, origin_size, origin_size), dtype=np.float32)

    recon_vol[:, :,
    int(recon_vol.shape[2] / 2) - crop_size: int(recon_vol.shape[2] / 2) + crop_size,
    int(recon_vol.shape[3] / 2) - crop_size: int(recon_vol.shape[3] / 2) + crop_size] = vol

    return recon_vol


def read_img(pat_id, img_len, file_path='../processed/', modality='lge'):
    images = []
    if modality == 'bssfp':
        folder = 'testA' if pat_id < 6 else 'trainA'
    else:
        folder = 'testB' if pat_id < 6 else 'trainB'
    modality = 'bSSFP' if modality == 'bssfp' else 'lge'
    for im in range(img_len):
        img = cv2.imread(os.path.join(file_path, "{}/pat_{}_{}_{}.png".format(folder, pat_id, modality, im)))
        images.append(img)
    return np.array(images)


def read_img_mscmr(pat_id, img_len, file_path='../processed/', modality='lge'):
    images = []
    if modality == 'bssfp':
        folder = 'testA' if pat_id < 6 else 'trainA'
    else:
        folder = 'testB' if pat_id < 6 else 'trainB'
    modality = 'bSSFP' if modality == 'bssfp' else 'lge'
    for im in range(img_len):
        img = cv2.imread(os.path.join(file_path, "{}/pat_{}_{}_{}.png".format(folder, pat_id, modality, im)))
        images.append(img)
    return np.array(images)


def crop_volume(vol, crop_size=112):
    """
    :param vol:
    :return:
    """

    return np.array(vol[:,
                    int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                    int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size])


def reconstruct_volume_torch(vol, crop_size=112, origin_size=256):
    """
    :param vol:
    :return:
    """
    recon_vol = torch.zeros((vol.size()[0], 4, origin_size, origin_size), dtype=torch.float32)

    recon_vol[:, :,
    int(recon_vol.size()[2] / 2) - crop_size: int(recon_vol.size()[2] / 2) + crop_size,
    int(recon_vol.size()[3] / 2) - crop_size: int(recon_vol.size()[3] / 2) + crop_size] = vol

    return recon_vol


def tranfer_data_2_scratch(args, start):
    scratch = None
    if args.scratch:
        jobid = os.environ['SLURM_JOB_ID']
        print(jobid)
        scratch = os.path.join('/scratch', jobid)
        if not Path(scratch).joinpath('trainA').exists():
            data_from = os.path.join(args.data_dir, '.')
            command = 'cp -a {} {}'.format(data_from, scratch)
            print(command)
            os.system(command)
            print('time used for file transfer: {}'.format(datetime.now() - start))
        else:
            print('Data already transferred.')
    return scratch


def get_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("-aug", help='whether not to augment the data', action='store_false')
    # parser.add_argument("-aug2", help='whether to augment the data with 2nd method', action='store_true')
    parser.add_argument("-load_weight", help='whether to load weight', action='store_true')
    parser.add_argument("-bs", help="the batch size of training", type=int, default=16)
    parser.add_argument("-ns", help="number of samples per epoch", type=int, default=None)
    parser.add_argument("-e", help="number of epochs", type=int, default=200)
    parser.add_argument('-sgd', action='store_true')
    parser.add_argument("-lr", help="learning rate of unet", type=float, default=1e-3)
    parser.add_argument("-lr_fix", help="learning rate of unet", type=float, default=1e-3)
    parser.add_argument("-offdecay", help="whether not to use learning rate decay for unet", action='store_false')
    parser.add_argument("-decay_e", help="the epochs to decay the unet learning rate", type=int, default=50)
    parser.add_argument("-apdx", help="the appendix to the checkpoint", type=str, default='point')
    parser.add_argument("-d1", help="whether to apply outer space discriminator", action='store_true')
    parser.add_argument("-d2", help="whether to apply entropy discriminator", action='store_true')
    parser.add_argument("-d4", help='whether to use pointnet', action='store_true')
    parser.add_argument("-d1lr", help="the learning rate for outer space discriminator", type=float, default=2.5e-5)
    parser.add_argument("-d2lr", help="the learning rate for entropy discriminator", type=float, default=2.5e-5)
    parser.add_argument("-d4lr", help="the learning rate for pointnet discriminator", type=float, default=2.5e-5)
    parser.add_argument("-dr", help="the ratio of the discriminators loss for the unet", type=float, default=.01)
    parser.add_argument("-wp", help="the weight for the loss of the point net ", type=float, default=1.)
    parser.add_argument("-data_dir", help="the directory to the data", type=str,
                        default='../../project_cross_modality/Working_Model/input_aug/')
    parser.add_argument("--scratch", help='Whether to transfer data to /scratch', action='store_true')
    parser.add_argument('--num-workers', help='number of workers to load data', type=int, default=2)
    parser.add_argument('--eval_bs', help='batch size during evaluation', type=int, default=32)
    parser.add_argument('--weight_dir', help='the directory to the segmentor weights', type=str, default=None)
    parser.add_argument('--d1_weight_dir', help='the directory to the d1 weights', type=str, default=None)
    parser.add_argument('--d2_weight_dir', help='the directory to the d2 weights', type=str, default=None)
    parser.add_argument('--d4_weight_dir', help='the directory to the d4 weights', type=str, default=None)
    parser.add_argument('--mode', help='', type=str, default='oneshot')
    parser.add_argument('--pat_id', help='', type=int, default=None)
    parser.add_argument('--slice_id', help='', type=int, default=None)
    parser.add_argument('--toggle_klc', help='Whether to apply keep_largest_component in evaluation during training.',
                        action='store_false')
    parser.add_argument("--momentum", type=float, default=.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument('-t2', help='Whether to include t2 in the source data.', action='store_true')

    args = parser.parse_args()
    return args


def get_appendix(args):
    appendix = args.apdx + '.lr{}'.format(args.lr)
    # if not (args.d1 or args.d2 or args.d4):
    appendix += '.{}'.format(args.mode)
    if args.sgd:
        appendix += '.sgd'
    else:
        appendix += '.adam'
    if args.pat_id is not None:
        appendix += '.pat{}'.format(args.pat_id)
    if args.slice_id is not None and args.mode == 'oneshot':
        appendix += '.slc{}'.format(args.slice_id)
    if args.d1:
        appendix += '.d1lr{}'.format(args.d1lr)
    if args.d2:
        appendix += '.d2lr{}'.format(args.d2lr)
    if args.d4:
        appendix += '.d4lr{}'.format(args.d4lr)
    appendix += '.dr{}'.format(args.dr)
    if not args.offdecay:
        appendix += '.offdecay'
    if args.decay_e != 50:
        appendix += '.decay_e{}'.format(args.decay_e)
    if args.wp != 1.:
        appendix += '.wp{}'.format(args.wp)
    if args.t2:
        appendix += '.t2'
    return appendix


def get_optimizers(args, model_gen, model_dis1, model_dis2, model_dis4):
    if args.sgd:
        optim_gen = torch.optim.SGD(model_gen.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optim_gen = torch.optim.Adam(
            model_gen.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99)
        )
    optim_dis1 = None
    if args.d1:
        optim_dis1 = torch.optim.SGD(
            model_dis1.parameters(),
            lr=args.d1lr,
            momentum=.99,
            weight_decay=.0005
        )
    optim_dis2 = None
    if args.d2:
        optim_dis2 = torch.optim.SGD(
            model_dis2.parameters(),
            lr=args.d2lr,
            momentum=.99,
            weight_decay=0.0005
        )
    optim_dis4 = None
    if args.d4:
        optim_dis4 = torch.optim.SGD(
            model_dis4.parameters(),
            lr=args.d4lr,
            momentum=.99,
            weight_decay=0.0005
        )
    print('Optimizers created.')
    return optim_gen, optim_dis1, optim_dis2, optim_dis4


def get_models(args):
    from networks.unet import Segmentation_model_Point
    from networks.GAN import UncertaintyDiscriminator
    from networks.PointNetCls import PointNetCls
    model_gen = Segmentation_model_Point(filters=32, pointnet=args.d4)
    model_gen.cuda()

    model_dis1 = None
    if args.d1:
        model_dis1 = UncertaintyDiscriminator(in_channel=4).cuda()
    model_dis2 = None
    if args.d2:
        model_dis2 = UncertaintyDiscriminator(in_channel=4).cuda()
    model_dis4 = None
    if args.d4:
        model_dis4 = PointNetCls().cuda()
    print("model created")
    return model_gen, model_dis1, model_dis2, model_dis4


def get_model_checkpoints(appendix, args):
    from utils.callbacks import ModelCheckPointCallback
    root_directory = '../weights/'
    if not os.path.exists(root_directory):
        os.mkdir(root_directory)
    weight_dir = root_directory + 'unet_model_checkpoint_{}.pt'.format(appendix)
    best_weight_dir = root_directory + 'best_unet_model_checkpoint_{}.pt'.format(appendix)
    modelcheckpoint_unet = ModelCheckPointCallback(n_epochs=args.e, save_best=True,
                                                   mode="max",
                                                   best_model_dir=best_weight_dir,
                                                   save_last_model=True,
                                                   model_name=weight_dir,
                                                   entire_model=False)

    modelcheckpoint_dis1 = None
    if args.d1:
        d1_weight_dir = root_directory + 'out_dis_{}.pt'.format(appendix)
        best_d1_weight_dir = root_directory + 'best_out_dis_{}.pt'.format(appendix)
        modelcheckpoint_dis1 = ModelCheckPointCallback(n_epochs=args.e,
                                                       mode="max",
                                                       best_model_dir=best_d1_weight_dir,
                                                       save_last_model=True,
                                                       model_name=d1_weight_dir,
                                                       entire_model=False)
    modelcheckpoint_dis2 = None
    if args.d2:
        d2_weight_dir = root_directory + 'entropy_dis_{}.pt'.format(appendix)
        best_d2_weight_dir = root_directory + 'best_entropy_dis_{}.pt'.format(appendix)
        modelcheckpoint_dis2 = ModelCheckPointCallback(n_epochs=args.e,
                                                       mode="max",
                                                       best_model_dir=best_d2_weight_dir,
                                                       save_last_model=True,
                                                       model_name=d2_weight_dir,
                                                       entire_model=False)
    modelcheckpoint_dis4 = None
    if args.d4:
        d4_weight_dir = root_directory + 'point_dis_{}.pt'.format(appendix)
        best_d4_weight_dir = root_directory + 'best_point_dis_{}.pt'.format(appendix)
        modelcheckpoint_dis4 = ModelCheckPointCallback(n_epochs=args.e,
                                                       mode="max",
                                                       best_model_dir=best_d4_weight_dir,
                                                       save_last_model=True,
                                                       model_name=d4_weight_dir,
                                                       entire_model=False)
    print('model checkpoints created.')
    if args.load_weight:
        try:
            model_gen.load_state_dict(torch.load(args.weight_dir))
        except:
            model_gen.load_state_dict(torch.load(args.weight_dir)['model_state_dict'])
        print('Segmentor weights loaded.')
        if args.d1:
            try:
                model_dis1.load_state_dict(torch.load(args.d1_weight_dir))
            except:
                model_dis1.load_state_dict(torch.load(args.d1_weight_dir)['model_state_dict'])
            print('D1 weights loaded')
        if args.d2:
            try:
                model_dis2.load_state_dict(torch.load(args.d2_weight_dir))
            except:
                model_dis2.load_state_dict(torch.load(args.d2_weight_dir)['model_state_dict'])
            print('D2 weights loaded')
        if args.d4:
            try:
                model_dis4.load_state_dict(torch.load(args.d4_weight_dir))
            except:
                model_dis4.load_state_dict(torch.load(args.d4_weight_dir)['model_state_dict'])
            print('D4 weights loaded')

    return modelcheckpoint_unet, modelcheckpoint_dis1, modelcheckpoint_dis2, modelcheckpoint_dis4


def print_epoch_result(train_result, lge_dice, epoch, args):
    epoch_len = len(str(epoch))
    seg_loss = train_result["seg_loss"]
    if args.d4:
        ver_s_loss, ver_t_loss = train_result['ver_s_loss'], train_result['ver_t_loss']
    print_msg_line1 = f'train_loss: {seg_loss:.3f} '
    if args.d4:
        print_msg_line1 += f'vertex_s_loss: {ver_s_loss:.5f}, vertex_t_loss: {ver_t_loss:.5f} '
    print_msg_line2 = f'lge_dice: {lge_dice:.3f} '
    if args.d1:
        dis1_acc1, dis1_acc2 = train_result["dis1_acc1"], train_result['dis1_acc2']
        print_msg_line2 += f'disctor1_train_acc1: {dis1_acc1: .3f} ' + f'disctor1_train_acc2: {dis1_acc2: .3f} '
    if args.d2:
        dis2_acc1, dis2_acc2 = train_result["dis2_acc1"], train_result['dis2_acc2']
        print_msg_line2 += f'disctor2_train_acc1: {dis2_acc1: .3f} ' + f'disctor2_train_acc2: {dis2_acc2: .3f} '
    if args.d4:
        dis4_acc1, dis4_acc2 = train_result["dis4_acc1"], train_result['dis4_acc2']
        print_msg_line2 += f'disctor4_train_acc1: {dis4_acc1: .3f} ' + f'disctor4_train_acc2: {dis4_acc2: .3f} '

    print_msg_line1 = f'[{epoch:>{epoch_len}}/{args.e:>{epoch_len}}] ' + print_msg_line1
    print_msg_line2 = ' ' * (2 * epoch_len + 4) + print_msg_line2
    print(print_msg_line1)
    print(print_msg_line2)


if __name__ == '__main__':

    model_gen, model_dis1, model_dis2, model_dis4 = get_models(args=get_arguments())
    print('12')
