# PyTorch includes
import torch
from torch.nn import BCELoss
import torch.nn.functional as F
import kornia

from datetime import datetime
import tqdm
import numpy as np
import os

from networks.unet import Segmentation_model_Point
from networks.GAN import UncertaintyDiscriminator
from networks.PointNetCls import PointNetCls

from utils.callbacks import ModelCheckPointCallback
from data_generator_mscmrseg import ImageProcessor, DataGenerator_PointNet
from utils.utils import soft_to_hard_pred
from utils.loss import jaccard_loss, batch_NN_loss
from utils.metric import evaluate, dice_coef_multilabel
from utils.timer import timeit

dic_loss = kornia.losses.DiceLoss()


def get_generators(ids_train, ids_valid, ids_train_lge, ids_valid_lge, batch_size=16, n_samples=1000, crop_size=224):
    trainA_generator = DataGenerator_PointNet(df=ids_train, channel="channel_first", apply_noise=True, phase="train",
                                              apply_online_aug=args.aug, aug2=args.aug2,
                                              batch_size=batch_size, source="source", crop_size=crop_size,
                                              n_samples=n_samples, data_dir=args.data_dir)
    validA_generator = DataGenerator_PointNet(df=ids_valid, channel="channel_first", apply_noise=False, phase="valid",
                                              apply_online_aug=False,
                                              batch_size=batch_size, source="source", crop_size=crop_size,
                                              n_samples=-1, data_dir=args.data_dir)
    trainB_generator = DataGenerator_PointNet(df=ids_train_lge, channel="channel_first", apply_noise=True,
                                              phase="train",
                                              apply_online_aug=args.aug, aug2=args.aug2,
                                              batch_size=batch_size, source="target", crop_size=crop_size,
                                              n_samples=n_samples, data_dir=args.data_dir)
    validB_generator = DataGenerator_PointNet(df=ids_valid_lge, channel="channel_first", apply_noise=False,
                                              phase="valid",
                                              apply_online_aug=False,
                                              batch_size=batch_size, source="target", crop_size=crop_size,
                                              n_samples=-1, data_dir=args.data_dir)
    testB_generator = DataGenerator_PointNet(df=ids_train_lge, channel="channel_first", apply_noise=True, phase="train",
                                             apply_online_aug=False,
                                             batch_size=batch_size, source="target", crop_size=crop_size,
                                             n_samples=-1, data_dir=args.data_dir)
    return iter(trainA_generator), iter(validA_generator), iter(trainB_generator), iter(validB_generator), iter(
        testB_generator)


def valid_model_with_one_dataset(seg_model, data_generator, hd=False):
    """
    to valid the segmentation model with one data set
    :param seg_model: the segmentation model
    :param data_generator: the data generator
    :param hd: whether to calculate Hausdorff Distance
    :return: the result dictionary
    """
    seg_model.eval()
    dice_list = []
    loss_list = []
    vert_loss_list = []
    hd_list = []
    with torch.no_grad():
        for x_batch, y_batch, z_batch in data_generator:
            prediction, _, vertS = seg_model(torch.tensor(x_batch).cuda())
            l1 = BCELoss()(torch.sigmoid(prediction), torch.tensor(y_batch, dtype=torch.float32).cuda())
            l2 = jaccard_loss(logits=torch.sigmoid(prediction), true=torch.tensor(y_batch, dtype=torch.float32).cuda(), activation=False)
            l3 = 0
            if args.d4:
                l3 = batch_NN_loss(x=vertS, y=torch.tensor(z_batch).cuda())
                vert_loss_list.append(l3.item())
            else:
                vert_loss_list.append(-1)

            l = l1 + l2 + l3
            loss_list.append(l.item())

            y_pred = prediction.cpu().detach().numpy()
            y_pred = soft_to_hard_pred(y_pred, 1)
            y_pred = np.moveaxis(y_pred, 1, -1)
            y_pred = np.argmax(y_pred, axis=-1)
            y_batch = np.moveaxis(y_batch, 1, -1)
            y_batch = np.argmax(y_batch, axis=-1)
            # y_pred = keep_largest_connected_components(mask=y_pred)
            result = evaluate(img_pred=y_pred, img_gt=y_batch, apply_hd=hd, apply_asd=False)
            dice_list.append((result["lv"][0] + result["myo"][0] + result["rv"][0]) / 3.)
            if hd:
                hd_list.append((result["lv"][1] + result["myo"][1] + result["rv"][1]) / 3.)
    output = {}
    output["dice"] = np.mean(np.array(dice_list))
    output["loss"] = np.mean(np.array(loss_list))
    output["valid_vert_loss"] = np.mean(np.array(vert_loss_list))

    if hd:
        output["hd"] = np.mean(np.array(hd_list))
    return output


def valid_model(seg_model, validA_iterator, validB_iterator, testB_generator):
    """
    to validate the segmentation model with validation and test set
    :param seg_model: the segmentation model
    :param validA_iterator: source validation set
    :param validB_iterator: target validation set
    :param testB_generator: taret test set
    :return: the result dictionary
    """
    valid_result = {}
    seg_model.eval()

    print("start to valid")

    output = valid_model_with_one_dataset(seg_model=seg_model, data_generator=validA_iterator, hd=False)
    val_dice = output["dice"]
    val_loss = output['loss']
    valid_vert_loss = output['valid_vert_loss']

    output = valid_model_with_one_dataset(seg_model=seg_model, data_generator=validB_iterator, hd=False)
    val_lge_dice = output['dice']
    val_lge_loss = output['loss']
    # val_lge_hd.append(output["hd"])

    output = valid_model_with_one_dataset(seg_model=seg_model, data_generator=testB_generator, hd=False)
    test_lge_dice = output['dice']
    test_lge_loss = output['loss']

    # test_lge_hd.append(output["hd"])
    valid_result["val_dice"] = val_dice
    valid_result['val_loss'] = val_loss
    valid_result['valid_vert_loss'] = valid_vert_loss
    valid_result['val_lge_dice'] = val_lge_dice
    valid_result['val_lge_loss'] = val_lge_loss
    valid_result['test_lge_dice'] = test_lge_dice
    valid_result['test_lge_loss'] = test_lge_loss

    return valid_result


@timeit
def train_epoch(model_gen, model_dis2, model_dis4, model_dis1=None,
                optim_gen=None, optim_dis2=None, optim_dis4=None, optim_dis1=None,
                trainA_iterator=None, trainB_iterator=None):
    """
    train the segmentation model for one epoch
    :param model_gen: the segmentation model
    :param model_dis2: the entropy discriminator model
    :param model_dis4: the point cloud discriminator model
    :param model_dis1: the output space discriminator model
    :param optim_gen: the optimizer for the segmentaton model
    :param optim_dis2: the optimizer for the entropy discriminator
    :param optim_dis4: the optimizer for the point cloud discriminator
    :param optim_dis1: the optimizer for the output space discriminator
    :param trainA_iterator: the source training data generator
    :param trainB_iterator:  the target training data generator
    :return: the result dictionary
    """
    source_domain_label = 1
    target_domain_label = 0
    smooth = 1e-7
    model_gen.train()
    if args.d1:
        model_dis1.train()
    if args.d2:
        model_dis2.train()
    if args.d4:
        model_dis4.train()

    train_result = {}

    running_seg_loss = []
    vertex_source_loss = []
    vertex_target_loss = []
    seg_dice = []

    running_adv_diff_loss = []
    running_dis_diff_loss = []

    d1_acc1, d1_acc2, d2_acc1, d2_acc2, d4_acc1, d4_acc2 = [], [], [], [], [], []

    for (imgA, maskA, vertexA), (imgB, _, vertexB) in zip(trainA_iterator, trainB_iterator):
        if args.d1:
            optim_dis1.zero_grad()
            for param in model_dis1.parameters():
                param.requires_grad = False
        if args.d2:
            optim_dis2.zero_grad()
            for param in model_dis2.parameters():
                param.requires_grad = False
        if args.d4:
            optim_dis4.zero_grad()
            for param in model_dis4.parameters():
                param.requires_grad = False
        optim_gen.zero_grad()
        for param in model_gen.parameters():
            param.requires_grad = True

        # 1. train the segmentation model with random source images in supervised manner
        oS, oS2, vertS = model_gen(torch.tensor(imgA).cuda())
        loss_seg = BCELoss()(torch.sigmoid(oS), torch.tensor(maskA, dtype=torch.float32).cuda())
        loss_seg2 = jaccard_loss(logits=torch.sigmoid(oS), true=torch.tensor(maskA, dtype=torch.float32).cuda(), activation=False)
        loss_seg3 = 0
        if args.d4:
            loss_seg3 = batch_NN_loss(x=vertS, y=torch.tensor(vertexA).cuda())
            vertex_source_loss.append(loss_seg3.item())

        loss_seg1 = loss_seg + loss_seg2 + args.wp * loss_seg3

        running_seg_loss.append((loss_seg + loss_seg2).item())

        loss_seg1.backward()

        y_pred = soft_to_hard_pred(oS.cpu().detach().numpy(), 1)
        seg_dice.append(dice_coef_multilabel(y_true=maskA, y_pred=y_pred, channel='channel_first'))

        # 2. train the segmentation model to fool the discriminators
        oT, oT2, vertT = model_gen(torch.tensor(imgB).cuda())
        loss_adv_diff2 = 0
        if args.d2:
            uncertainty_mapT = -1.0 * torch.sigmoid(oT) * torch.log(torch.sigmoid(oT) + smooth)
            D_out2 = model_dis2(uncertainty_mapT)
            loss_adv_diff2 = args.dr * F.binary_cross_entropy_with_logits(D_out2,
                                                                          torch.FloatTensor(D_out2.data.size()).fill_(
                                                                              source_domain_label).cuda())

        loss_adv_diff_point = 0
        if args.d4:
            loss_vert_target = batch_NN_loss(x=vertT, y=torch.tensor(vertexB).cuda())
            vertex_target_loss.append(loss_vert_target.item())
            D_out4 = model_dis4(vertT.transpose(2, 1))[0]
            loss_adv_diff_point = args.dr * F.binary_cross_entropy_with_logits(D_out4, torch.FloatTensor(D_out4.data.size()).fill_(
                source_domain_label).cuda())

        loss_adv_diff1 = 0
        if args.d1:
            D_out1 = model_dis1(oT)
            loss_adv_diff1 = args.dr * F.binary_cross_entropy_with_logits(D_out1,
                                                                          torch.FloatTensor(D_out1.data.size()).fill_(
                                                                              source_domain_label).cuda())

        loss_adv_diff = loss_adv_diff2 + loss_adv_diff_point + loss_adv_diff1
        running_adv_diff_loss.append(loss_adv_diff.item())

        loss_adv_diff.backward()
        optim_gen.step()

        # 3. train the discriminators with images from source domain
        if args.d1:
            for param in model_dis1.parameters():
                param.requires_grad = True
        if args.d2:
            for param in model_dis2.parameters():
                param.requires_grad = True
        if args.d4:
            for param in model_dis4.parameters():
                param.requires_grad = True
        for param in model_gen.parameters():
            param.requires_grad = False

        oS = oS.detach()
        oT = oT.detach()
        if args.d2:
            uncertainty_mapS = -1.0 * torch.sigmoid(oS) * torch.log(torch.sigmoid(oS) + smooth)
            D_out2 = model_dis2(uncertainty_mapS)
            loss_D_same2 = F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(
                source_domain_label).cuda())
            loss_D_same2.backward()
            D_out2 = torch.sigmoid(D_out2.detach()).cpu().numpy()
            D_out2 = np.where(D_out2 >= .5, 1, 0)
            d2_acc1.append(np.mean(D_out2))

        if args.d1:
            D_out1 = model_dis1(oS)
            loss_D_same1 = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(
                source_domain_label).cuda())
            loss_D_same1.backward()
            D_out1 = torch.sigmoid(D_out1.detach()).cpu().numpy()
            D_out1 = np.where(D_out1 >= .5, 1, 0)
            d1_acc1.append(np.mean(D_out1))

        if args.d4:
            vertS = vertS.detach()
            D_out4 = model_dis4(vertS.transpose(2, 1))[0]
            loss_D_same4 = F.binary_cross_entropy_with_logits(D_out4, torch.FloatTensor(D_out4.data.size()).fill_(
                source_domain_label).cuda())
            loss_D_same4.backward()
            D_out4 = torch.sigmoid(D_out4.detach()).cpu().numpy()
            D_out4 = np.where(D_out4 >= .5, 1, 0)
            d4_acc1.append(np.mean(D_out4))

        # 4. train the discriminators with images from different domain
        if args.d2:
            D_out2 = model_dis2(uncertainty_mapT.detach())
            loss_D_diff2 = F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(
                target_domain_label).cuda())
            loss_D_diff2.backward()
            running_dis_diff_loss.append(loss_D_diff2.item())
            D_out2 = torch.sigmoid(D_out2.detach()).cpu().numpy()
            D_out2 = np.where(D_out2 >= .5, 1, 0)
            d2_acc2.append(1 - np.mean(D_out2))

        if args.d1:
            D_out1 = model_dis1(oT)
            loss_D_diff1 = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(
                target_domain_label).cuda())
            loss_D_diff1.backward()
            D_out1 = torch.sigmoid(D_out1.detach()).cpu().numpy()
            D_out1 = np.where(D_out1 >= .5, 1, 0)
            d1_acc2.append(1 - np.mean(D_out1))

        if args.d4:
            vertT = vertT.detach()
            D_out4 = model_dis4(vertT.transpose(2, 1))[0]
            loss_D_diff_4 = F.binary_cross_entropy_with_logits(D_out4, torch.FloatTensor(D_out4.data.size()).fill_(
                target_domain_label).cuda())
            # loss_D_diff = loss_D_diff_1 + loss_D_diff_2
            loss_D_diff_4.backward()
            D_out4 = torch.sigmoid(D_out4.detach()).cpu().numpy()
            D_out4 = np.where(D_out4 >= .5, 1, 0)
            d4_acc2.append(1 - np.mean(D_out4))

        # 5. update parameters
        if args.d1:
            optim_dis1.step()
        if args.d2:
            optim_dis2.step()
        if args.d4:
            optim_dis4.step()

    train_result["seg_loss"] = np.mean(np.array(running_seg_loss))
    train_result['seg_dice'] = np.mean(np.array(seg_dice))
    if args.d2:
        train_result['dis2_acc1'] = np.mean(np.array(d2_acc1))
        train_result['dis2_acc2'] = np.mean(np.array(d2_acc2))
    if args.d1:
        train_result['dis1_acc1'] = np.mean(np.array(d1_acc1))
        train_result['dis1_acc2'] = np.mean(np.array(d1_acc2))
    if args.d4:
        train_result['dis4_acc1'] = np.mean(np.array(d4_acc1))
        train_result['dis4_acc2'] = np.mean(np.array(d4_acc2))
    train_result['ver_s_loss'] = np.mean(np.array(vertex_source_loss))
    train_result['ver_t_loss'] = np.mean(np.array(vertex_target_loss))
    return train_result


def print_epoch_result(train_result, valid_result, epoch, max_epochs):
    epoch_len = len(str(max_epochs))
    seg_loss, seg_dice, ver_s_loss, ver_t_loss = train_result["seg_loss"], train_result['seg_dice'], train_result['ver_s_loss'], train_result['ver_t_loss']
    val_dice, val_loss, val_lge_dice, val_lge_loss, test_lge_dice, test_lge_loss, valid_vert_loss = valid_result[
                                                                                                        "val_dice"], \
                                                                                                    valid_result[
                                                                                                        'val_loss'], \
                                                                                                    valid_result[
                                                                                                        'val_lge_dice'], \
                                                                                                    valid_result[
                                                                                                        'val_lge_loss'], \
                                                                                                    valid_result[
                                                                                                        'test_lge_dice'], \
                                                                                                    valid_result[
                                                                                                        'test_lge_loss'], \
                                                                                                    valid_result[
                                                                                                        'valid_vert_loss']

    print_msg_line1 = f'valid_loss: {val_loss:.5f} ' + f'valid_lge_loss: {val_lge_loss:.5f} ' + f'test_lge_loss: {test_lge_loss:.5f} '
    if args.d4:
        print_msg_line1 += f'vertex_s_loss: {ver_s_loss:.5f}, vertex_t_loss: {ver_t_loss:.5f} '
    print_msg_line2 = f'valid_dice: {val_dice:.5f} ' + \
                      f'valid_lge_dice: {val_lge_dice:.5f} ' + \
                      f'test_lge_dice: {test_lge_dice:.5f} ' + f'valid_vert_loss: {valid_vert_loss:.5f} '

    print_msg_line1 = f'train_loss: {seg_loss:.5f} ' + print_msg_line1
    print_msg_line2 = f'train_dice: {seg_dice:.5f} ' + print_msg_line2
    if args.d1:
        dis1_acc1, dis1_acc2 = train_result["dis1_acc1"], train_result['dis1_acc2']
        print_msg_line2 += f'disctor1_train_acc1: {dis1_acc1: 5f} ' + f'disctor1_train_acc2: {dis1_acc2: 5f} '
    if args.d2:
        dis2_acc1, dis2_acc2 = train_result["dis2_acc1"], train_result['dis2_acc2']
        print_msg_line2 += f'disctor2_train_acc1: {dis2_acc1: 5f} ' + f'disctor2_train_acc2: {dis2_acc2: 5f} '
    if args.d4:
        dis4_acc1, dis4_acc2 = train_result["dis4_acc1"], train_result['dis4_acc2']
        print_msg_line2 += f'disctor4_train_acc1: {dis4_acc1: 5f} ' + f'disctor4_train_acc2: {dis4_acc2: 5f} '

    print_msg_line1 = f'[{epoch:>{epoch_len}}/{max_epochs:>{epoch_len}}] ' + print_msg_line1
    print_msg_line2 = ' ' * (2 * epoch_len + 4) + print_msg_line2
    print(print_msg_line1)
    print(print_msg_line2)


@timeit
def main(batch_size=24, n_samples=1000, n_epochs=200):
    ids_train = ImageProcessor.split_data("./../input_aug/aug_trainA.csv")
    ids_valid = ImageProcessor.split_data("./../input_aug/testA.csv")
    ids_train_lge = ImageProcessor.split_data('./../input_aug/aug_trainB.csv')
    ids_valid_lge = ImageProcessor.split_data('./../input_aug/testB.csv')
    print("Trainining on {} trainA, {} trainB, validating on {} testA and {} testB samples...!!".format(len(ids_train),
                                                                                                        len(
                                                                                                            ids_train_lge),
                                                                                                        len(ids_valid),
                                                                                                        len(
                                                                                                            ids_valid_lge)))

    trainA_iterator, \
    validA_iterator, \
    trainB_iterator, validB_iterator, testB_generator = get_generators(ids_train,
                                                                       ids_valid,
                                                                       ids_train_lge,
                                                                       ids_valid_lge,
                                                                       batch_size=batch_size,
                                                                       n_samples=n_samples,
                                                                       crop_size=224)

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
    lr_gen = args.lr

    the_epoch = 0
    best_valid_lge_dice = 0
    best_train_result = {}
    best_valid_result = {}
    # create model checkpoint instances
    # create directory for the weights
    root_directory = '../weights/'
    if not os.path.exists(root_directory):
        os.mkdir(root_directory)
    weight_dir = root_directory + 'unet_model_checkpoint_{}.pt'.format(appendix)
    best_weight_dir = root_directory + 'best_unet_model_checkpoint_{}.pt'.format(appendix)
    modelcheckpoint_unet = ModelCheckPointCallback(n_epochs=n_epochs, save_best=True,
                                                   mode="max",
                                                   best_model_name=best_weight_dir,
                                                   save_last_model=True,
                                                   model_name=weight_dir,
                                                   entire_model=False)
    if args.d1:
        d1_weight_dir = root_directory + 'out_dis_{}.pt'.format(appendix)
        best_d1_weight_dir = root_directory + 'best_out_dis_{}.pt'.format(appendix)
        modelcheckpoint_dis1 = ModelCheckPointCallback(n_epochs=n_epochs,
                                                       mode="max",
                                                       best_model_name=best_d1_weight_dir,
                                                       save_last_model=True,
                                                       model_name=d1_weight_dir,
                                                       entire_model=False)
    if args.d2:
        d2_weight_dir = root_directory + 'entropy_dis_{}.pt'.format(appendix)
        best_d2_weight_dir = root_directory + 'best_entropy_dis_{}.pt'.format(appendix)
        modelcheckpoint_dis2 = ModelCheckPointCallback(n_epochs=n_epochs,
                                                       mode="max",
                                                       best_model_name=best_d2_weight_dir,
                                                       save_last_model=True,
                                                       model_name=d2_weight_dir,
                                                       entire_model=False)
    if args.d4:
        d4_weight_dir = root_directory + 'point_dis_{}.pt'.format(appendix)
        best_d4_weight_dir = root_directory + 'best_point_dis_{}.pt'.format(appendix)
        modelcheckpoint_dis4 = ModelCheckPointCallback(n_epochs=n_epochs,
                                                       mode="max",
                                                       best_model_name=best_d4_weight_dir,
                                                       save_last_model=True,
                                                       model_name=d4_weight_dir,
                                                       entire_model=False)

    if args.load_weight:
        model_gen.load_state_dict(torch.load(weight_dir))
        if args.d1:
            model_dis1.load_state_dict(torch.load(d1_weight_dir))
        if args.d2:
            model_dis2.load_state_dict(torch.load(d2_weight_dir))
        if args.d4:
            model_dis4.load_state_dict(torch.load(d4_weight_dir))
        valid_result = valid_model(seg_model=model_gen, validA_iterator=validA_iterator,
                                   validB_iterator=validB_iterator, testB_generator=testB_generator)
        val_dice, val_loss, val_lge_dice, val_lge_loss, test_lge_dice, test_lge_loss = valid_result["val_dice"], \
                                                                                       valid_result['val_loss'], \
                                                                                       valid_result['val_lge_dice'], \
                                                                                       valid_result['val_lge_loss'], \
                                                                                       valid_result['test_lge_dice'], \
                                                                                       valid_result['test_lge_loss']
        print_msg_line1 = f'valid_loss: {val_loss:.5f} ' + f'valid_lge_loss: {val_lge_loss:.5f} ' + f'test_lge_loss: {test_lge_loss:.5f} '
        print_msg_line2 = f'valid_dice: {val_dice:.5f} ' + \
                          f'valid_lge_dice: {val_lge_dice:.5f} ' + \
                          f'test_lge_dice: {test_lge_dice:.5f} '
        print(print_msg_line1)
        print(print_msg_line2)
    print("Training started....!")

    seg_dice, seg_loss, d2_acc1, d2_acc2 = [], [], [], []
    pcloud_s_loss, pcloud_t_loss = [], []
    d1_acc1, d1_acc2, d4_acc1, d4_acc2 = [], [], [], []
    val_dice, val_loss, val_lge_dice, val_lge_loss, test_lge_dice, test_lge_loss = [], [], [], [], [], []
    seg_lr, disctor2_lr = [], []

    for epoch in tqdm.trange(n_epochs, desc='Train', ncols=80):
        train_result = train_epoch(model_gen=model_gen, model_dis2=model_dis2,
                                   model_dis4=model_dis4, model_dis1=model_dis1,
                                   optim_gen=optim_gen, optim_dis2=optim_dis2,
                                   optim_dis4=optim_dis4, optim_dis1=optim_dis1,
                                   trainA_iterator=trainA_iterator, trainB_iterator=trainB_iterator)
        seg_loss.append(train_result["seg_loss"])
        seg_dice.append(train_result['seg_dice'])
        if args.d2:
            d2_acc1.append(train_result['dis2_acc1'])
            d2_acc2.append(train_result['dis2_acc2'])
        if args.d1:
            d1_acc1.append(train_result['dis1_acc1'])
            d1_acc2.append(train_result['dis1_acc2'])
        if args.d4:
            pcloud_s_loss.append(train_result['ver_s_loss'])
            pcloud_t_loss.append(train_result['ver_t_loss'])
            d4_acc1.append(train_result['dis4_acc1'])
            d4_acc2.append(train_result['dis4_acc2'])

        valid_result = valid_model(seg_model=model_gen, validA_iterator=validA_iterator,
                                   validB_iterator=validB_iterator, testB_generator=testB_generator)
        val_dice.append(valid_result["val_dice"])
        val_loss.append(valid_result['val_loss'])
        val_lge_dice.append(valid_result['val_lge_dice'])
        val_lge_loss.append(valid_result['val_lge_loss'])
        test_lge_dice.append(valid_result['test_lge_dice'])
        test_lge_loss.append(valid_result['test_lge_loss'])

        seg_lr.append(optim_gen.param_groups[0]['lr'])
        if args.d2:
            disctor2_lr.append(optim_dis2.param_groups[0]['lr'])

        print_epoch_result(train_result, valid_result, epoch + 1, n_epochs)

        if best_valid_lge_dice < valid_result["val_lge_dice"]:
            best_valid_lge_dice = valid_result["val_lge_dice"]
            best_train_result = train_result
            best_valid_result = valid_result
            the_epoch = epoch + 1
        if (datetime.now() - start_time).seconds > max_duration:
            epoch = n_epochs - 1
        monitor_score = valid_result["val_lge_dice"]
        modelcheckpoint_unet.step(monitor=monitor_score, model=model_gen, epoch=epoch + 1)
        if args.d1:
            modelcheckpoint_dis1.step(monitor=monitor_score, model=model_dis1, epoch=epoch + 1)
        if args.d2:
            modelcheckpoint_dis2.step(monitor=monitor_score, model=model_dis2, epoch=epoch + 1)
        if args.d4:
            modelcheckpoint_dis4.step(monitor=monitor_score, model=model_dis4, epoch=epoch + 1)
        if (datetime.now() - start_time).seconds > max_duration:
            break
        if args.offdecay:
            if (epoch + 1) % args.decay_e == 0:
                lr_gen = lr_gen * 0.2
                for param_group in optim_gen.param_groups:
                    param_group['lr'] = lr_gen
    print("Best model on epoch {}: train_dice {}, valid_dice {}, lge_dice {}, test_lge_dice {}".format(
        the_epoch, best_train_result['seg_dice'], best_valid_result['val_dice'], best_valid_result['val_lge_dice'],
        best_valid_result['test_lge_dice']))

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(comment=appendix + ".Scr{}".format(np.around(best_valid_lge_dice, 3)))
    i = 1
    print("write a training summary")
    for t_loss, t_dice, v_loss, v_dice, lge_loss, lge_dice, t_lge_loss, t_lge_dice, u_lr in zip(seg_loss,
                                                                                                seg_dice,
                                                                                                val_loss,
                                                                                                val_dice,
                                                                                                val_lge_loss,
                                                                                                val_lge_dice,
                                                                                                test_lge_loss,
                                                                                                test_lge_dice,
                                                                                                seg_lr):
        writer.add_scalar('Loss/Training', t_loss, i)
        writer.add_scalar('Loss/Validation', v_loss, i)
        writer.add_scalar('Loss/Valid_LGE', lge_loss, i)
        writer.add_scalar('Loss/Test_LGE', t_lge_loss, i)
        writer.add_scalar('Dice/Training', t_dice, i)
        writer.add_scalar('Dice/Validation', v_dice, i)
        writer.add_scalar('Dice/LGE', lge_dice, i)
        writer.add_scalar('Dice/T_LGE', t_lge_dice, i)
        writer.add_scalar('Lr/unet', u_lr, i)
        writer.add_scalars('Dice', {'Training': t_dice,
                                    'Validation': v_dice,
                                    'Valid_LGE': lge_dice,
                                    'Test_LGE': t_lge_dice}, i)
        i += 1
    if args.d1:
        i = 1
        for a1, a2 in zip(d1_acc1, d1_acc2):
            writer.add_scalar('d1_acc1', a1, i)
            writer.add_scalar('d1_acc2', a2, i)
            i += 1
    if args.d2:
        i = 1
        for a1, a2 in zip(d2_acc1, d2_acc2):
            writer.add_scalar('d2_acc1', a1, i)
            writer.add_scalar('d2_acc2', a2, i)
            i += 1
    if args.d4:
        i = 1
        for p_s_loss, p_t_loss, a1, a2 in zip(pcloud_s_loss, pcloud_t_loss, d4_acc1, d4_acc2):
            writer.add_scalar('Loss/PointCloudS', p_s_loss, i)
            writer.add_scalar('Loss/PointCloudT', p_t_loss, i)
            writer.add_scalar('d4_acc1', a1, i)
            writer.add_scalar('d4_acc2', a2, i)
            i += 1
    writer.close()


def get_appendix():
    appendix = args.apdx + '.lr{}'.format(args.lr_fix)
    if args.d1:
        appendix += '.d1lr{}'.format(args.d1lr)
    if args.d2:
        appendix += '.d2lr{}'.format(args.d2lr)
    if args.d4:
        appendix += '.d4lr{}'.format(args.d4lr)
    if not args.aug:
        appendix += '.aug'
    if args.aug2:
        appendix += '.aug2'
    if not args.offdecay:
        appendix += '.offdecay'
    if args.decay_e != 50:
        appendix += '.decay_e{}'.format(args.decay_e)
    if args.wp != 1.:
        appendix += '.wp{}'.format(args.wp)
    return appendix


if __name__ == '__main__':
    start_time = datetime.now()
    max_duration = 24 * 3600 - 10 * 60  # 85800 seconds. set the maximum running time to prevent from exceeding time limitation
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("cuda available: {}".format(torch.cuda.is_available()))
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-aug", help='whether not to augment the data', action='store_false')
    parser.add_argument("-aug2", help='whether to augment the data with 2nd method', action='store_true')
    parser.add_argument("-load_weight", help='whether to load weight', action='store_true')
    parser.add_argument("-bs", help="the batch size of training", type=int, default=16)
    parser.add_argument("-ns", help="number of samples per epoch", type=int, default=2000)
    parser.add_argument("-e", help="number of epochs", type=int, default=200)
    parser.add_argument("-lr", help="learning rate of unet", type=float, default=1e-3)
    parser.add_argument("-lr_fix", help="learning rate of unet", type=float, default=1e-3)
    parser.add_argument("-offdecay", help="whether not to use learning rate decay for unet", action='store_false')
    parser.add_argument("-decay_e", help="the epochs to decay the unet learning rate", type=int, default=50)
    parser.add_argument("-apdx", help="the appendix to the checkpoint", type=str, default='train_point_imgaug')
    parser.add_argument("-d1", help="whether to apply outer space discriminator", action='store_true')
    parser.add_argument("-d2", help="whether to apply entropy discriminator", action='store_true')
    parser.add_argument("-d4", help='whether to use pointnet', action='store_true')
    parser.add_argument("-d1lr", help="the learning rate for outer space discriminator", type=float, default=2.5e-5)
    parser.add_argument("-d2lr", help="the learning rate for entropy discriminator", type=float, default=2.5e-5)
    parser.add_argument("-d4lr", help="the learning rate for pointnet discriminator", type=float, default=2.5e-5)
    parser.add_argument("-dr", help="the ratio of the discriminators loss for the unet", type=float, default=.01)
    parser.add_argument("-wp", help="the weight for the loss of the point net ", type=float, default=1.)
    parser.add_argument("-data_dir", help="the directory to the data", type=str, default='./../input_aug/')

    args = parser.parse_args()

    appendix = get_appendix()
    print(appendix)
    # import torch.backends.cudnn as cudnn

    torch.autograd.set_detect_anomaly(True)
    # cudnn.benchmark = True
    # cudnn.enabled = True

    main(batch_size=args.bs, n_samples=args.ns, n_epochs=args.e)
