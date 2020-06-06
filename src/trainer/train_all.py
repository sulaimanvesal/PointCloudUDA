# from datetime import datetime
# import os
import os.path as osp
import tqdm
import numpy as np
# from sklearn.metrics import accuracy_score
# PyTorch includes
import torch
# from torchvision import transforms
# from torch.utils.data import DataLoader
from torch.nn import BCELoss
import kornia
# import torch.nn.functional as F

# Custom includes
# from dataloaders import fundus_dataloader as DL
# from dataloaders import custom_transforms as tr
from networks.deeplabv3 import *
from networks.unet import Segmentation_model_Point
from networks.efficientunet.efficientunet import *
from networks.GAN import BoundaryDiscriminator, UncertaintyDiscriminator
from networks.pointNet_discriminator import PointNetCls

from utils.callbacks import EarlyStoppingCallback, ModelCheckPointCallback
from data_generator import ImageProcessor, DataGenerator_PointNet
from utils_ import soft_to_hard_pred, jaccard_loss, batch_NN_loss
from metric import metrics, dice_coef_multilabel
import os

from timer import timeit

here = osp.dirname(osp.abspath(__file__))
dic_loss = kornia.losses.DiceLoss()


def get_generators(ids_train, ids_valid, ids_train_lge, ids_valid_lge, batch_size=16, n_samples=1000, crop_size=224,
                   mh=False):
    trainA_generator = DataGenerator_PointNet(df=ids_train, channel="channel_first", apply_noise=True, phase="train",
                                              apply_online_aug=True,
                                              batch_size=batch_size, source="source", crop_size=crop_size,
                                              n_samples=n_samples, match_hist=mh)
    validA_generator = DataGenerator_PointNet(df=ids_valid, channel="channel_first", apply_noise=False, phase="valid",
                                              apply_online_aug=False,
                                              batch_size=batch_size, source="source", crop_size=crop_size,
                                              n_samples=-1, match_hist=mh)
    trainB_generator = DataGenerator_PointNet(df=ids_train_lge, channel="channel_first", apply_noise=True,
                                              phase="train",
                                              apply_online_aug=True,
                                              batch_size=batch_size, source="target", crop_size=crop_size,
                                              n_samples=n_samples)
    validB_generator = DataGenerator_PointNet(df=ids_valid_lge, channel="channel_first", apply_noise=False,
                                              phase="valid",
                                              apply_online_aug=False,
                                              batch_size=batch_size, source="target", crop_size=crop_size,
                                              n_samples=-1)
    testB_generator = DataGenerator_PointNet(df=ids_train_lge, channel="channel_first", apply_noise=True, phase="train",
                                             apply_online_aug=False,
                                             batch_size=batch_size, source="target", crop_size=crop_size,
                                             n_samples=-1)
    return iter(trainA_generator), iter(validA_generator), iter(trainB_generator), iter(validB_generator), iter(
        testB_generator)


def valid_model_with_one_dataset(seg_model, data_generator, hd=False):
    seg_model.eval()
    dice_list = []
    loss_list = []
    vert_loss_list = []
    hd_list = []
    with torch.no_grad():
        for x_batch, y_batch, z_batch in data_generator:
            prediction, _, vertS = seg_model(torch.tensor(x_batch).cuda())
            l1 = BCELoss()(torch.sigmoid(prediction), torch.tensor(y_batch, dtype=torch.float32).cuda())
            l2 = jaccard_loss(logits=torch.sigmoid(prediction), true=torch.tensor(np.argmax(y_batch, axis=1)).cuda())
            l3 = 0
            if args.d4:
                l3 = batch_NN_loss(x=vertS, y=torch.tensor(z_batch).cuda())
                vert_loss_list.append(l3.item())
            else:
                vert_loss_list.append(-1)

            if args.d4:
                l = (l1 + l2 + l3)/3
            else:
                l = (l1 + l2)/2
            loss_list.append(l.item())

            y_pred = prediction.cpu().detach().numpy()
            y_pred = soft_to_hard_pred(y_pred, 1)
            y_pred = np.moveaxis(y_pred, 1, -1)
            y_pred = np.argmax(y_pred, axis=-1)
            y_batch = np.moveaxis(y_batch, 1, -1)
            y_batch = np.argmax(y_batch, axis=-1)
            result = metrics(img_pred=y_pred, img_gt=y_batch, apply_hd=hd, apply_asd=False)
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
def train_epoch(epoch_number, model_gen, model_dis2, model_dis3, model_dis4, model_dis1=None,
                optim_gen=None, optim_dis2=None, optim_dis3=None, optim_dis4=None, optim_dis1=None,
                trainA_iterator=None, trainB_iterator=None):
    source_domain_label = 1
    target_domain_label = 0
    smooth = 1e-7
    model_gen.train()
    if args.d1:
        model_dis1.train()
    if args.d3:
        model_dis3.train()
    model_dis2.train()
    if args.d4:
        model_dis4.train()

    train_result = {}

    running_seg_loss = []
    running_ver_loss = []
    seg_dice = []

    running_adv_diff_loss = []
    running_dis_same_loss = []
    running_dis_diff_loss = []

    d1_acc1, d1_acc2, d2_acc1, d2_acc2, d3_acc1, d3_acc2, d4_acc1, d4_acc2 = [], [], [], [], [], [], [], []

    for (imgA, maskA, vertexA), (imgB, maskB, _) in zip(trainA_iterator, trainB_iterator):

        optim_gen.zero_grad()
        if args.d1:
            optim_dis1.zero_grad()
        optim_dis2.zero_grad()
        if args.d3:
            optim_dis3.zero_grad()
        if args.d4:
            optim_dis4.zero_grad()

        # 1. train generator with random images
        if args.d1:
            for param in model_dis1.parameters():
                param.requires_grad = False
        for param in model_dis2.parameters():
            param.requires_grad = False
        if args.d3:
            for param in model_dis3.parameters():
                param.requires_grad = False
        if args.d4:
            for param in model_dis4.parameters():
                param.requires_grad = False
        for param in model_gen.parameters():
            param.requires_grad = True

        oS, oS2, vertS = model_gen(torch.tensor(imgA).cuda())
        loss_seg = BCELoss()(torch.sigmoid(oS), torch.tensor(maskA, dtype=torch.float32).cuda())
        loss_seg2 = jaccard_loss(logits=torch.sigmoid(oS), true=torch.tensor(np.argmax(maskA, axis=1)).cuda())
        loss_seg3 = 0
        if args.d4:
            loss_seg3 = batch_NN_loss(x=vertS, y=torch.tensor(vertexA).cuda())
            running_ver_loss.append(loss_seg3.item())
        if args.d4:
            loss_seg1 = (loss_seg + loss_seg2 + loss_seg3)/3
        else:
            loss_seg1 = (loss_seg + loss_seg2)/2

        running_seg_loss.append(loss_seg1.item())
        loss_seg1.backward()

        y_pred = soft_to_hard_pred(oS.cpu().detach().numpy(), 1)
        seg_dice.append(dice_coef_multilabel(y_true=maskA, y_pred=y_pred, channel='channel_first'))

        oT, oT2, vertT = model_gen(torch.tensor(imgB).cuda())
        uncertainty_mapT = -1.0 * torch.sigmoid(oT) * torch.log(torch.sigmoid(oT) + smooth)
        D_out2 = model_dis2(uncertainty_mapT)
        loss_adv_diff2 = F.binary_cross_entropy_with_logits(D_out2,
                                                                      torch.FloatTensor(D_out2.data.size()).fill_(
                                                                          source_domain_label).cuda())

        loss_adv_diff_point = 0
        if args.d4:
            D_out4 = model_dis4(vertT.transpose(2, 1))[0]
            loss_adv_diff_point = args.dr * F.binary_cross_entropy_with_logits(D_out4, torch.FloatTensor(
                D_out4.data.size()).fill_(
                source_domain_label).cuda())

        loss_adv_diff1 = 0
        if args.d1:
            D_out1 = model_dis1(F.softmax(oT))
            loss_adv_diff1 = F.binary_cross_entropy_with_logits(D_out1,
                                                                          torch.FloatTensor(D_out1.data.size()).fill_(
                                                                              source_domain_label).cuda())

        loss_adv_diff3 = 0
        if args.d3:
            D_out3 = model_dis3(F.softmax(oT2))
            loss_adv_diff3 = F.binary_cross_entropy_with_logits(D_out3,
                                                                          torch.FloatTensor(D_out3.data.size()).fill_(
                                                                              source_domain_label).cuda())

        loss_adv_diff = 0.01 *(loss_adv_diff2 + loss_adv_diff_point + loss_adv_diff1 + loss_adv_diff3)
        running_adv_diff_loss.append(loss_adv_diff.item())

        loss_adv_diff.backward()
        optim_gen.step()

        # 3. train discriminator with images from same domain
        if args.d1:
            for param in model_dis1.parameters():
                param.requires_grad = True
        for param in model_dis2.parameters():
            param.requires_grad = True
        if args.d3:
            for param in model_dis3.parameters():
                param.requires_grad = True
        if args.d4:
            for param in model_dis4.parameters():
                param.requires_grad = True
        for param in model_gen.parameters():
            param.requires_grad = False

        oS = oS.detach()
        uncertainty_mapS = -1.0 * torch.sigmoid(oS) * torch.log(torch.sigmoid(oS) + smooth)
        D_out2 = model_dis2(uncertainty_mapS)
        loss_D_same2 = F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(
            source_domain_label).cuda())
        #loss_D_same2.backward()
        D_out2 = torch.sigmoid(D_out2.detach()).cpu().numpy()
        D_out2 = np.where(D_out2 >= .5, 1, 0)
        d2_acc1.append(np.mean(D_out2))

        if args.d1:
            D_out1 = model_dis1(F.softmax(oS))
            loss_D_same1 = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(
                source_domain_label).cuda())

            loss_D_same = loss_D_same1 + loss_D_same2
            loss_D_same.backward()
            D_out1 = torch.sigmoid(D_out1.detach()).cpu().numpy()
            D_out1 = np.where(D_out1 >= .5, 1, 0)
            d1_acc1.append(np.mean(D_out1))

        if args.d3:
            D_out3 = model_dis3(F.softmax(oS2.detach()))
            loss_D_same3 = F.binary_cross_entropy_with_logits(D_out3, torch.FloatTensor(D_out3.data.size()).fill_(
                source_domain_label).cuda())
            loss_D_same = loss_D_same3 + loss_D_same2
            loss_D_same.backward()
            D_out3 = torch.sigmoid(D_out3.detach()).cpu().numpy()
            D_out3 = np.where(D_out3 >= .5, 1, 0)
            d3_acc1.append(np.mean(D_out3))

        if args.d4:
            vertS = vertS.detach()
            D_out4 = model_dis4(vertS.transpose(2, 1))[0]
            loss_D_same4 = F.binary_cross_entropy_with_logits(D_out4, torch.FloatTensor(D_out4.data.size()).fill_(
                source_domain_label).cuda())

            loss_D_same = loss_D_same4 + loss_D_same2
            loss_D_same.backward()
            D_out4 = torch.sigmoid(D_out4.detach()).cpu().numpy()
            D_out4 = np.where(D_out4 >= .5, 1, 0)
            d4_acc1.append(np.mean(D_out4))

        # loss_D_same1 = loss_D_same1_1 + loss_D_same1_2
        running_dis_same_loss.append(loss_D_same2.item())

        # 4. train discriminator with images from different domain
        uncertainty_mapT = uncertainty_mapT.detach()
        D_out2 = model_dis2(uncertainty_mapT)
        loss_D_diff2 = F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(
            target_domain_label).cuda())
        #loss_D_diff2.backward()
        running_dis_diff_loss.append(loss_D_diff2.item())
        D_out2 = torch.sigmoid(D_out2.detach()).cpu().numpy()
        D_out2 = np.where(D_out2 >= .5, 1, 0)
        d2_acc2.append(1 - np.mean(D_out2))

        if args.d1:
            D_out1 = model_dis1(F.softmax(oT.detach()))
            loss_D_diff1 = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(
                target_domain_label).cuda())

            loss_D_diff = loss_D_diff1 + loss_D_diff2
            loss_D_diff.backward()
            D_out1 = torch.sigmoid(D_out1.detach()).cpu().numpy()
            D_out1 = np.where(D_out1 >= .5, 1, 0)
            d1_acc2.append(1 - np.mean(D_out1))

        if args.d3:
            D_out3 = model_dis3(F.softmax(oT2.detach()))
            loss_D_diff3 = F.binary_cross_entropy_with_logits(D_out3, torch.FloatTensor(D_out3.data.size()).fill_(
                target_domain_label).cuda())
            loss_D_diff = loss_D_diff3 + loss_D_diff2
            loss_D_diff.backward()
            D_out3 = torch.sigmoid(D_out3.detach()).cpu().numpy()
            D_out3 = np.where(D_out3 >= .5, 1, 0)
            d3_acc2.append(1 - np.mean(D_out3))

        if args.d4:
            vertT = vertT.detach()
            D_out4 = model_dis4(vertT.transpose(2, 1))[0]
            loss_D_diff4 = F.binary_cross_entropy_with_logits(D_out4, torch.FloatTensor(D_out4.data.size()).fill_(
                target_domain_label).cuda())
            # loss_D_diff = loss_D_diff_1 + loss_D_diff_2
            loss_D_diff = loss_D_diff4 + loss_D_diff2
            loss_D_diff.backward()
            D_out4 = torch.sigmoid(D_out4.detach()).cpu().numpy()
            D_out4 = np.where(D_out4 >= .5, 1, 0)
            d4_acc2.append(1 - np.mean(D_out4))

        # 5. update parameters
        if args.d1:
            optim_dis1.step()
        optim_dis2.step()
        if args.d3:
            optim_dis3.step()
        if args.d4:
            optim_dis4.step()

    train_result["seg_loss"] = np.mean(np.array(running_seg_loss))
    train_result['seg_dice'] = np.mean(np.array(seg_dice))
    train_result['dis2_acc1'] = np.mean(np.array(d2_acc1))
    train_result['dis2_acc2'] = np.mean(np.array(d2_acc2))
    if args.d1:
        train_result['dis1_acc1'] = np.mean(np.array(d1_acc1))
        train_result['dis1_acc2'] = np.mean(np.array(d1_acc2))
    if args.d3:
        train_result['dis3_acc1'] = np.mean(np.array(d3_acc1))
        train_result['dis3_acc2'] = np.mean(np.array(d3_acc2))
    if args.d4:
        train_result['dis4_acc1'] = np.mean(np.array(d4_acc1))
        train_result['dis4_acc2'] = np.mean(np.array(d4_acc2))
    train_result['ver_loss'] = np.mean(np.array(running_ver_loss))
    return train_result


def print_epoch_result(train_result, valid_result, epoch, max_epochs):
    epoch_len = len(str(max_epochs))
    seg_loss, seg_dice, dis2_acc1, dis2_acc2, ver_loss = train_result["seg_loss"], train_result['seg_dice'], \
                                                         train_result["dis2_acc1"], train_result['dis2_acc2'], \
                                                         train_result['ver_loss']
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
        print_msg_line1 += f'vertex_loss: {ver_loss:.5f} '
    print_msg_line2 = f'valid_dice: {val_dice:.5f} ' + \
                      f'valid_lge_dice: {val_lge_dice:.5f} ' + \
                      f'test_lge_dice: {test_lge_dice:.5f} ' + f'valid_vert_loss: {valid_vert_loss:.5f} '

    print_msg_line1 = f'train_loss: {seg_loss:.5f} ' + print_msg_line1
    print_msg_line2 = f'train_dice: {seg_dice:.5f} ' + print_msg_line2
    print_msg_line2 += f'disctor2_train_acc1: {dis2_acc1: 5f} ' + f'disctor2_train_acc2: {dis2_acc2: 5f} '

    print_msg_line1 = f'[{epoch + 1:>{epoch_len}}/{max_epochs:>{epoch_len}}] ' + print_msg_line1
    print_msg_line2 = ' ' * (2 * epoch_len + 4) + print_msg_line2
    print(print_msg_line1)
    print(print_msg_line2)


@timeit
def main(batch_size=24, n_samples=1000, n_epochs=200):
    ids_train = ImageProcessor.split_data("./../input/trainA.csv")
    ids_valid = ImageProcessor.split_data("./../input/testA.csv")
    ids_train_lge = ImageProcessor.split_data('./../input/trainB.csv')
    ids_valid_lge = ImageProcessor.split_data('./../input/testB.csv')
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
                                                                       crop_size=224,
                                                                       mh=True)

    # 2. model
    # model_gen = DeepLab(num_classes=4, backbone='resnet', output_stride=16,
    #                     sync_bn=False, freeze_bn=False).cuda()

    # 3. model
    if args.model == 'effunet':
        model_gen = get_efficientunet_b2(out_channels=4, concat_input=True, pretrained=True).cuda()
    else:
        model_gen = Segmentation_model_Point(filters=32)

    model_gen.cuda()

    # model_dis = BoundaryDiscriminator().cuda()
    model_dis1 = None
    if args.d1:
        model_dis1 = UncertaintyDiscriminator(in_channel=4).cuda()
    model_dis2 = UncertaintyDiscriminator(in_channel=4).cuda()
    model_dis3 = None
    if args.d3:
        from networks.GAN import OutputDiscriminator
        model_dis3 = OutputDiscriminator(in_channel=4).cuda()
    model_dis4 = None
    if args.d4:
        model_dis4 = PointNetCls(k=2).cuda()

    # 3. optimizer

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
            #momentum=.99,
            #weight_decay=.0005
        )
    optim_dis2 = torch.optim.SGD(
        model_dis2.parameters(),
        lr=args.d2lr,
        #momentum=.99,
        #weight_decay=0.0005
    )
    optim_dis3 = None
    if args.d3:
        optim_dis3 = torch.optim.SGD(
            model_dis3.parameters(),
            lr=args.d3lr,
            #momentum=.99,
            #weight_decay=0.0005
        )
    optim_dis4 = None
    if args.d4:
        optim_dis4 = torch.optim.SGD(
            model_dis4.parameters(),
            lr=args.d4lr,
            #momentum=.99,
            #weight_decay=0.0005
        )
    lr_gen = args.lr

    the_epoch = 0
    best_valid_lge_dice = 0
    best_train_result = {}
    best_valid_result = {}
    # create directory for the weights
    root_directory = '../weights/' + 'DeepLabv3' + '/'
    if not os.path.exists(root_directory):
        os.mkdir(root_directory)
    weight_dir = root_directory + 'unet_model_checkpoint_{}.pt'.format(appendix)
    best_weight_dir = root_directory + 'best_unet_model_checkpoint_{}.pt'.format(appendix)
    modelcheckpoint_unet = ModelCheckPointCallback(save_best=True,
                                                   mode="max",
                                                   best_model_name=best_weight_dir,
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
    d2_weight_dir = root_directory + 'entropy_dis_{}.pt'.format(appendix)
    best_d2_weight_dir = root_directory + 'best_entropy_dis_{}.pt'.format(appendix)
    modelcheckpoint_dis2 = ModelCheckPointCallback(n_epochs=n_epochs,
                                                   mode="max",
                                                   best_model_name=best_d2_weight_dir,
                                                   save_last_model=True,
                                                   model_name=d2_weight_dir,
                                                   entire_model=False)
    if args.d3:
        d3_weight_dir = root_directory + 'feature_dis_{}.pt'.format(appendix)
        best_d3_weight_dir = root_directory + 'best_feature_dis_{}.pt'.format(appendix)
        modelcheckpoint_dis3 = ModelCheckPointCallback(n_epochs=n_epochs,
                                                       mode="max",
                                                       best_model_name=best_d3_weight_dir,
                                                       save_last_model=True,
                                                       model_name=d3_weight_dir,
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
    # model_gen.load_state_dict(torch.load(root_directory + 'unet_multilevel_model_checkpoint_stage2.pt'))
    if args.load_weight:
        model_gen.load_state_dict(torch.load(weight_dir))
        if args.d1:
            model_dis1.load_state_dict(torch.load(d1_weight_dir))
        model_dis2.load_state_dict(torch.load(d2_weight_dir))
        if args.d3:
            model_dis3.load_state_dict(torch.load(d3_weight_dir))
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
    # output = valid_model_with_one_dataset(seg_model=model_gen, data_generator=testB_generator, hd=False)

    seg_dice, seg_loss, d2_acc1, d2_acc2 = [], [], [], []
    d1_acc1, d1_acc2, d3_acc1, d3_acc2, d4_acc1, d4_acc2 = [], [], [], [], [], []
    val_dice, val_loss, val_lge_dice, val_lge_loss, test_lge_dice, test_lge_loss = [], [], [], [], [], []
    seg_lr, disctor2_lr = [], []

    for epoch in tqdm.trange(n_epochs, desc='Train', ncols=80):
        train_result = train_epoch(epoch, model_gen=model_gen, model_dis2=model_dis2, model_dis3=model_dis3,
                                   model_dis4=model_dis4, model_dis1=model_dis1,
                                   optim_gen=optim_gen, optim_dis2=optim_dis2, optim_dis3=optim_dis3,
                                   optim_dis4=optim_dis4, optim_dis1=optim_dis1,
                                   trainA_iterator=trainA_iterator, trainB_iterator=trainB_iterator)
        seg_loss.append(train_result["seg_loss"])
        seg_dice.append(train_result['seg_dice'])
        d2_acc1.append(train_result['dis2_acc1'])
        d2_acc2.append(train_result['dis2_acc2'])
        if args.d1:
            d1_acc1.append(train_result['dis1_acc1'])
            d1_acc2.append(train_result['dis1_acc2'])
        if args.d3:
            d3_acc1.append(train_result['dis3_acc1'])
            d3_acc2.append(train_result['dis3_acc2'])
        if args.d4:
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
        disctor2_lr.append(optim_dis2.param_groups[0]['lr'])

        print_epoch_result(train_result, valid_result, epoch, n_epochs)

        if best_valid_lge_dice < valid_result["val_lge_dice"]:
            the_epoch = epoch + 1
            best_valid_lge_dice = valid_result["val_lge_dice"]
            best_train_result = train_result
            best_valid_result = valid_result
            monitor_score = best_valid_lge_dice
            modelcheckpoint_unet.step(monitor=monitor_score, model=model_gen, epoch=the_epoch)
            if args.d1:
                modelcheckpoint_dis1.step(monitor=monitor_score, model=model_dis1, epoch=the_epoch)
            modelcheckpoint_dis2.step(monitor=monitor_score, model=model_dis2, epoch=the_epoch)
            if args.d3:
                modelcheckpoint_dis3.step(monitor=monitor_score, model=model_dis3, epoch=the_epoch)
            if args.d4:
                modelcheckpoint_dis4.step(monitor=monitor_score, model=model_dis4, epoch=the_epoch)

        if (epoch + 1) % 100 == 0:
            lr_gen = lr_gen * 0.2
            for param_group in optim_gen.param_groups:
                param_group['lr'] = lr_gen
    print("Best model on epoch {}: train_dice {}, valid_dice {}, lge_dice {}, test_lge_dice {}".format(
        the_epoch, best_train_result['seg_dice'], best_valid_result['val_dice'], best_valid_result['val_lge_dice'],
        best_valid_result['test_lge_dice']))

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(comment=appendix)
    i = 1
    print("write a training summary")
    for t_loss, t_dice, v_loss, v_dice, lge_loss, lge_dice, t_lge_loss, t_lge_dice, a1, a2, u_lr, d2_lr in zip(seg_loss,
                                                                                                               seg_dice,
                                                                                                               val_loss,
                                                                                                               val_dice,
                                                                                                               val_lge_loss,
                                                                                                               val_lge_dice,
                                                                                                               test_lge_loss,
                                                                                                               test_lge_dice,
                                                                                                               d2_acc1,
                                                                                                               d2_acc2,
                                                                                                               seg_lr,
                                                                                                               disctor2_lr):
        writer.add_scalar('Loss/Training', t_loss, i)
        writer.add_scalar('Loss/Validation', v_loss, i)
        writer.add_scalar('Loss/LGE', lge_loss, i)
        writer.add_scalar('Loss/T_LGE', t_lge_loss, i)
        writer.add_scalar('Dice/Training', t_dice, i)
        writer.add_scalar('Dice/Validation', v_dice, i)
        writer.add_scalar('Dice/LGE', lge_dice, i)
        writer.add_scalar('Dice/T_LGE', t_lge_dice, i)
        writer.add_scalar('Lr/unet', u_lr, i)
        writer.add_scalar('Lr/d2', d2_lr, i)
        writer.add_scalar('d2_acc1', a1, i)
        writer.add_scalar('d2_acc2', a2, i)
        i += 1
    if args.d1:
        i = 1
        for a1, a2, lr in zip(d1_acc1, d1_acc2):
            writer.add_scalar('d1_acc1', a1, i)
            writer.add_scalar('d1_acc2', a2, i)
            i += 1
    if args.d3:
        i = 1
        for a1, a2, lr in zip(d3_acc1, d3_acc2):
            writer.add_scalar('d3_acc1', a1, i)
            writer.add_scalar('d3_acc2', a2, i)
            i += 1
    if args.d4:
        i = 1
        for a1, a2, lr in zip(d4_acc1, d4_acc2):
            writer.add_scalar('d4_acc1', a1, i)
            writer.add_scalar('d4_acc2', a2, i)
            i += 1
    writer.close()


def get_appendix():
    appendix = args.apdx + '.' + args.model + '.lr{}'.format(args.lr_fix)
    if args.d1:
        appendix += '.d1lr{}'.format(args.d1lr)
    appendix += '.d2lr{}'.format(args.d2lr)
    if args.d3:
        appendix += '.d3lr{}'.format(args.d3lr)
    if args.d4:
        appendix += '.d4lr{}'.format(args.d4lr)
    if args.raug:
        appendix += '.raug'
    if args.drop:
        appendix += '.drop'
    if args.softmax:
        appendix += '.softmax'
    return appendix


if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-raug", help='whether to use raugmented data', action='store_true')
    parser.add_argument("-load_weight", help='whether to load weight', action='store_true')
    parser.add_argument("-bs", help="the batch size of training", type=int, default=16)
    parser.add_argument("-ns", help="number of samples per epoch", type=int, default=2000)
    parser.add_argument("-e", help="number of epochs", type=int, default=400)
    parser.add_argument("-lr", help="learning rate of unet", type=float, default=1e-3)
    parser.add_argument("-lr_fix", help="learning rate of unet", type=float, default=1e-3)
    parser.add_argument("-apdx", help="the appendix to the checkpoint", type=str, default='train_point')
    parser.add_argument("-model", help="the unet model chosen to use", type=str, default='effunet')  # sk, sd
    parser.add_argument("-out_ch", help="the out channels of the first conv layer(only used for effunet)", type=int,
                        default=32)
    parser.add_argument("-d1", help="whether to apply outer space discriminator", action='store_false')
    parser.add_argument("-d3", help="whether to apply feature discriminator", action='store_false')
    parser.add_argument("-d4", help='whether to use pointnet', action='store_true')
    parser.add_argument("-d1lr", help="the learning rate for outer space discriminator", type=float, default=2.5e-5)
    parser.add_argument("-d2lr", help="the learning rate for entropy discriminator", type=float, default=2.5e-5)
    parser.add_argument("-d3lr", help="the learning rate for feature discriminator", type=float, default=2.5e-5)
    parser.add_argument("-d4lr", help="the learning rate for pointnet discriminator", type=float, default=2.5e-5)
    parser.add_argument("-drop", help="whether to apply dropout in decoder", action='store_true')
    parser.add_argument("-softmax",
                        help="whether to apply softmax as the first layer of the out space discriminator and feature discriminator",
                        action='store_true')
    parser.add_argument("-dr", help="the ratio of the discriminators loss for the unet", type=float, default=.01)

    args = parser.parse_args()
    assert args.model == 'resnet' or args.model == 'effunet' or args.model == 'sd', "model has to be amoong 'resnet', 'sk' and 'sd'"

    appendix = get_appendix()
    print(appendix)
    # import torch.backends.cudnn as cudnn

    torch.autograd.set_detect_anomaly(True)
    # cudnn.benchmark = True
    # cudnn.enabled = True

    main(batch_size=args.bs, n_samples=args.ns, n_epochs=args.e)
