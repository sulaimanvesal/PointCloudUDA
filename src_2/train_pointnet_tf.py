from datetime import datetime
import tqdm
import numpy as np
import os
import math
import kornia
dic_loss = kornia.losses.DiceLoss()
# PyTorch includes
import torch
# from torch.nn import BCELoss, CrossEntropyLoss
import torch.nn.functional as F

# network includes
# from networks.deeplabv3 import *
from networks.unet import Segmentation_model_Point
# from networks.SDUnet import UNet as SDUnet
from networks.GAN import UncertaintyDiscriminator
from networks.PointNetCls import PointNetCls
# custom includes
from utils.callbacks import ModelCheckPointCallback
from data_generator_CIF_np import ImageProcessor, DataGenerator_PointNet
from utils_ import soft_to_hard_pred, jaccard_loss, batch_NN_loss
from metric import metrics2, dice_coef_multilabel
from timer import timeit


def get_generators(ids_train, ids_valid, ids_train_lge, ids_valid_lge, batch_size=16, n_samples=2000, crop_size=0, mh=False):
    trainA_generator = DataGenerator_PointNet(df=ids_train, channel="channel_first", apply_noise=args.gn, gn_prob=args.gn_prob,
                                              phase="train", apply_online_aug=args.offaug,
                                     batch_size=batch_size, source="source", crop_size=crop_size,
                                     n_samples=n_samples, match_hist=mh, ifvert=args.d4, heavy_aug=args.heavy_aug)
    validA_generator = DataGenerator_PointNet(df=ids_valid, channel="channel_first", apply_noise=False, phase="valid",
                                     apply_online_aug=False,
                                     batch_size=batch_size, source="source", crop_size=crop_size,
                                     n_samples=-1, match_hist=mh, ifvert=args.d4)
    trainB_generator = DataGenerator_PointNet(df=ids_train_lge, channel="channel_first", apply_noise=args.gn, gn_prob=args.gn_prob,
                                              phase="train",apply_online_aug=args.offaug,
                                     batch_size=batch_size, source="target", crop_size=crop_size,
                                     n_samples=n_samples, ifvert=args.d4, heavy_aug=args.heavy_aug)
    validB_generator = DataGenerator_PointNet(df=ids_valid_lge, channel="channel_first", apply_noise=False, phase="valid",
                                     apply_online_aug=False,
                                     batch_size=batch_size, source="target", crop_size=crop_size,
                                     n_samples=-1, ifvert=args.d4)
    testB_generator = DataGenerator_PointNet(df=ids_train_lge, channel="channel_first", apply_noise=False, phase="train",
                                    apply_online_aug=False,
                                    batch_size=batch_size, source="target", crop_size=crop_size,
                                    n_samples=-1, ifvert=args.d4)
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
            if args.softmax:
                pred = F.softmax(prediction, dim=1)
                l1 = F.cross_entropy(pred, torch.tensor(np.argmax(y_batch, axis=1), dtype=torch.long).cuda())
            else:
                pred = F.sigmoid(prediction)
                l1 = F.binary_cross_entropy(pred, torch.tensor(y_batch, dtype=torch.float32).cuda())
            if args.dice:
                l2 = dic_loss(prediction, torch.tensor(np.argmax(y_batch, axis=1), dtype=torch.int64).cuda())
            else:
                l2 = jaccard_loss(logits=pred, true=torch.tensor(y_batch, dtype=torch.float32).cuda(), activation=False)
            l3 = 0
            if args.d4:
                l3 = batch_NN_loss(x=vertS, y=torch.tensor(z_batch).cuda())
                vert_loss_list.append(l3.item())
            else:
                vert_loss_list.append(-1)

            l = l1 + l2 + l3
            loss_list.append(l.item())

            y_pred = prediction.cpu().detach().numpy()
            # y_pred = soft_to_hard_pred(y_pred, 1)
            y_pred = np.argmax(y_pred, axis=1)
            y_batch = np.argmax(y_batch, axis=1)
            # y_pred = keep_largest_connected_components(mask=y_pred)
            result = metrics2(img_pred=y_pred, img_gt=y_batch, apply_hd=hd, apply_asd=False)
            dice_list.append((result["lv"][0] + result["myo"][0] + result["la"][0] + result['aa'][0]) / 4.)
            if hd:
                hd_list.append((result["lv"][1] + result["myo"][1] + result["la"][1] + result['aa'][0]) / 4.)
            # dice_list.append(dice_coef_multilabel(y_true=y_batch, y_pred=y_pred, channel='channel_first'))
            # if hd:
            #     y_pred = np.moveaxis(y_pred, 1, -1)
            #     y_pred = np.argmax(y_pred, axis=-1)
            #     y_pred = keep_largest_connected_components(mask=y_pred)
            #     hd_list.append(hausdorff_multilabel(y_true=y_batch, y_pred=y_pred))
    output = {}
    output["dice"] = np.mean(np.array(dice_list))
    output["loss"] = np.mean(np.array(loss_list))
    output["vert_loss"] = np.mean(np.array(vert_loss_list))

    if hd:
        output["hd"] = np.mean(np.array(hd_list))
    return output

@timeit
def valid_model(seg_model, validA_iterator, validB_iterator, testB_generator):
    valid_result = {}
    seg_model.eval()

    print("start to valid")

    output = valid_model_with_one_dataset(seg_model=seg_model, data_generator=validA_iterator, hd=False)
    val_dice = output["dice"]
    val_loss = output['loss']
    val_vert_loss = output['vert_loss']


    output = valid_model_with_one_dataset(seg_model=seg_model, data_generator=validB_iterator, hd=False)
    val_lge_dice = output['dice']
    val_lge_loss = output['loss']
    val_lge_vert_loss = output['vert_loss']

    output = valid_model_with_one_dataset(seg_model=seg_model, data_generator=testB_generator, hd=False)
    test_lge_dice = output['dice']
    test_lge_loss = output['loss']

    # test_lge_hd.append(output["hd"])
    valid_result["val_dice"] = val_dice
    valid_result['val_loss'] = val_loss
    valid_result['val_vert_loss'] = val_vert_loss
    valid_result['val_lge_dice'] = val_lge_dice
    valid_result['val_lge_loss'] = val_lge_loss
    valid_result['val_lge_vert_loss'] = val_lge_vert_loss
    valid_result['test_lge_dice'] = test_lge_dice
    valid_result['test_lge_loss'] = test_lge_loss

    return valid_result


@timeit
def train_epoch(model_gen, model_dis2, model_dis3, model_dis4, model_dis1=None,
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
    if args.d2:
        model_dis2.train()
    if args.d4:
        model_dis4.train()

    train_result = {}

    running_seg_loss = []
    vertex_source_loss = []
    vertex_target_loss = []
    entropy_loss = []
    entropy_loss_T = []
    seg_dice = []

    running_dis_diff_loss = []

    d1_acc1,d1_acc2,d2_acc1,d2_acc2,d3_acc1,d3_acc2,d4_acc1,d4_acc2 = [],[],[],[],[],[],[],[]

    # take data from generators
    for (imgA, maskA, vertexA), (imgB, _, vertexB) in zip(trainA_iterator, trainB_iterator):

        # set the gradients of all the models to 0
        optim_gen.zero_grad()
        if args.d1:
            optim_dis1.zero_grad()
        if args.d2:
            optim_dis2.zero_grad()
        if args.d3:
            optim_dis3.zero_grad()
        if args.d4:
            optim_dis4.zero_grad()

        # 1. train the generator (do not update the params in the discriminators)
        if args.d1:
            for param in model_dis1.parameters():
                param.requires_grad = False
        if args.d2:
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
        if args.softmax:
            predS = F.softmax(oS, dim=1)
            loss_seg = F.cross_entropy(predS, torch.tensor(np.argmax(maskA, axis=1), dtype=torch.long).cuda())
        else:
            predS = F.sigmoid(oS)
            loss_seg = F.binary_cross_entropy(predS, torch.tensor(maskA, dtype=torch.float32).cuda())
        if args.dice:
            loss_seg2 = dic_loss(oS, torch.tensor(np.argmax(maskA, axis=1), dtype=torch.int64).cuda())
        else:
            loss_seg2 = jaccard_loss(logits=predS, true=torch.tensor(maskA, dtype=torch.float32).cuda(), activation=False)
        loss_seg3 = 0
        if args.d4:
            loss_seg3 = batch_NN_loss(x=vertS, y=torch.tensor(vertexA).cuda())
            vertex_source_loss.append(loss_seg3.item())
        c = predS.size()[1]
        uncertainty_mapS = -1.0 * predS * torch.log(predS + smooth) / math.log(c)
        temp_loss = torch.mean(torch.sum(uncertainty_mapS, dim=1))
        entropy_loss.append(temp_loss.item())
        loss_entropy = 0
        if args.d2 and args.etpls:
            loss_entropy = temp_loss
        loss_seg1 = loss_seg + loss_seg2 + args.wp * loss_seg3 + loss_entropy

        #loss_seg2 = dic_loss(torch.sigmoid(oS), torch.tensor(maskA, dtype=torch.float32).cuda())
        #oss_seg1 = loss_seg + loss_seg2
        running_seg_loss.append((loss_seg + loss_seg2).item())
        # backpropogate the loss of the generator
        loss_seg1.backward()

        y_pred = soft_to_hard_pred(oS.cpu().detach().numpy(), 1)
        seg_dice.append(dice_coef_multilabel(y_true=maskA, y_pred=y_pred, channel='channel_first'))
        # optim_gen.step()

        oT, oT2, vertT = model_gen(torch.tensor(imgB).cuda())
        predT = F.softmax(oT, dim=1) if args.softmax else F.sigmoid(oT)
        c = predT.size()[1]
        uncertainty_mapT = -1.0 * predT * torch.log(predT + smooth) / math.log(c)
        temp_loss = torch.mean(torch.sum(uncertainty_mapT, dim=1))
        entropy_loss_T.append(temp_loss.item())
        loss_adv_diff = 0
        if args.Tetpls:
            loss_adv_diff += temp_loss
        if args.d1 or args.d2 or args.d3 or args.d4:
            loss_adv_entropy = 0
            if args.d2:
                D_out2 = model_dis2(uncertainty_mapT)
                loss_adv_entropy = args.dr * F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(
                    source_domain_label).cuda())

            loss_adv_point = 0
            if args.d4:
                loss_vert_target = batch_NN_loss(x=vertT, y=torch.tensor(vertexB).cuda())
                vertex_target_loss.append(loss_vert_target.item())
                D_out4 = model_dis4(vertT.transpose(2,1))[0]
                loss_adv_point = args.dr * F.binary_cross_entropy_with_logits(D_out4, torch.FloatTensor(D_out4.data.size()).fill_(
                    source_domain_label).cuda())

            loss_adv_output = 0
            if args.d1:
                D_out1 = model_dis1(predT)
                loss_adv_output = args.dr * F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_domain_label).cuda())

            loss_adv_feature = 0
            if args.d3:
                D_out3 = model_dis3(oT2)
                loss_adv_feature = args.dr * F.binary_cross_entropy_with_logits(D_out3, torch.FloatTensor(D_out3.data.size()).fill_(source_domain_label).cuda())

            loss_adv_diff += args.w2 * loss_adv_entropy + args.w4 * loss_adv_point + args.w1 * loss_adv_output + loss_adv_feature
        if loss_adv_diff != 0:
            try:
                loss_adv_diff.backward()
            except:
                print("error!!!!")
                print("value of the loss: {}".format(loss_adv_diff.item()))
                print("exit")
                exit(1)
        optim_gen.step()


        if args.d1 or args.d2 or args.d3 or args.d4:
            # 3. train discriminator with images from source domain
            if args.d1:
                for param in model_dis1.parameters():
                    param.requires_grad = True
            if args.d2:
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

            if args.d2:
                D_out2 = model_dis2(uncertainty_mapS.detach())
                loss_D_same2 = F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(
                    source_domain_label).cuda())
                loss_D_same2.backward()
                D_out2 = torch.sigmoid(D_out2.detach()).cpu().numpy()
                D_out2 = np.where(D_out2 >= .5, 1, 0)
                d2_acc1.append(np.mean(D_out2))

            if args.d1:
                D_out1 = model_dis1(predS.detach())
                loss_D_same1 = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(
                    source_domain_label).cuda())
                loss_D_same1.backward()
                D_out1 = torch.sigmoid(D_out1.detach()).cpu().numpy()
                D_out1 = np.where(D_out1 >= .5, 1, 0)
                d1_acc1.append(np.mean(D_out1))

            if args.d3:
                D_out3 = model_dis3(oS2.detach())
                loss_D_same3 = F.binary_cross_entropy_with_logits(D_out3, torch.FloatTensor(D_out3.data.size()).fill_(
                    source_domain_label).cuda())
                loss_D_same3.backward()
                D_out3 = torch.sigmoid(D_out3.detach()).cpu().numpy()
                D_out3 = np.where(D_out3 >= .5, 1, 0)
                d3_acc1.append(np.mean(D_out3))

            if args.d4:
                vertS = vertS.detach()
                D_out4 = model_dis4(vertS.transpose(2,1))[0]
                loss_D_same4 = F.binary_cross_entropy_with_logits(D_out4, torch.FloatTensor(D_out4.data.size()).fill_(
                    source_domain_label).cuda())
                loss_D_same4.backward()
                D_out4 = torch.sigmoid(D_out4.detach()).cpu().numpy()
                D_out4 = np.where(D_out4 >= .5, 1, 0)
                d4_acc1.append(np.mean(D_out4))

            # 4. train discriminator with images from target domain
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
                D_out1 = model_dis1(predT.detach())
                loss_D_diff1 = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(
                    target_domain_label).cuda())
                loss_D_diff1.backward()
                D_out1 = torch.sigmoid(D_out1.detach()).cpu().numpy()
                D_out1 = np.where(D_out1 >= .5, 1, 0)
                d1_acc2.append(1 - np.mean(D_out1))

            if args.d3:
                D_out3 = model_dis3(oT2.detach())
                loss_D_diff3 = F.binary_cross_entropy_with_logits(D_out3, torch.FloatTensor(D_out3.data.size()).fill_(
                    target_domain_label).cuda())
                loss_D_diff3.backward()
                D_out3 = torch.sigmoid(D_out3.detach()).cpu().numpy()
                D_out3 = np.where(D_out3 >= .5, 1, 0)
                d3_acc2.append(1 - np.mean(D_out3))

            if args.d4:
                vertT = vertT.detach()
                D_out4 = model_dis4(vertT.transpose(2,1))[0]
                loss_D_diff_4 = F.binary_cross_entropy_with_logits(D_out4, torch.FloatTensor(D_out4.data.size()).fill_(
                    target_domain_label).cuda())
                #loss_D_diff = loss_D_diff_1 + loss_D_diff_2
                loss_D_diff_4.backward()
                D_out4 = torch.sigmoid(D_out4.detach()).cpu().numpy()
                D_out4 = np.where(D_out4 >= .5, 1, 0)
                d4_acc2.append(1 - np.mean(D_out4))


            # 5. update parameters
            if args.d1:
                optim_dis1.step()
            if args.d2:
                optim_dis2.step()
            if args.d3:
                optim_dis3.step()
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
    if args.d3:
        train_result['dis3_acc1'] = np.mean(np.array(d3_acc1))
        train_result['dis3_acc2'] = np.mean(np.array(d3_acc2))
    if args.d4:
        train_result['dis4_acc1'] = np.mean(np.array(d4_acc1))
        train_result['dis4_acc2'] = np.mean(np.array(d4_acc2))
    train_result['ver_s_loss'] = np.mean(np.array(vertex_source_loss))
    train_result['ver_t_loss'] = np.mean(np.array(vertex_target_loss))
    train_result['entropy_loss'] = np.mean(np.array(entropy_loss))
    train_result['entropy_loss_T'] = np.mean(np.array(entropy_loss_T))
    return train_result

def print_epoch_result(train_result, valid_result, epoch, max_epochs):
    epoch_len = len(str(max_epochs))
    seg_loss, seg_dice = train_result["seg_loss"], train_result['seg_dice']
    val_dice, val_loss, val_lge_dice, val_lge_loss, test_lge_dice, test_lge_loss, valid_vert_loss = valid_result["val_dice"], valid_result['val_loss'], \
                                                                                   valid_result['val_lge_dice'], valid_result['val_lge_loss'],\
                                                                                   valid_result['test_lge_dice'], valid_result['test_lge_loss'], valid_result['val_vert_loss']

    print_msg_line1 = f'valid_loss: {val_loss:.5f} ' + f'valid_lge_loss: {val_lge_loss:.5f} ' + f'test_lge_loss: {test_lge_loss:.5f} '
    if args.d4:
        ver_s_loss, ver_t_loss = train_result['ver_s_loss'], train_result['ver_t_loss']
        print_msg_line1 += f'vertex_loss: {ver_s_loss:.5f}, vertex_t_loss: {ver_t_loss:.5f} '
    print_msg_line2 = f'valid_dice: {val_dice:.5f} ' + \
                      f'valid_lge_dice: {val_lge_dice:.5f} ' + \
                      f'test_lge_dice: {test_lge_dice:.5f} ' + f'valid_vert_loss: {valid_vert_loss:.5f} '

    print_msg_line1 = f'train_loss: {seg_loss:.5f} ' + print_msg_line1
    print_msg_line2 = f'train_dice: {seg_dice:.5f} ' + print_msg_line2
    if args.d2:
        dis2_acc1, dis2_acc2 = train_result["dis2_acc1"], train_result['dis2_acc2']
        print_msg_line2 += f'disctor2_train_acc1: {dis2_acc1: 5f} ' + f'disctor2_train_acc2: {dis2_acc2: 5f} '

    print_msg_line1 = f'[{epoch + 1:>{epoch_len}}/{max_epochs:>{epoch_len}}] ' + print_msg_line1
    print_msg_line2 = ' ' * (2 * epoch_len + 4) + print_msg_line2
    print(print_msg_line1)
    print(print_msg_line2)


@timeit
def main(batch_size=24, n_samples=2000, n_epochs=200):
    max_duration = 24 * 3600 - 60 * 60  # 85800 seconds
    mr_train = ImageProcessor.split_data("./../input/mr_train_list.csv")
    mr_valid = ImageProcessor.split_data("./../input/mr_val_list.csv")
    ct_train = ImageProcessor.split_data('./../input/ct_train_list.csv')
    ct_valid = ImageProcessor.split_data('./../input/ct_val_list.csv')
    print("Trainining on {} trainA, {} trainB, validating on {} testA and {} testB samples...!!".format(len(mr_train),
                                                                                                        len(
                                                                                                            ct_train),
                                                                                                        len(mr_valid),
                                                                                                        len(
                                                                                                            ct_valid)))

    trainA_iterator, \
    validA_iterator, \
    trainB_iterator, validB_iterator, testB_generator = get_generators(mr_train,
                                                                       mr_valid,
                                                                       ct_train,
                                                                       ct_valid,
                                                                       batch_size=batch_size,
                                                                       n_samples=n_samples,
                                                                       crop_size=0,
                                                                       mh=args.offmh)

    # 2. model
   # model_gen = DeepLab(num_classes=4, backbone='resnet', output_stride=16,
   #                     sync_bn=False, freeze_bn=False).cuda()

    # 3. model
    if args.model == 'sk':
        from networks.skunet import SKUNet
        model_gen = SKUNet(in_channel=1, out_channel=args.out_ch, feature_dis=args.d3, pointnet=args.d4, n_class=5).cuda()
    # elif args.model == 'sd':
    #     model_gen = SDUnet(n_channels=1, n_classes=5, feature_dis=args.d3, pointnet=args.d4)
    else:
        model_gen = Segmentation_model_Point(filters=args.nf, in_channels=3, feature_dis=args.d3, pointnet=args.d4, drop=args.drop,
                                             n_class=5, fc_inch=121, heinit=args.he, multicuda=args.multicuda)

    if args.multicuda:
        model_gen.tomulticuda()
    else:
        model_gen.cuda()

    # model_dis = BoundaryDiscriminator().cuda()
    model_dis1 = None
    if args.d1:
        model_dis1 = UncertaintyDiscriminator(in_channel=5, heinit=args.he).cuda()
    model_dis2 = None
    if args.d2:
        model_dis2 = UncertaintyDiscriminator(in_channel=5, heinit=args.he).cuda()
    model_dis3 = None
    if args.d3:
        from networks.GAN import OutputDiscriminator
        model_dis3 = OutputDiscriminator(in_channel=5, softmax=args.softmax).cuda()
    model_dis4 = None
    if args.d4:
        model_dis4 = PointNetCls().cuda()
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
    optim_dis3 = None
    if args.d3:
        optim_dis3 = torch.optim.SGD(
            model_dis3.parameters(),
            lr=args.d3lr,
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
    best_valid_lge_dice = -1
    best_train_result = {}
    best_valid_result = {}
    # create directory for the weights
    root_directory = '../weights/' + 'DeepLabv3'+ '/'
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
    #model_gen.load_state_dict(torch.load(root_directory + 'unet_multilevel_model_checkpoint_stage2.pt'))
    if args.load_weight:
        checkpoint = torch.load(weight_dir)
        try:
            model_gen.load_state_dict(checkpoint['model_state_dict'])
            optim_gen.load_state_dict(checkpoint['optimizer_state_dict'])
            if args.lr_fix != args.lr:
                for param_group in optim_gen.param_groups:
                    param_group['lr'] = args.lr
            print('model loaded from dict')
        except:
            model_gen.load_state_dict(checkpoint)
            print('model loaded from single state')
        if args.d1:
            checkpoint = torch.load(d1_weight_dir)
            try:
                model_dis1.load_state_dict(checkpoint['model_state_dict'])
                try:
                    optim_dis1.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    pass
            except:
                model_dis1.load_state_dict(checkpoint)
        if args.d2:
            checkpoint = torch.load(d2_weight_dir)
            try:
                model_dis2.load_state_dict(checkpoint['model_state_dict'])
                try:
                    optim_dis2.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    pass
            except:
                model_dis2.load_state_dict(checkpoint)
        if args.d3:
            checkpoint = torch.load(d3_weight_dir)
            try:
                model_dis3.load_state_dict(checkpoint['model_state_dict'])
                try:
                    optim_dis3.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    pass
            except:
                model_dis3.load_state_dict(checkpoint)
        if args.d4:
            checkpoint = torch.load(d4_weight_dir)
            try:
                model_dis4.load_state_dict(checkpoint['model_state_dict'])
                try:
                    optim_dis4.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    pass
            except:
                model_dis4.load_state_dict(checkpoint)
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
    d1_acc1, d1_acc2, d3_acc1, d3_acc2, d4_acc1, d4_acc2 = [], [], [], [], [], []
    val_dice, val_loss, val_lge_dice, val_lge_loss, test_lge_dice, test_lge_loss = [], [], [], [], [], []
    val_vert_loss, val_lge_vert_loss = [], []
    seg_lr, disctor2_lr = [], []
    etp_loss, etp_loss_T = [], []
    max_time_elapse_epoch = 0
    for epoch in tqdm.trange(n_epochs, desc='Train', ncols=80):
        iter_start_time = datetime.now()
        train_result = train_epoch(model_gen=model_gen, model_dis2=model_dis2, model_dis3=model_dis3, model_dis4=model_dis4, model_dis1=model_dis1,
                                   optim_gen=optim_gen, optim_dis2=optim_dis2, optim_dis3=optim_dis3, optim_dis4=optim_dis4, optim_dis1=optim_dis1,
                                   trainA_iterator=trainA_iterator, trainB_iterator=trainB_iterator)

        seg_loss.append(train_result["seg_loss"])
        seg_dice.append(train_result['seg_dice'])
        etp_loss.append(train_result['entropy_loss'])
        etp_loss_T.append(train_result['entropy_loss_T'])
        if args.d2:
            d2_acc1.append(train_result['dis2_acc1'])
            d2_acc2.append(train_result['dis2_acc2'])
        if args.d1:
            d1_acc1.append(train_result['dis1_acc1'])
            d1_acc2.append(train_result['dis1_acc2'])
        if args.d3:
            d3_acc1.append(train_result['dis3_acc1'])
            d3_acc2.append(train_result['dis3_acc2'])
        if args.d4:
            pcloud_s_loss.append(train_result['ver_s_loss'])
            pcloud_t_loss.append(train_result['ver_t_loss'])
            d4_acc1.append(train_result['dis4_acc1'])
            d4_acc2.append(train_result['dis4_acc2'])

        valid_result = valid_model(seg_model=model_gen, validA_iterator=validA_iterator,
                                   validB_iterator=validB_iterator, testB_generator=testB_generator)
        val_dice.append(valid_result["val_dice"])
        val_loss.append(valid_result['val_loss'])
        val_vert_loss.append(valid_result['val_vert_loss'])
        val_lge_dice.append(valid_result['val_lge_dice'])
        val_lge_loss.append(valid_result['val_lge_loss'])
        val_lge_vert_loss.append(valid_result['val_lge_vert_loss'])
        test_lge_dice.append(valid_result['test_lge_dice'])
        test_lge_loss.append(valid_result['test_lge_loss'])

        seg_lr.append(optim_gen.param_groups[0]['lr'])
        if args.d2:
            disctor2_lr.append(optim_dis2.param_groups[0]['lr'])

        print_epoch_result(train_result, valid_result, epoch, n_epochs)

        if best_valid_lge_dice < valid_result["val_lge_dice"]:
            best_valid_lge_dice = valid_result["val_lge_dice"]
            best_train_result = train_result
            best_valid_result = valid_result
            the_epoch = epoch + 1
        if (datetime.now() - start_time).seconds > max_duration:
            epoch = n_epochs - 1
            ifbreak = True
        else:
            ifbreak = False
        monitor_score = valid_result["val_lge_dice"]
        modelcheckpoint_unet.step(monitor=monitor_score, model=model_gen, epoch=epoch + 1, optimizer=optim_gen)
        if args.d1:
            modelcheckpoint_dis1.step(monitor=monitor_score, model=model_dis1, epoch=epoch + 1, optimizer=optim_dis1)
        if args.d2:
            modelcheckpoint_dis2.step(monitor=monitor_score, model=model_dis2, epoch=epoch + 1, optimizer=optim_dis2)
        if args.d3:
            modelcheckpoint_dis3.step(monitor=monitor_score, model=model_dis3, epoch=epoch + 1, optimizer=optim_dis3)
        if args.d4:
            modelcheckpoint_dis4.step(monitor=monitor_score, model=model_dis4, epoch=epoch + 1, optimizer=optim_dis4)
        if ifbreak:
            break
        if args.offdecay:
            if (epoch + 1) % 100 == 0:
                lr_gen = lr_gen * 0.2
                for param_group in optim_gen.param_groups:
                    param_group['lr'] = lr_gen
        print("time elapsed: {} hours".format(np.around((datetime.now() - start_time).seconds / 3600., 1)))
        max_time_elapse_epoch = max(np.around((datetime.now() - iter_start_time).seconds), max_time_elapse_epoch)
        max_duration = 24 * 3600 - max_time_elapse_epoch - 15 * 60
    print("Best model on epoch {}: train_dice {}, valid_dice {}, lge_dice {}, test_lge_dice {}".format(
        the_epoch, best_train_result['seg_dice'], best_valid_result['val_dice'], best_valid_result['val_lge_dice'],
        best_valid_result['test_lge_dice']))

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(comment=appendix + ".Scr{}".format(np.around(best_valid_lge_dice, 3)))
    i = 1
    print("write a training summary")
    for t_loss, t_dice, v_loss, v_dice, lge_loss, lge_dice, t_lge_loss, t_lge_dice, u_lr, eloss, t_eloss in zip(seg_loss,
                                                                                                              seg_dice,
                                                                                                              val_loss,
                                                                                                              val_dice,
                                                                                                              val_lge_loss,
                                                                                                              val_lge_dice,
                                                                                                              test_lge_loss,
                                                                                                              test_lge_dice,
                                                                                                              seg_lr, etp_loss, etp_loss_T):
        writer.add_scalars('Loss', {'Training': t_loss,
                                    'Validation': v_loss,
                                    'Valid_LGE': lge_loss,
                                    'Test_LGE': t_lge_loss}, i)
        writer.add_scalars('Dice', {'Training': t_dice,
                                    'Validation': v_dice,
                                    'Valid_LGE': lge_dice,
                                    'Test_LGE': t_lge_dice}, i)
        writer.add_scalar('Lr/unet', u_lr, i)
        writer.add_scalars('Entropy', {'S_Entropy': eloss,
                                       'T_Entropy': t_eloss}, i)
        i += 1
    if args.d2:
        i = 1
        for a1, a2 in zip(d2_acc1, d2_acc2):
            writer.add_scalar('d2_acc1', a1, i)
            writer.add_scalar('d2_acc2', a2, i)
            i += 1
    if args.d1:
        i = 1
        for a1, a2 in zip(d1_acc1, d1_acc2):
            writer.add_scalar('d1_acc1', a1, i)
            writer.add_scalar('d1_acc2', a2, i)
            i += 1
    if args.d3:
        i = 1
        for a1, a2 in zip(d3_acc1, d3_acc2):
            writer.add_scalar('d3_acc1', a1, i)
            writer.add_scalar('d3_acc2', a2, i)
            i += 1
    if args.d4:
        i = 1
        for p_s_loss, p_t_loss, v_vert_l, v_lge_vert_l, a1, a2 in zip(pcloud_s_loss, pcloud_t_loss, val_vert_loss, val_lge_vert_loss, d4_acc1, d4_acc2):
            writer.add_scalars('Loss_Point', {'PointCloudS': p_s_loss,
                                        'PointCloudT': p_t_loss,
                                        'vPointCloudS': v_vert_l,
                                        'vPointCloudT': v_lge_vert_l}, i)
            writer.add_scalar('d4_acc1', a1, i)
            writer.add_scalar('d4_acc2', a2, i)
            i += 1
    writer.close()
    # evaluate the model
    print('Evaluate the model')
    best_model_name_base, ext = os.path.splitext(best_weight_dir)
    root_directory = '{}{}{}{}'.format(best_model_name_base, '.Scr', np.around(modelcheckpoint_unet.best_result, 3), ext)
    print("root directory: {}".format(root_directory))
    from predict_output_tf import evaluate_segmentation
    evaluate_segmentation(root_directory=root_directory, unet_model=model_gen, save=False, model_name='', ifhd=True, ifasd=True)

def get_appendix():
    appendix = args.apdx + '.' + args.model + '.lr{}'.format(args.lr_fix)
    if args.nf != 32:
        appendix += '.nf{}'.format(args.nf)
    if args.d1:
        appendix += '.d1lr{}'.format(args.d1lr)
    if args.d2:
        appendix += '.d2lr{}'.format(args.d2lr)
    if args.d3:
        appendix += '.d3lr{}'.format(args.d3lr)
    if args.d4:
        appendix += '.d4lr{}'.format(args.d4lr)
    if args.w1 != 1:
        appendix += '.w1_{}'.format(args.w1)
    if args.w2 != 1:
        appendix += '.w2_{}'.format(args.w2)
    if args.w4 != 1:
        appendix += '.w4_{}'.format(args.w4)
    if args.raug:
        appendix += '.raug'
    if not args.offaug:
        appendix += '.offaug'
    if args.gn:
        appendix += '.gn'
        if args.gn_prob != .5:
            appendix += str(args.gn_prob)
    if not args.offmh:
        appendix += '.offmh'
    if args.heavy_aug:
        appendix += '.hvyaug'
    if args.drop:
        appendix += '.drop'
    if args.softmax:
        appendix += '.softmax'
    if not args.offdecay:
        appendix += '.offdecay'
    if args.wp != 1.:
        appendix += '.wp{}'.format(args.wp)
    if args.etpls:
        appendix += '.etpls'
    if args.Tetpls:
        appendix += '.Tetpls'
    if args.dice:
        appendix += '.dice'
    if args.he:
        appendix += '.he'
    return appendix


if __name__ == '__main__':
    start_time = datetime.now()
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-raug", help='whether to use raugmented data', action='store_true')
    parser.add_argument("-offaug", help='whether not to augment the data', action='store_false')
    parser.add_argument("-gn", help='whether to apply gaussian noise', action='store_true')
    parser.add_argument("-gn_prob", help='The probability to apply gaussian noise', type=float, default=.5)
    parser.add_argument("-heavy_aug", help='whether to apply heavy augmentation', action='store_true')
    parser.add_argument("-offmh", help='whether not to apply histogram matching', action='store_false')
    parser.add_argument("-load_weight", help='whether to load weight', action='store_true')
    parser.add_argument("-bs", help="the batch size of training", type=int, default=16)
    parser.add_argument("-ns", help="number of samples per epoch", type=int, default=2000)
    parser.add_argument("-e", help="number of epochs", type=int, default=200)
    parser.add_argument("-lr", help="learning rate of unet", type=float, default=1e-3)
    parser.add_argument("-lr_fix", help="learning rate of unet", type=float, default=1e-3)
    parser.add_argument("-offdecay", help="whether not to use learning rate decay for unet", action='store_false')
    parser.add_argument("-apdx", help="the appendix to the checkpoint", type=str, default='train_point_tf')
    parser.add_argument("-model", help="the unet model chosen to use", type=str, default='resnet') # sk, sd
    parser.add_argument("-nf", help="number of filters", type=int, default=32)
    parser.add_argument("-out_ch", help="the out channels of the first conv layer(only used for SKUnet)", type=int,
                        default=32)
    parser.add_argument("-d1", help="whether to apply outer space discriminator", action='store_true')
    parser.add_argument("-d2", help="whether to apply entropy discriminator", action='store_true')
    parser.add_argument("-d3", help="whether to apply feature discriminator", action='store_true')
    parser.add_argument("-d4", help='whether to use pointnet', action='store_true')
    parser.add_argument("-d1lr", help="the learning rate for outer space discriminator", type=float, default=2.5e-5)
    parser.add_argument("-d2lr", help="the learning rate for entropy discriminator", type=float, default=2.5e-5)
    parser.add_argument("-d3lr", help="the learning rate for feature discriminator", type=float, default=2.5e-5)
    parser.add_argument("-d4lr", help="the learning rate for pointnet discriminator", type=float, default=2.5e-5)
    parser.add_argument("-drop", help="whether to apply dropout in decoder", action='store_true')
    parser.add_argument("-softmax",
                        help="whether to apply softmax as the last activation layer of segmentation model",
                        action='store_true')
    parser.add_argument("-dr", help="the ratio of the discriminators loss for the unet", type=float, default=.01)
    parser.add_argument("-wp", help="the weight for the loss of the point net", type=float, default=1.)
    parser.add_argument("-w1", help="the weight for the loss of the output space(unet)", type=float, default=1.)
    parser.add_argument("-w2", help="the weight for the loss of the entropy space(unet)", type=float, default=1.)
    parser.add_argument("-w4", help="the weight for the loss of the point net(unet)", type=float, default=1.)
    parser.add_argument("-etpls", help="whether to apply entropy loss", action='store_true')
    parser.add_argument("-Tetpls", help="whether to apply entropy loss", action='store_true')
    parser.add_argument("-dice", help="whether to use dice loss", action='store_true')
    parser.add_argument("-he", help="whether to use He initializer", action='store_true')
    parser.add_argument("-multicuda", help="whether to use two cuda gpus", action='store_true')

    args = parser.parse_args()
    assert args.model == 'resnet' or args.model == 'sk' or args.model == 'sd', "model has to be amoong 'resnet', 'sk' and 'sd'"
    # assert not (args.softmax and args.sigmoid), "Only one of 'softmax' or 'sigmoid' can be used for activation function."
    assert args.gn_prob > 0 and args.gn_prob <= 1, 'gn_prob must be > 0 and <= 1'

    appendix = get_appendix()
    print(appendix)
    # import torch.backends.cudnn as cudnn

    torch.autograd.set_detect_anomaly(True)
    # cudnn.benchmark = True
    # cudnn.enabled = True

    main(batch_size=args.bs, n_samples=args.ns, n_epochs=args.e)