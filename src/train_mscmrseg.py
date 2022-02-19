# PyTorch includes
import torch
from torch.nn import BCELoss, CrossEntropyLoss
import torch.nn.functional as F
# import kornia

from datetime import datetime
import tqdm
import numpy as np
import os

# from data_generator_mscmrseg import ImageProcessor, DataGenerator_PointNet
from dataset.bSSFP_dataset import get_bssfp_dataloader
from dataset.LGE_dataset import get_lge_dataloader
from utils.utils import soft_to_hard_pred, tranfer_data_2_scratch, get_arguments, get_appendix, get_optimizers, \
    get_models, get_model_checkpoints, print_epoch_result
from utils.loss import loss_calc, batch_NN_loss
# from utils.metric import evaluate, dice_coef_multilabel
from utils.timer import timeit
from evaluator import Evaluator

# dic_loss = kornia.losses.DiceLoss()

@timeit
def train_epoch(model_gen, model_dis2, model_dis4, model_dis1=None,
                optim_gen=None, optim_dis2=None, optim_dis4=None, optim_dis1=None,
                trainA_loader=None, trainB_loader=None, trainB_iterator=None):
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
    :param trainA_loader: the source training data generator
    :param trainB_loader:  the target training data generator
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

    running_adv_diff_loss = []
    running_dis_diff_loss = []

    d1_acc1, d1_acc2, d2_acc1, d2_acc2, d4_acc1, d4_acc2 = [], [], [], [], [], []
    print('get bssfp iterator...')
    trainA_iterator = enumerate(trainA_loader)
    # print('get lge iterator...')
    # trainB_iterator = enumerate(trainB_loader)
    print("Get data...")
    if args.mode == 'oneshot':
        try:
            _, (images_t, img_t_aug, tar_name) = next(trainB_iterator)
        except StopIteration:
            trainB_iterator = enumerate(trainB_loader)
            _, (images_t, img_t_aug, tar_name) = next(trainB_iterator)
        idx = tar_name.index('pat_{}_lge_{}.png'.format(args.pat_id, args.slice_id))
        imgB = images_t[idx:idx + 1, ...].cuda()
        # img_t_aug_temp = img_t_aug[idx:idx + 1, ...]
        print("the image selected as target:", tar_name[idx])
        target_name = [tar_name[idx]]

    for _, (imgA, maskA, nameA) in trainA_iterator:
        if args.mode == 'fewshot' or args.mode == 'fulldata':
            try:
                _, (imgB, img_t_aug_temp, target_name) = next(trainB_iterator)
            except StopIteration:
                trainB_iterator = enumerate(trainB_loader)
                _, (imgB, img_t_aug_temp, target_name) = next(trainB_iterator)  # (1, 3, 256, 256)
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
        oS, oS2, vertS = model_gen(imgA.cuda())
        loss_seg = loss_calc(oS, maskA, jaccard=True)
        running_seg_loss.append(loss_seg.item())
        # if args.d4:
        #     loss_seg3 = batch_NN_loss(x=vertS, y=torch.tensor(vertexA).cuda())
        #     vertex_source_loss.append(loss_seg3.item())
        #     loss_seg = loss_seg + args.wp * loss_seg3
        loss_seg.backward()
        # y_pred = soft_to_hard_pred(oS.cpu().detach().numpy(), 1)
        # seg_dice.append(dice_coef_multilabel(y_true=maskA, y_pred=y_pred, channel='channel_first'))
        # 2. train the segmentation model to fool the discriminators
        if args.d1 or args.d2 or args.d4:
            oT, oT2, vertT = model_gen(imgB.cuda())
            loss_adv_diff2 = 0
            if args.d2:
                uncertainty_mapT = -1.0 * torch.sigmoid(oT) * torch.log(torch.sigmoid(oT) + smooth)
                D_out2 = model_dis2(uncertainty_mapT)
                loss_adv_diff2 = args.dr * F.binary_cross_entropy_with_logits(D_out2,
                                                                              torch.FloatTensor(D_out2.data.size()).fill_(
                                                                                  source_domain_label).cuda())
            loss_adv_diff_point = 0
            # if args.d4:
            #     loss_vert_target = batch_NN_loss(x=vertT, y=torch.tensor(vertexB).cuda())
            #     vertex_target_loss.append(loss_vert_target.item())
            #     D_out4 = model_dis4(vertT.transpose(2, 1))[0]
            #     loss_adv_diff_point = args.dr * F.binary_cross_entropy_with_logits(D_out4, torch.FloatTensor(D_out4.data.size()).fill_(
            #         source_domain_label).cuda())
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
        # oS = oS.detach()
        # oT = oT.detach()
        if args.d2:
            uncertainty_mapS = -1.0 * torch.sigmoid(oS.detach()) * torch.log(torch.sigmoid(oS.detach()) + smooth)
            D_out2 = model_dis2(uncertainty_mapS)
            loss_D_same2 = F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(
                source_domain_label).cuda())
            loss_D_same2.backward()
            D_out2 = torch.sigmoid(D_out2.detach()).cpu().numpy()
            D_out2 = np.where(D_out2 >= .5, 1, 0)
            d2_acc1.append(np.mean(D_out2))

        if args.d1:
            D_out1 = model_dis1(oS.detach().contiguous())
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
            D_out2 = model_dis2(uncertainty_mapT.detach().contiguous())
            loss_D_diff2 = F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(
                target_domain_label).cuda())
            loss_D_diff2.backward()
            running_dis_diff_loss.append(loss_D_diff2.item())
            D_out2 = torch.sigmoid(D_out2.detach()).cpu().numpy()
            D_out2 = np.where(D_out2 >= .5, 1, 0)
            d2_acc2.append(1 - np.mean(D_out2))

        if args.d1:
            D_out1 = model_dis1(oT.detach().contiguous())
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
    if args.d2:
        train_result['dis2_acc1'] = np.mean(np.array(d2_acc1))
        train_result['dis2_acc2'] = np.mean(np.array(d2_acc2))
    if args.d1:
        train_result['dis1_acc1'] = np.mean(np.array(d1_acc1))
        train_result['dis1_acc2'] = np.mean(np.array(d1_acc2))
    if args.d4:
        train_result['dis4_acc1'] = np.mean(np.array(d4_acc1))
        train_result['dis4_acc2'] = np.mean(np.array(d4_acc2))
    if args.d4:
        train_result['ver_s_loss'] = np.mean(np.array(vertex_source_loss))
        train_result['ver_t_loss'] = np.mean(np.array(vertex_target_loss))
    return train_result


@timeit
def main():
    scratch = tranfer_data_2_scratch(args, start_time)
    trainloader, s_length = get_bssfp_dataloader(scratch, args, 224)
    # dataloader for LGE images
    targetloader = get_lge_dataloader(scratch, args, 224, aug=True, length=s_length)
    model_gen, model_dis1, model_dis2, model_dis4 = get_models(args)
    optim_gen, optim_dis1, optim_dis2, optim_dis4 = get_optimizers(args, model_gen, model_dis1, model_dis2, model_dis4)
    lr_gen = args.lr
    the_epoch = 0
    best_valid_lge_dice = 0
    best_train_result = {}
    best_valid_result = {}
    tobreak = False
    # create model checkpoint instances
    # create directory for the weights
    modelcheckpoint_unet, modelcheckpoint_dis1, modelcheckpoint_dis2, modelcheckpoint_dis4 = get_model_checkpoints(
        appendix, args)
    evaluator = Evaluator(data_dir=scratch if args.scratch else args.data_dir)
    if args.weight_dir is not None:
        results = evaluator.evaluate_single_dataset(seg_model=model_gen, ifhd=False, ifasd=False, modality='lge',
                                                    phase='valid', bs=args.eval_bs, klc=args.toggle_klc)
        lge_dice = np.round((results['dc'][0] + results['dc'][2] + results['dc'][4]) / 3, 3)
        print('pretrained model valid lge dice: {}'.format(lge_dice))

    print("Training started....!")
    seg_loss, d2_acc1, d2_acc2 = [], [], []
    pcloud_s_loss, pcloud_t_loss = [], []
    d1_acc1, d1_acc2, d4_acc1, d4_acc2 = [], [], [], []
    val_dice = []
    seg_lr, disctor2_lr = [], []
    trainb_iterator = enumerate(targetloader)
    for epoch in tqdm.trange(args.e, desc='Train', ncols=80):
        epoch_start = datetime.now()
        print('Start epoch {}'.format(epoch))
        train_result = train_epoch(model_gen=model_gen, model_dis2=model_dis2,
                                   model_dis4=model_dis4, model_dis1=model_dis1,
                                   optim_gen=optim_gen, optim_dis2=optim_dis2,
                                   optim_dis4=optim_dis4, optim_dis1=optim_dis1,
                                   trainA_loader=trainloader, trainB_loader=targetloader,
                                   trainB_iterator=trainb_iterator)
        seg_loss.append(train_result["seg_loss"])
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
        print('start to evaluate')
        results = evaluator.evaluate_single_dataset(seg_model=model_gen, ifhd=False, ifasd=False, modality='lge',
                                                    phase='valid', bs=args.eval_bs, klc=args.toggle_klc)
        lge_dice = np.round((results['dc'][0] + results['dc'][2] + results['dc'][4]) / 3, 3)
        val_dice.append(lge_dice)

        seg_lr.append(optim_gen.param_groups[0]['lr'])
        if args.d2:
            disctor2_lr.append(optim_dis2.param_groups[0]['lr'])

        print_epoch_result(train_result, lge_dice, epoch + 1, args)

        if best_valid_lge_dice < lge_dice:
            best_valid_lge_dice = lge_dice
            best_train_result = train_result
            best_valid_result = lge_dice
            the_epoch = epoch + 1
        if (datetime.now() - start_time).seconds > max_duration:
            tobreak = True
        modelcheckpoint_unet.step(monitor=lge_dice, model=model_gen, optimizer=optim_gen, epoch=epoch + 1, tobreak=tobreak)
        if args.d1:
            modelcheckpoint_dis1.step(monitor=lge_dice, model=model_dis1, optimizer=optim_dis1, epoch=epoch + 1, tobreak=tobreak)
        if args.d2:
            modelcheckpoint_dis2.step(monitor=lge_dice, model=model_dis2, optimizer=optim_dis2, epoch=epoch + 1, tobreak=tobreak)
        if args.d4:
            modelcheckpoint_dis4.step(monitor=lge_dice, model=model_dis4, optimizer=optim_dis4, epoch=epoch + 1, tobreak=tobreak)
        if tobreak:
            break
        if args.offdecay:
            if (epoch + 1) % args.decay_e == 0:
                lr_gen = lr_gen * 0.2
                for param_group in optim_gen.param_groups:
                    param_group['lr'] = lr_gen
        print('Time elapsed for epoch {}: {}'.format(epoch + 1, datetime.now() - epoch_start))
    print("Best model on epoch {}: valid_dice {}".format(
        the_epoch, best_valid_result))

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(comment=appendix + ".e{}.Scr{}".format(modelcheckpoint_unet.epoch,
                                                                  np.around(best_valid_lge_dice, 3)))
    i = 1
    print("write a training summary")
    for t_loss, v_dice, u_lr in zip(seg_loss, val_dice, seg_lr):
        writer.add_scalar('Loss/Training', t_loss, i)
        writer.add_scalar('Lr/unet', u_lr, i)
        writer.add_scalars('Dice', {
                                    'Validation': v_dice,
                                    }, i)
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
    model_name = '{}.e{}.Scr{}{}'.format(modelcheckpoint_unet.best_model_name_base, modelcheckpoint_unet.epoch,
                                         np.around(modelcheckpoint_unet.best_result, 3), modelcheckpoint_unet.ext)
    print("the weight of the best unet model: {}".format(model_name))
    try:
        model_gen.load_state_dict(torch.load(model_name)['model_state_dict'])
        print("segmentor load from state dict")
    except:
        model_gen.load_state_dict(torch.load(model_name))
    print("model loaded")
    evaluator.evaluate_single_dataset(seg_model=model_gen, modality='lge', phase='test', ifhd=True, ifasd=False,
                                      save=False, weight_dir=None, bs=args.eval_bs, toprint=True,
                                      lge_train_test_split=None)
    return


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    start_time = datetime.now()
    max_duration = 24 * 3600 - 10 * 60  # 85800 seconds. set the maximum running time to prevent from exceeding time limitation
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("cuda available: {}".format(torch.cuda.is_available()))
    args = get_arguments()
    print('batch_size: {}'.format(args.bs))
    appendix = get_appendix(args)
    print(appendix)
    import torch.backends.cudnn as cudnn
    torch.autograd.set_detect_anomaly(True)
    cudnn.benchmark = True
    cudnn.enabled = True
    main()
    # print('Program finished. Time elapsed: {}'.format(datetime.now() - start_time))
