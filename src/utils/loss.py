import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def jaccard_loss(true, logits, eps=1e-7):
    """
    Computes the Jaccard loss, a.k.a the IoU loss.
    Args:
        true: the ground truth of shape [B, H, W] or [B, 1, H, W]
        logits: the output of the segmentation model (without softmax) [B, C, H, W]
        eps:

    Returns:
    The Jaccard loss
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.long().cuda().squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()  # B, C, H, W
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return 1 - jacc_loss


def loss_calc(pred, label, gpu=0, jaccard=False):
    """
    This function returns cross entropy loss plus jaccard loss for semantic segmentation
    Args:
        pred: the logits of the prediction with shape [B, C, H, W]
        label: the ground truth with shape [B, H, W]
        gpu: the gpu number
        jaccard: if apply jaccard loss

    Returns:

    """
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    loss = criterion(pred, label.long().cuda())
    if jaccard:
        loss = loss + jaccard_loss(true=label.long().cuda(), logits=pred)
    return loss


def batch_NN_loss(x, y):
    """
    calculate the distance loss between two point sets
    :param x: a point sets
    :param y: another point sets
    :return: the loss
    """
    def batch_pairwise_dist(x, y):
        """
        compute batch-wise distances of two point sets
        :param x: a point set
        :param y: another point set
        :return: the distance matrix
        """
        # 32, 2500, 3
        bs, num_points, points_dim = x.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
        rx = xx[:, diag_ind, diag_ind]
        rx = rx.unsqueeze(1)
        rx = rx.expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    bs, num_points, points_dim = x.size()
    dist1 = torch.sqrt(batch_pairwise_dist(x, y) + 0.00001)
    values1, indices1 = dist1.min(dim=2)

    dist2 = torch.sqrt(batch_pairwise_dist(y, x) + 0.00001)
    values2, indices2 = dist2.min(dim=2)
    a = torch.div(torch.sum(values1,1), num_points)
    b = torch.div(torch.sum(values2,1), num_points)
    sum = torch.div(torch.sum(a), bs) + torch.div(torch.sum(b), bs)
    return sum
