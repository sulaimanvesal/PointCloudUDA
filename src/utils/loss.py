import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from math import log


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

class DiceCoefMultilabelLoss(nn.Module):

    def __init__(self, cuda=True):
        super().__init__()
        # self.smooth = torch.tensor(1., dtype=torch.float32)
        # self.one = torch.tensor(1., dtype=torch.float32).cuda()
        self.activation = torch.nn.Softmax2d()

    def dice_loss(self, predict, target):
        predict = predict.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = predict * target
        score = (intersection.sum() * 2. + 1.) / (predict.sum() + target.sum() + 1.)
        return 1. - score

    def forward(self, predict, target, numLabels=4, channel='channel_first'):
        assert channel == 'channel_first' or channel == 'channel_last', r"channel has to be either 'channel_first' or 'channel_last'"
        dice = 0
        predict = self.activation(predict)
        if channel == 'channel_first':
            for index in range(numLabels):
                # Lme = [0.1, 0.1, 0.3, 0.5]
                temp = self.dice_loss(predict[:, index, :, :], target[:, index, :, :])
                dice += temp
        else:
            for index in range(numLabels):
                # Lme = [0.1, 0.1, 0.3, 0.5]
                temp = self.dice_loss(predict[:, :, :, index], target[:, :, :, index])
                dice += temp

        dice = dice / numLabels

        # entropy = entropy_loss(y_true) + ent
        return dice

class ShannonEntropyLoss(nn.Module):

    def __init__(self, cuda=True, loss_weight=1, n_class=4):
        super().__init__()
        self.loss_weight = loss_weight
        self.n_class = n_class
        self.activation = torch.nn.Softmax2d()

    def forward(self, predict):
        predict = self.activation(predict)
        return self.loss_weight * -(predict * torch.log(predict + 1e-20)).mean() / log(self.n_class)

def batch_pairwise_dist(x, y):
    """
    compute the distances of point pairs
    :param x: a point set
    :param y: another point set
    :return: distance matrix
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

def batch_NN_loss(x, y):
    """
    the loss function for point cloud regression
    :param x: the prediction of point cloud
    :param y: the ground truth
    :return:
    """

    bs, num_points, points_dim = x.size()
    dist1 = torch.sqrt(batch_pairwise_dist(x, y) + 0.00001)
    values1, indices1 = dist1.min(dim=2)

    dist2 = torch.sqrt(batch_pairwise_dist(y, x) + 0.00001)
    values2, indices2 = dist2.min(dim=2)
    a = torch.div(torch.sum(values1,1), num_points)
    b = torch.div(torch.sum(values2,1), num_points)
    sum = torch.div(torch.sum(a), bs) + torch.div(torch.sum(b), bs)

    return sum


def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
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
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)
