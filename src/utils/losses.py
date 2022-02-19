import torch
from torch import nn
from torch.nn import functional as F


class InterpolatedSupervisedContrastiveLoss(nn.Module):

    def __init__(self, temperature):
        super(InterpolatedSupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(
            self,
            features,
            labels_1,
            labels_2,
            dominant_labels,
            lambdas,
            normalize=True):
        """Computes the Interpolated Supervised Contrastive Loss (ISCL).

        Args:
            features (torch.Tensor): Features representation with shape (N ,d)
                N corresponds to batch size and d to features dimension.
            labels_1 (torch.Tensor): Grountruth labels associated to
                interpolation coefficients $\lambda$ with shape (N,)
            labels_2 (torch.Tensor): Grountruth labels associated to
                interpolation coefficients $1 - \lambda$ with shape (N,)
            dominant_labels (torch.Tensor): Dominant grountruth labels with
                shape (N,). dominant labels are the labels associated to the
                highest corresponding interpolation coefficients.
                (i. e $\max(\lambda, 1 - \lambda)$)
            lambdas (torch.Tensor): Interpolation coefficients drawn in Mixup
                with shape (N, ).
            normalize (bool, optional): Boolean to indicate whether or not
                apply a L2 normalization. If set to false, it means that
                `features`have already been normalized.

        Returns:
            Torch.Tensor: A loss scalar.
        """

        # Normalize features if required
        if normalize:
            features = F.normalize(features, dim=-1, p=2)
        # Get the pairwise distances matrix with shape (batch_size, batch_size)
        pairwise_distance = torch.matmul(
            features, features.T) / self.temperature
        # For numerical stability
        logits_max, _ = torch.max(pairwise_distance, dim=1, keepdim=True)
        pairwise_distance = pairwise_distance - logits_max.detach()
        # Compute losses for each sample
        losses = lambdas * self.compute_per_sample_loss(
            pairwise_distance, labels_1, dominant_labels) + \
            (1-lambdas) * self.compute_per_sample_loss(
                pairwise_distance, labels_2, dominant_labels)
        return losses.mean()

    def compute_per_sample_loss(
            self,
            pairwise_distance,
            query_labels,
            anchors_labels):

        n = pairwise_distance.size(0)
        anchor_matrix = torch.logical_not(
            torch.eye(n, dtype=torch.bool, device=pairwise_distance.device))
        # Get the positive matrix where row with index `i`
        # indicates for a query at index i which anchors have the labels y_i
        pos_matrix = query_labels.unsqueeze(1).eq(anchors_labels.unsqueeze(0))
        # Remove comparison between query and itself (i.e elements on the diagonal)
        pos_matrix = torch.logical_and(pos_matrix, anchor_matrix)
        # Get positive counts for each query
        pos_counts = (pos_matrix.sum(dim=-1)).type(pairwise_distance.type())

        exp_logits = torch.exp(pairwise_distance)
        losses = - (1 / pos_counts) * (pos_matrix * (
            pairwise_distance - torch.log(
                (anchor_matrix * exp_logits).sum(dim=-1, keepdim=True)
            ))).sum(dim=-1)
        return losses


class SoftmaxCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(SoftmaxCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = inputs.clamp(1e-6, 1)
        loss = - (targets * torch.log(inputs)).sum(dim=-1).mean()
        return loss


class SupConLoss(nn.Module):
    """modified supcon loss for segmentation application, the main difference is that the label for different view
    could be different if after spatial transformation"""
    def __init__(self, temperature=0.07,
                 contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):
        # input features shape: [bsz, v, c, w, h]
        # input labels shape: [bsz, v, w, h]
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # of size (bsz*v, c, h, w)

        kernels = contrast_feature.permute(0, 2, 3, 1)
        kernels = kernels.reshape(-1, contrast_feature.shape[1], 1, 1)
        # kernels = kernels[non_background_idx]
        logits = torch.div(F.conv2d(contrast_feature, kernels), self.temperature)  # of size (bsz*v, bsz*v*h*w, h, w)
        logits = logits.permute(1, 0, 2, 3)
        logits = logits.reshape(logits.shape[0], -1)

        if labels is not None:
            labels = torch.cat(torch.unbind(labels, dim=1), dim=0)
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)

            bg_bool = torch.eq(labels.squeeze().cpu(), torch.zeros(labels.squeeze().shape))
            non_bg_bool = ~ bg_bool
            non_bg_bool = non_bg_bool.int().to(device)
        else:
            mask = torch.eye(logits.shape[0]//contrast_count).float().to(device)
            mask = mask.repeat(contrast_count, contrast_count)
            # print(mask.shape)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(mask.shape[0]).view(-1, 1).to(device),0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        if labels is not None:
            # only consider the contrastive loss for non-background pixel
            loss = (loss * non_bg_bool).sum() / (non_bg_bool.sum())
        else:
            loss = loss.mean()
        return loss


class LocalConLoss(nn.Module):
    def __init__(self, temperature=0.7, stride=4):
        super(LocalConLoss, self).__init__()
        self.temp = temperature
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.supconloss = SupConLoss(temperature=self.temp)
        self.stride = stride

    def forward(self, features, labels=None):
        # input features: [bsz, num_view, c, h ,w], h & w are the image size
        # resample feature maps to reduce memory consumption and running time
        features = features[:, :, :, ::self.stride, ::self.stride]

        if labels is not None:
            labels = labels[:, :, ::self.stride, ::self.stride]
            if labels.sum() == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss

            loss = self.supconloss(features, labels)
            return loss
        else:
            loss = self.supconloss(features)
            return loss


class BlockConLoss(nn.Module):
    def __init__(self, temperature=0.7, block_size=32):
        super(BlockConLoss, self).__init__()
        self.block_size = block_size
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.supconloss = SupConLoss(temperature=temperature)

    def forward(self, features, labels=None):
        # input features: [bsz, num_view, c, h ,w], h & w are the image size
        shape = features.shape
        img_size = shape[-1]
        div_num = img_size // self.block_size
        if labels is not None:
            loss = []
            for i in range(div_num):
                # print("Iteration index:", idx, "Batch_size:", b)
                for j in range(div_num):
                    # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                    block_features = features[:, :, :, i*self.block_size:(i+1)*self.block_size,
                                  j*self.block_size:(j+1)*self.block_size]
                    block_labels = labels[:,:, i*self.block_size:(i+1)*self.block_size,
                                  j*self.block_size:(j+1)*self.block_size]

                    if block_labels.sum() == 0:
                        continue

                    tmp_loss = self.supconloss(block_features, block_labels)
                    loss.append(tmp_loss)

            if len(loss) == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss
            loss = torch.stack(loss).mean()
            return loss

        else:
            loss = []
            for i in range(div_num):
                # print("Iteration index:", idx, "Batch_size:", b)
                for j in range(div_num):
                    # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                    block_features = features[:, :, :, i * self.block_size:(i + 1) * self.block_size,
                                     j * self.block_size:(j + 1) * self.block_size]

                    tmp_loss = self.supconloss(block_features)

                    loss.append(tmp_loss)

            loss = torch.stack(loss).mean()
            return loss