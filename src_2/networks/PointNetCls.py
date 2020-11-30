from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    """
    Spatial transformer network
    Compute transformation matrix of the inputted pointcloud
    """
    def __init__(self, dim=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.in1 = nn.InstanceNorm1d(64, track_running_stats=True)
        self.in2 = nn.InstanceNorm1d(128, track_running_stats=True)
        self.in3 = nn.InstanceNorm1d(1024, track_running_stats=True)
        self.in4 = nn.InstanceNorm1d(512, track_running_stats=True)
        self.in5 = nn.InstanceNorm1d(256, track_running_stats=True)


    def forward(self, x):
        batchsize = x.size()[0]
        if batchsize > 1:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.in1(self.conv1(x)))
            x = F.relu(self.in2(self.conv2(x)))
            x = F.relu(self.in3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            x = F.relu(self.in4(self.fc1(x)))
            x = F.relu(self.in5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False, sample_transform=True, kernel_size=1, stride=1, in_channel=3, dim=3, ext=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(dim=dim)
        self._ext = ext
        if self._ext:
            self.conv1 = torch.nn.Conv1d(in_channel, 8, kernel_size, stride, kernel_size // 2)
            self.bn1 = nn.BatchNorm1d(8)
            self.conv1_1 = torch.nn.Conv1d(8, 64, kernel_size, stride, kernel_size // 2)
            self.bn1_1 = nn.BatchNorm1d(64)
            self.conv2 = torch.nn.Conv1d(64, 128, kernel_size, stride, kernel_size // 2)
            self.bn2 = nn.BatchNorm1d(128)
            self.conv2_1 = torch.nn.Conv1d(128, 256, kernel_size, stride, kernel_size // 2)
            self.bn2_1 = nn.BatchNorm1d(256)
            self.conv3 = torch.nn.Conv1d(256, 512, kernel_size, stride, kernel_size // 2)
            self.bn3 = nn.BatchNorm1d(512)
            self.conv3_1 = torch.nn.Conv1d(512, 1024, kernel_size, stride, kernel_size // 2)
            self.bn3_1 = nn.BatchNorm1d(1024)
        else:
            self.conv1 = torch.nn.Conv1d(in_channel, 64, kernel_size, stride, kernel_size // 2)
            self.conv2 = torch.nn.Conv1d(64, 128, kernel_size, stride, kernel_size // 2)
            self.conv3 = torch.nn.Conv1d(128, 1024, kernel_size, stride, kernel_size // 2)
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self._sample_transform = sample_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = None
        if self._sample_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        if self._ext:
            x = F.relu(self.bn1_1(self.conv1_1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        if self._ext:
            x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = self.bn3(self.conv3(x))
        if self._ext:
            x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, feature_transform=False, sample_transform=True, kernel_size=1, stride=1, in_channel=3, dim=3, ext=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, sample_transform=sample_transform,
                                 kernel_size=kernel_size, stride=stride, in_channel=in_channel, dim=dim, ext=ext)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.in1 = nn.InstanceNorm1d(512, track_running_stats=True)
        self.in2 = nn.InstanceNorm1d(256, track_running_stats=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        batchsize = x.size()[0]
        if batchsize > 1:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        else:
            x = F.relu(self.in1(self.fc1(x)))
            x = F.relu(self.in2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    #batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = torch.rand(1,3,3600).cuda()
    cls = PointNetCls().cuda()
    out, _, _ = cls(sim_data)
    print('class', out.size())

