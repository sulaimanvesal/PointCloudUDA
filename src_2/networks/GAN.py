import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()

        filter_num_list = [4096, 2048, 1024, 1]

        self.fc1 = nn.Linear(24576, filter_num_list[0])
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(filter_num_list[0], filter_num_list[1])
        self.fc3 = nn.Linear(filter_num_list[1], filter_num_list[2])
        self.fc4 = nn.Linear(filter_num_list[2], filter_num_list[3])

        # self.sigmoid = nn.Sigmoid()
        self._initialize_weights()


    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    # m.bias.data.copy_(1.0)
                    m.bias.data.zero_()


    def forward(self, x):

        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.leakyrelu(self.fc3(x))
        x = self.fc4(x)
        return x


class OutputDiscriminator(nn.Module):
    def __init__(self, in_channel=2, softmax=False, init=False):
        super(OutputDiscriminator, self).__init__()
        self._softmax = softmax
        filter_num_list = [64, 128, 256, 512, 1]
        self.upsample = nn.UpsamplingBilinear2d(size=(224, 224))
        self.conv1 = nn.Conv2d(in_channel, filter_num_list[0], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(filter_num_list[0], filter_num_list[1], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(filter_num_list[1], filter_num_list[2], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv4 = nn.Conv2d(filter_num_list[2], filter_num_list[3], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv5 = nn.Conv2d(filter_num_list[3], filter_num_list[4], kernel_size=4, stride=2, padding=2, bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        # self.sigmoid = nn.Sigmoid()
        if init:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        x = self.upsample(x)
        if self._softmax:
            x = F.softmax(x, dim=1)
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.conv4(x))
        x = self.conv5(x)
        return x


class UncertaintyDiscriminator(nn.Module):
    def __init__(self, in_channel=2, heinit=False, ext=False):
        # assert not(softmax and sigmoid), "Only one of 'softmax' or 'sigmoid' can be used for activation function."
        super(UncertaintyDiscriminator, self).__init__()
        # self._softmax = softmax
        # self._sigmoid = sigmoid
        filter_num_list = [64, 128, 256, 512, 1]

        self.conv1 = nn.Conv2d(in_channel, filter_num_list[0], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(filter_num_list[0], filter_num_list[1], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(filter_num_list[1], filter_num_list[2], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv4 = nn.Conv2d(filter_num_list[2], filter_num_list[3], kernel_size=4, stride=2, padding=2, bias=False)
        if ext:
            self.conv4_2 = nn.Conv2d(filter_num_list[3], 1024, kernel_size=3, stride=2, padding=1, bias=False)
            self.conv4_3 = nn.Conv2d(1024, filter_num_list[2], kernel_size=3, stride=2, padding=1, bias=False)
            self.conv5 = nn.Conv2d(filter_num_list[2], filter_num_list[4], kernel_size=4, stride=2, padding=2,
                                   bias=False)
        else:
            self.conv5 = nn.Conv2d(filter_num_list[3], filter_num_list[4], kernel_size=4, stride=2, padding=2, bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        self._ext = ext
        # self.sigmoid = nn.Sigmoid()
        self._initialize_weights(heinit=heinit)


    def _initialize_weights(self, heinit=False):
        if heinit:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    prod = float(np.prod(m.weight.size()[1:]))
                    prod = np.sqrt(2 / prod)
                    m.weight.data.normal_(0.0, prod)
                    if m.bias is not None:
                        m.bias.data.zero_()
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, 0.02)
                    if m.bias is not None:
                        m.bias.data.zero_()


    def forward(self, x):
        # if self._softmax:
        #     x = F.softmax(x, dim=1)
        # elif self._sigmoid:
        #     x = F.sigmoid(x)
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.conv4(x))
        if self._ext:
            x = self.leakyrelu(self.conv4_2(x))
            x = self.leakyrelu(self.conv4_3(x))
        x = self.conv5(x)
        return x

class BoundaryDiscriminator(nn.Module):
    def __init__(self, ):
        super(BoundaryDiscriminator, self).__init__()

        filter_num_list = [64, 128, 256, 512, 1]

        self.conv1 = nn.Conv2d(1, filter_num_list[0], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(filter_num_list[0], filter_num_list[1], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(filter_num_list[1], filter_num_list[2], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv4 = nn.Conv2d(filter_num_list[2], filter_num_list[3], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv5 = nn.Conv2d(filter_num_list[3], filter_num_list[4], kernel_size=4, stride=2, padding=2, bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        # self.sigmoid = nn.Sigmoid()
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.conv4(x))
        x = self.conv5(x)
        return x

class BoundaryEntDiscriminator(nn.Module):
    def __init__(self, ):
        super(BoundaryEntDiscriminator, self).__init__()

        filter_num_list = [64, 128, 256, 512, 1]

        self.conv1 = nn.Conv2d(3, filter_num_list[0], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(filter_num_list[0], filter_num_list[1], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(filter_num_list[1], filter_num_list[2], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv4 = nn.Conv2d(filter_num_list[2], filter_num_list[3], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv5 = nn.Conv2d(filter_num_list[3], filter_num_list[4], kernel_size=4, stride=2, padding=2, bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        # self.sigmoid = nn.Sigmoid()
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.conv4(x))
        x = self.conv5(x)
        return x

if __name__ == '__main__':
    model_dis = UncertaintyDiscriminator(in_channel=2).cuda()
    img = torch.rand((1, 2, 256, 256)).cuda()
    output = model_dis(img)
    print(output.size())
