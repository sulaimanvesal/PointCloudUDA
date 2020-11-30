from torch import nn, cat
from torch import rand
import torch
import numpy as np


class Encoder(nn.Module):

    def __init__(self, filters=64, in_channels=3, n_block=3, kernel_size=(3, 3), batch_norm=True, padding='same'):
        super().__init__()
        self.filter = filters
        for i in range(n_block):
            out_ch = filters * 2 ** i
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = filters * 2 ** (i - 1)

            if padding == 'same':
                pad = kernel_size[0] // 2
            else:
                pad = 0
            model = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                     nn.LeakyReLU(inplace=True)]
            if batch_norm:
                model += [nn.BatchNorm2d(num_features=out_ch)]
            model += [nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                      nn.LeakyReLU(inplace=True)]
            if batch_norm:
                model += [nn.BatchNorm2d(num_features=out_ch)]
            self.add_module('encoder%d' % (i + 1), nn.Sequential(*model))
            conv = [nn.Conv2d(in_channels=in_ch * 3, out_channels=out_ch, kernel_size=1), nn.LeakyReLU(inplace=True)]
            self.add_module('conv1_%d' % (i + 1), nn.Sequential(*conv))

    def forward(self, x):
        skip = []
        output = x
        res = None
        i = 0
        for name, layer in self._modules.items():
            if i % 2 == 0:
                output = layer(output)
                skip.append(output)
            else:
                if i > 1:
                    output = cat([output, res], 1)
                    output = layer(output)
                output = nn.MaxPool2d(kernel_size=(2,2))(output)
                res = output
            i += 1
        return output, skip


class Bottleneck(nn.Module):
    def __init__(self, filters=64, n_block=3, depth=4, kernel_size=(3,3)):
        super().__init__()
        out_ch = filters * 2 ** n_block
        in_ch = filters * 2 ** (n_block - 1)
        for i in range(depth):
            dilate = 2 ** i
            model = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=dilate,
                          dilation=dilate),nn.LeakyReLU(inplace=True)]
            self.add_module('bottleneck%d' % (i + 1), nn.Sequential(*model))
            if i == 0:
                in_ch = out_ch

    def forward(self, x):
        bottleneck_output = 0
        output = x
        for _, layer in self._modules.items():
            output = layer(output)
            bottleneck_output += output
        return bottleneck_output


class PointNet(nn.Module):
    def __init__(self, num_points = 300, fc_inch=81, conv_inch=512, ext=False):
        super().__init__()
        self.num_points = num_points
        self.ReLU = nn.LeakyReLU(inplace=True)
        # Final convolution is initialized differently form the rest
        if ext:
            self.conv1 = nn.Conv2d(conv_inch, conv_inch * 2, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(conv_inch * 2, conv_inch, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(conv_inch, self.num_points, kernel_size=6)
        self.final_fc = nn.Linear(fc_inch, 3)
        self._ext = ext

    def forward(self, x):
        if self._ext:
            x = self.ReLU(self.conv1(x))
            x = self.ReLU(self.conv2(x))
        x = self.ReLU(self.final_conv(x))
        x = x.view(x.size(0), x.size(1), -1)
        x = self.final_fc(x)
        return x # [8, 300, 3]



class Decoder(nn.Module):
    def __init__(self, filters=64, n_block=4, kernel_size=(3, 3), batch_norm=True, padding='same', drop=False):
        super().__init__()
        self.n_block = n_block
        if padding == 'same':
            pad = kernel_size[0] // 2
        else:
            pad = 0
        for i in reversed(range(n_block)):
            out_ch = filters * 2 ** i
            in_ch = 2 * out_ch
            model = [nn.UpsamplingNearest2d(scale_factor=2),
                     nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
                               padding=pad)]
            self.add_module('decoder1_%d' % (i + 1), nn.Sequential(*model))

            model = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                     nn.LeakyReLU(inplace=True)]
            if drop:
                model += [nn.Dropout(.5)]
            if batch_norm:
                model += [nn.BatchNorm2d(num_features=out_ch)]
            model += [nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                      nn.LeakyReLU(inplace=True)]
            if batch_norm:
                model += [nn.BatchNorm2d(num_features=out_ch)]
            self.add_module('decoder2_%d' % (i + 1), nn.Sequential(*model))

    def forward(self, x, skip):
        i = 0
        output = x
        for _, layer in self._modules.items():
            output = layer(output)
            if i % 2 == 0:
                output = cat([skip.pop(), output], 1)
            i += 1
        return output


class Segmentation_model(nn.Module):
    def __init__(self, filters=32, in_channels=3, n_block=4, bottleneck_depth=4, n_class=4, feature_dis=False):
        super().__init__()
        self.encoder = Encoder(filters=filters, in_channels=in_channels, n_block=n_block)
        self.bottleneck = Bottleneck(filters=filters, n_block=n_block, depth=bottleneck_depth)
        self.decoder = Decoder(filters=filters, n_block=n_block)
        self.classifier = nn.Conv2d(in_channels=filters, out_channels=n_class, kernel_size=(1, 1))
        if feature_dis:
            self.classifier2 = nn.Conv2d(in_channels=512, out_channels=n_class, kernel_size=(1, 1))
        self._feature_dis = feature_dis
        # self.activation = nn.Softmax2d()

    def forward(self, x, features_out=True):
        output, skip = self.encoder(x)
        output_bottleneck = self.bottleneck(output)
        output = self.decoder(output_bottleneck, skip)
        output = self.classifier(output)
        output2 = None
        if self._feature_dis:
            output2 = self.classifier2(output_bottleneck)
        if features_out:
            return output, output2, None
        else:
            return output


class Segmentation_model_Point(nn.Module):
    # filters=32, n_block=4, pointoff: 13,483,844 params
    # filters=64, n_block=3, pointoff: 13,404,804 params
    # filters=64, n_block=4, pointoff: 53,915,268 params
    def __init__(self, filters=32, in_channels=3, n_block=4, bottleneck_depth=4, n_class=4, pointnet=False,
                 feature_dis=False, drop=False, fc_inch=81, heinit=False, multicuda=False, extpn=False):
        super().__init__()
        self._pointnet = pointnet
        self.encoder = Encoder(filters=filters, in_channels=in_channels, n_block=n_block)
        self.bottleneck = Bottleneck(filters=filters, n_block=n_block, depth=bottleneck_depth)
        if pointnet:
            self.pointNet = PointNet(num_points=300, fc_inch=fc_inch, conv_inch=512 * filters//32, ext=extpn)
        self.decoder = Decoder(filters=filters, n_block=n_block, drop=drop)
        self.classifier = nn.Conv2d(in_channels=filters, out_channels=n_class, kernel_size=(1, 1))
        if feature_dis:
            self.classifier2 = nn.Conv2d(in_channels=512, out_channels=n_class, kernel_size=(1, 1))
        self._feature_dis = feature_dis
        # self.activation = nn.Softmax2d()
        self._initialize_weights(heinit=heinit)
        self._multicuda = multicuda
        if self._multicuda:
            self._cuda0 = torch.device('cuda:0')
            self._cuda1 = torch.device('cuda:1')

    def tomulticuda(self):
        if self._multicuda:
            self.encoder = self.encoder.to(self._cuda0)
            self.bottleneck = self.bottleneck.to(self._cuda1)
            self.decoder = self.decoder.to(self._cuda1)
            self.classifier = self.classifier.to(self._cuda1)
            if self._pointnet:
                self.pointNet = self.pointNet.to(self._cuda1)

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

    def forward(self, x, features_out=True, print_shape=False):
        output, skip = self.encoder(x)
        if self._multicuda:
            output = output.to(self._cuda1)
            for i in range(len(skip)):
                skip[i] = skip[i].to(self._cuda1)
        output_bottleneck = self.bottleneck(output)
        output2 = None
        if self._feature_dis:
            output2 = self.classifier2(output_bottleneck)
        output_pointNet = None
        if self._pointnet:
            output_pointNet = self.pointNet(output_bottleneck)
        output = self.decoder(output_bottleneck, skip)
        output = self.classifier(output)
        if self._multicuda:
            output = output.to(self._cuda0)
        if print_shape:
            print("bottelneck: {}".format(output_bottleneck.size()))
            print("output: {}".format(output.size()))
            if self._pointnet:
                print("pointcloud: {}".format(output_pointNet.size()))
        if features_out:
            return output, output2, output_pointNet
        else:
            return output

if __name__ == '__main__':
    img = rand((2, 3, 256, 256)).cuda()
    model = Segmentation_model_Point(filters=32, n_block=4, pointnet=False or True, fc_inch=121)
    output, _, vert = model.cuda().forward(img, print_shape=True)
    print(vert.size())
    input()
    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    print(get_n_params(model)) # 13,483,844 | 19,013,990
    #
    # img = rand((2, 3, 256, 256)).cuda()
    # model = Segmentation_model_Point(filters=32, n_block=5, pointnet=False, fc_inch=121)
    # output = model.cuda().forward(img, print_shape=True)
    # def get_n_params(model):
    #     pp = 0
    #     for p in list(model.parameters()):
    #         nn = 1
    #         for s in list(p.size()):
    #             nn = nn * s
    #         pp += nn
    #     return pp
    #
    # print(get_n_params(model)) # 13,483,844 | 19,013,990
    #
    # model = Segmentation_model_Point(filters=64, n_block=4, pointnet=False, fc_inch=121)
    # output = model.cuda().forward(img, print_shape=True)
    # def get_n_params(model):
    #     pp = 0
    #     for p in list(model.parameters()):
    #         nn = 1
    #         for s in list(p.size()):
    #             nn = nn * s
    #         pp += nn
    #     return pp

    print(get_n_params(model)) # 13,483,844 | 19,013,990
