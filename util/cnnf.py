import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""Ref:
- https://zhuanlan.zhihu.com/p/29786939
- https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
- https://pytorch.org/docs/stable/nn.html#zeropad2d
- https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/pad.md
"""


class CNN_F(nn.Module):
    """CNN-F / VGG-F"""

    def __init__(self, weight_file="../data/DCMH.imagenet-vgg-f.mat"):
        super(CNN_F, self).__init__()
        # layers = sio.loadmat(weight_file)["net"][0][0][0][0]  # vgg_net.mat
        layers = sio.loadmat(weight_file)['layers'][0]  # DCMH.imagenet-vgg-f.mat
        # print(layers.shape)  # (19,)

        self.conv = nn.Sequential(
            nn.Sequential(
                make_conv(layers[0]),
                # nn.BatchNorm2d(layers[0][0][0][3][0][-1]),
                nn.ReLU(inplace=True),
                # make_lrn(layers[2]),
                nn.LocalResponseNorm(2, alpha=0.0001, beta=0.75, k=2.0),
                make_pool(layers[3])
            ),
            nn.Sequential(
                make_conv(layers[4]),
                # nn.BatchNorm2d(layers[4][0][0][3][0][-1]),
                nn.ReLU(inplace=True),
                # make_lrn(layers[6]),
                nn.LocalResponseNorm(2, alpha=0.0001, beta=0.75, k=2.0),
                make_pool(layers[7])
            ),
            nn.Sequential(
                make_conv(layers[8]),
                # nn.BatchNorm2d(layers[8][0][0][3][0][-1]),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                make_conv(layers[10]),
                # nn.BatchNorm2d(layers[10][0][0][3][0][-1]),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                make_conv(layers[12]),
                # nn.BatchNorm2d(layers[12][0][0][3][0][-1]),
                nn.ReLU(inplace=True),
                make_pool(layers[14])
            ),
        )

        self.fc = nn.Sequential(
            nn.Sequential(
                make_conv(layers[15]),
                # nn.BatchNorm2d(layers[15][0][0][3][0][-1]),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.5)
            ),
            nn.Sequential(
                make_conv(layers[17]),
                # nn.BatchNorm2d(layers[17][0][0][3][0][-1]),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.5)
            ),
            nn.Flatten(),  # (n, 4096)
        )

    def forward(self, x, skip=None):
        assert skip in (None, "skip-conv", "skip-fc")
        if skip != "skip-conv":
            x = self.conv(x)
        if skip != "skip-fc":
            x = self.fc(x)
        return x


def make_conv(layer):
    """pytorch: (n, C, h, w)
    tf: (n, h, w, C)
    ref:
    - https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L329
    """
    layer = layer[0][0]
    # print("name:", layer[0])
    # print("type:", layer[1])
    # k, b = layer[2][0]
    k, b = layer[0][0]
    #b = b.flatten()
    # print("kernel:", k.shape, ", bias:", b.shape)
    # shape = layer[3][0]
    shape = k.shape
    # print("shape:", shape)
    # pad = layer[4][0]
    pad = layer[1][0]
    # print("pad:", pad)
    # stride = layer[5][0]
    stride = layer[4][0]
    # print("stride:", stride)

    conv = nn.Conv2d(shape[2], shape[3], shape[:2],
                     stride=tuple(stride))  # must convert to tuple
                    #  padding=tuple(pad))
    conv.weight.data = torch.from_numpy(k.transpose((3, 2, 0, 1)))
    conv.bias.data = torch.from_numpy(b.flatten())

    if np.sum(pad) > 0:
        padding = nn.ZeroPad2d(tuple(pad.astype(np.int32)))
        conv = nn.Sequential(padding, conv)

    return conv


class LRN(nn.Module):
    """ref:
    - https://zhuanlan.zhihu.com/p/29786939
    - https://www.jianshu.com/p/c06aea337d5d
    """
    def __init__(self, local_size=1, bias=1.0, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1), #0.2.0_4会报错，需要在最新的分支上AvgPool3d才有padding参数
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))

        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.bias).pow(self.beta)#这里的1.0即为bias
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.bias).pow(self.beta)
        x = x.div(div)
        return x


def make_lrn(layer):
    layer = layer[0][0]
    # print("name:", layer[0])
    # print("type:", layer[1])
    param = layer[2][0]
    # print("local_size/depth_radius:", param[0])
    # print("bias:", param[1])
    # print("alpha:", param[2])
    # print("beta:", param[3])

    lrn = LRN(int(param[0]), param[1], param[2], param[3])
    return lrn


def make_pool(layer):
    layer = layer[0][0]
    # print("name:", layer[0])
    # print("type:", layer[1])
    # print("pool type:", layer[2])
    # k_size = layer[3][0]
    k_size = layer[5][0]
    # stride = layer[4][0]
    stride = layer[1][0]
    # print("stride:", stride)
    # pad = layer[5][0]
    pad = layer[2][0]
    # print("pad:", pad)

    pool = nn.MaxPool2d(tuple(k_size),
                        stride=tuple(stride))
                        # padding=tuple(pad))
    if np.sum(pad) > 0:
        padding = nn.ZeroPad2d(tuple(pad.astype(np.int32)))
        pool = nn.Sequential(padding, pool)

    return pool


if __name__ == "__main__":
    cnnf = CNN_F("../data/DCMH.imagenet-vgg-f.mat")
    print(cnnf)
    x = torch.empty(1, 3, 224, 224)
    o = cnnf(x)
    print(o.size())
