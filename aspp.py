import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat


class Single_ASPPModule(Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(Single_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm(planes)
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ASPP(Module):
    def __init__(self, output_stride):
        super(ASPP, self).__init__()
        inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = Single_ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = Single_ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = Single_ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = Single_ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])
        self.global_avg_pool = nn.Sequential(GlobalPooling(),
                                             nn.Conv(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm(256)
        self.relu = nn.ReLU()

    def execute(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = x5.broadcast((1,1,x4.shape[2],x4.shape[3]))
        x = concat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class GlobalPooling (Module):
    def __init__(self):
        super(GlobalPooling, self).__init__()
    def execute (self, x):
        return jt.mean(x, dims=[2,3], keepdims=1)
