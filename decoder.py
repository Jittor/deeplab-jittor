import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        low_level_inplanes = 256

        self.conv1 = nn.Conv(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Conv(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Conv(256, num_classes, kernel_size=1, stride=1, bias=True))


    def execute(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x_inter = nn.resize(x, size=(low_level_feat.shape[2], low_level_feat.shape[3]) , mode='bilinear')
        x_concat = concat((x_inter, low_level_feat), dim=1)

        x = self.last_conv(x_concat)
        return x

