import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import to_grayscale
from torchvision import datasets, transforms
from WA import WaveletAttention
# 定义 U-Net 的各层（DoubleConv, Down, Up, OutConv 等）
import torch
import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    if dilation==1:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias)
    elif dilation==2:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=2, bias=bias, dilation=dilation)

    else:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=3, bias=bias, dilation=dilation)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class ResAttentionBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttentionBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        m.append(CALayer(n_feats, 16))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class SSB(nn.Module):
    def __init__(self, n_feats, kernel_size, act,res_scale,conv=default_conv):
        super(SSB, self).__init__()
        self.spa = ResBlock(conv,
                            n_feats,
                            kernel_size,
                            act=act,
                            res_scale=res_scale)
        self.spc = ResAttentionBlock(conv,
                                     n_feats,
                                     1,
                                     act=act,
                                     res_scale=res_scale)

    def forward(self, x):
        return self.spc(self.spa(x))


class DynamicSpatialAttention(nn.Module):
    def __init__(self, in_channels=32, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Conv2d(in_channels, in_channels, kernel_size=1),                                                                                                  # 微信公众号:AI缝合术
            nn.ReLU(),
            nn.Conv2d(in_channels, kernel_size**2, kernel_size=1)  # [B, k*k, 1, 1]                                                                                               # 微信公众号:AI缝合术
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 每个样本生成一个动态卷积核 [B, k*k, 1, 1] → [B, 1, k, k]                                                                                                      # 微信公众号:AI缝合术
        kernels = self.kernel_generator(x).view(B, 1, self.kernel_size, self.kernel_size)                                                                                        # 微信公众号:AI缝合术                                                  # 微信公众号:AI缝合术
        # 2. 对每个样本取通道平均 [B, 1, H, W]
        x_mean = x.mean(dim=1, keepdim=True)
        # 3. reshape 成 grouped convolution 所需格式
        x_mean = x_mean.view(1, B, H, W)  # → [1, B, H, W]
        kernels = kernels.view(B, 1, self.kernel_size, self.kernel_size)  # [B, 1, k, k]                                                                                # 微信公众号:AI缝合术
        # 4. 执行 grouped convolution，每个 kernel 只作用于对应的样本
        att = F.conv2d(
            x_mean,
            weight=kernels,
            padding=self.kernel_size // 2,
            groups=B
        )
        # 5. reshape 回原格式 + sigmoid
        att = att.view(B, 1, H, W)
        att = self.sigmoid(att)
        # 6. 应用注意力图
        return x * att



# U-Net 架构
class CSSAB(nn.Module):
    def __init__(self, P, L, size,):
        super(CSSAB, self).__init__()
        self.P, self.L, self.size = P, L, size
        kernel_size = 3
        n_blocks = 3
        act = nn.ReLU(True)
        res_scale = 0.1
        self.encoder2 = nn.Sequential(
            # SpectralSpatialAttention(in_channels=L),
            nn.Conv2d(L, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            # SpectralSpatialAttention(in_channels=128),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(),
            # SpectralSpatialAttention(in_channels=64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, momentum=0.5),
            nn.LeakyReLU(),
            # SpectralSpatialAttention(in_channels=32),
            nn.BatchNorm2d(32, momentum=0.5),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            # SpectralSpatialAttention(in_channels=16),
            nn.BatchNorm2d(16, momentum=0.5),
            nn.LeakyReLU(),
            nn.Conv2d(16, self.P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.P, momentum=0.5),

        )
        self.encoder3 = nn.Sequential(

            SSB(L, kernel_size, act=act, res_scale=res_scale),
            nn.Conv2d(L, 120, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(120),
            nn.Dropout(0.1),
            SSB(120, kernel_size, act=act, res_scale=res_scale),

            nn.Conv2d(120, 60, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(60),
            nn.Dropout(0.1),

            nn.Conv2d(60, self.P, kernel_size=3, stride=1, padding=1),
        )
        self.smooth = nn.Sequential(
            # nn.Conv2d(P, P, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Softmax(dim=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )

        self.wa = WaveletAttention(channels=L, use_fc=True)

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def get_last_layer(self):
        conv_layer = self.decoder[0]
        weight = conv_layer.weight
        return weight

    def forward(self, x):
        x = self.wa(x)
        x = self.encoder3(x)
        abu_est = self.smooth(x)
        re_result = self.decoder(abu_est)

        # decoder_weight = self.get_last_layer()
        return abu_est, re_result

# 现在你可以创建一个U-Net模型实例并进行训练或预测：
#model = UNet(n_channels=156, n_classes=3, bilinear=False, spec_type="all")

# 假设输入形状为(95, 95, 156)，需要调整为(1, 156, 95, 95)
#input_tensor = torch.randn(1, 156, 224, 224)  # batch_size=1, channels=156, height=95, width=95
#output = model(input_tensor)
#print(output.shape)

