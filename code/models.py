import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.center = self.conv_block(256, 512)
        self.dec4 = self.conv_block(512 + 256, 256)
        self.dec3 = self.conv_block(256 + 128, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec1 = self.conv_block(64 + 32, 32)
        self.final = nn.Conv2d(32, 3, kernel_size=1)
        self._initialize_weights()

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))
        center = self.center(nn.MaxPool2d(2)(enc4))
        dec4 = self.dec4(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(center), enc4], dim=1))
        dec3 = self.dec3(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(dec2), enc1], dim=1))
        output = self.final(dec1)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


class Critic(nn.Module):
    def __init__(self, channels=4, conv_dim=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, conv_dim, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(conv_dim, conv_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(conv_dim * 4, 1, 4, 2, 1, bias=True),
            nn.AdaptiveAvgPool2d(1)

        )
        self._initialize_weights()

    def forward(self, x):
        return self.net(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)