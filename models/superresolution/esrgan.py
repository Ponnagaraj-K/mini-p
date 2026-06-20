import torch
import torch.nn as nn


class ResidualDenseBlock(nn.Module):
    def __init__(self, features=64, growth=32):
        super().__init__()
        self.conv1 = nn.Conv2d(features, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(features + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(features + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(features + 3 * growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(features + 4 * growth, features, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.scale = 0.2

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * self.scale + x


class RRDB(nn.Module):
    def __init__(self, features=64):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(features)
        self.rdb2 = ResidualDenseBlock(features)
        self.rdb3 = ResidualDenseBlock(features)
        self.scale = 0.2

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * self.scale + x


class ESRGANGenerator(nn.Module):
    """
    Lightweight ESRGAN for 4x super resolution.
    Recovers fine details (hull shape, fin structure) needed for submarine identification.
    Reduced RRDB blocks (6 instead of 23) for edge device compatibility.
    """
    def __init__(self, in_channels=3, out_channels=3, features=64, num_rrdb=6, scale=4):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, features, 3, 1, 1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(features) for _ in range(num_rrdb)])
        self.conv_body = nn.Conv2d(features, features, 3, 1, 1)

        # Upsampling
        upsample_layers = []
        for _ in range(scale // 2):
            upsample_layers += [
                nn.Conv2d(features, features * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        self.upsample = nn.Sequential(*upsample_layers)
        self.conv_last = nn.Conv2d(features, out_channels, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        body = self.conv_body(self.rrdb_blocks(feat))
        feat = feat + body
        return self.conv_last(self.upsample(feat))


def get_esrgan(pretrained_path=None, device='cpu'):
    model = ESRGANGenerator()
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.eval()
    model.to(device)
    return model
