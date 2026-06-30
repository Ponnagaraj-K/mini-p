import torch
import torch.nn as nn

# Fix: Set random seed for reproducibility
torch.manual_seed(42)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_bn=True, dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False) if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2) if down else nn.ReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    """Attention mechanism to handle red-channel loss in underwater images"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualDenoiseBlock(nn.Module):
    """Residual denoising block for edge preservation"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + 0.1 * self.block(x)


class UNetGenerator(nn.Module):
    """
    U-Net based generator for underwater image enhancement.
    Enhanced with: Color correction (ChannelAttention) + Denoising (ResidualBlocks) + Edge preservation
    """
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, features, down=True, use_bn=False)       # 128
        self.enc2 = ConvBlock(features, features * 2, down=True)                     # 64
        self.enc3 = ConvBlock(features * 2, features * 4, down=True)                 # 32
        self.enc4 = ConvBlock(features * 4, features * 8, down=True)                 # 16

        # Channel attention for color correction
        self.attention = ChannelAttention(features * 8)

        # Bottleneck with denoising
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU(),
            ResidualDenoiseBlock(features * 8)
        )

        # Decoder with skip connections + denoising
        self.dec1 = ConvBlock(features * 8, features * 8, down=False, dropout=True)          # 16
        self.denoise1 = ResidualDenoiseBlock(features * 8)
        self.dec2 = ConvBlock(features * 8 * 2, features * 4, down=False, dropout=True)      # 32
        self.denoise2 = ResidualDenoiseBlock(features * 4)
        self.dec3 = ConvBlock(features * 4 * 2, features * 2, down=False)                    # 64
        self.denoise3 = ResidualDenoiseBlock(features * 2)
        self.dec4 = ConvBlock(features * 2 * 2, features, down=False)                        # 128
        self.denoise4 = ResidualDenoiseBlock(features)

        # Output layer
        self.output = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Attention on deep features for color correction
        e4 = self.attention(e4)

        # Bottleneck with denoising
        b = self.bottleneck(e4)

        # Decoding with skip connections + denoising for edge preservation
        d1 = self.denoise1(self.dec1(b))
        d2 = self.denoise2(self.dec2(torch.cat([d1, e4], dim=1)))
        d3 = self.denoise3(self.dec3(torch.cat([d2, e3], dim=1)))
        d4 = self.denoise4(self.dec4(torch.cat([d3, e2], dim=1)))

        return self.output(torch.cat([d4, e1], dim=1))


def get_generator(pretrained_path=None, device='cpu'):
    model = UNetGenerator()
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.eval()  # Fix: always call eval() after loading
    model.to(device)
    return model
