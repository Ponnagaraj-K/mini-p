import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator — evaluates image patches as real/fake.
    More effective than full-image discriminator for texture/detail quality.
    """
    def __init__(self, in_channels=6, features=[64, 128, 256, 512]):
        super().__init__()
        # in_channels=6 because input = concatenated [input_image, target_image]
        layers = []
        in_ch = in_channels
        for i, out_ch in enumerate(features):
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4,
                          stride=1 if i == len(features) - 1 else 2,
                          padding=1, bias=False),
                nn.Identity() if i == 0 else nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_ch = out_ch

        layers.append(nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, input_img, target_img):
        x = torch.cat([input_img, target_img], dim=1)
        return self.model(x)


def get_discriminator(device='cpu'):
    model = PatchGANDiscriminator()
    model.to(device)
    return model
