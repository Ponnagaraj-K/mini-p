import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights


class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss — preserves structural features important for detection"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        # Use features up to relu3_3
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, enhanced, target):
        enhanced_features = self.feature_extractor(enhanced)
        target_features = self.feature_extractor(target)
        return self.criterion(enhanced_features, target_features)


class SSIMLoss(nn.Module):
    """SSIM-based structural similarity loss"""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.channel = 3
        self.window = self._create_window()

    def _create_window(self):
        import math
        sigma = 1.5
        gauss = torch.Tensor([
            math.exp(-(x - self.window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(self.window_size)
        ])
        gauss = gauss / gauss.sum()
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
        return window_2d.expand(self.channel, 1, self.window_size, self.window_size).contiguous()

    def forward(self, img1, img2):
        import torch.nn.functional as F
        window = self.window.to(img1.device)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()


class EnhancementLoss(nn.Module):
    """
    Combined loss for underwater enhancement:
    - Adversarial loss: makes output look realistic
    - Perceptual loss: preserves features needed for detection
    - SSIM loss: maintains structural integrity
    - L1 loss: pixel-level accuracy
    """
    def __init__(self, lambda_perceptual=5.0, lambda_ssim=10.0, lambda_l1=150.0):
        super().__init__()
        self.perceptual = PerceptualLoss()
        self.ssim = SSIMLoss()
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim
        self.lambda_l1 = lambda_l1

    def generator_loss(self, disc_fake, enhanced, target):
        adv_loss = self.bce(disc_fake, torch.ones_like(disc_fake))
        perc_loss = self.perceptual(enhanced, target)
        ssim_loss = self.ssim(enhanced, target)
        l1_loss = self.l1(enhanced, target)
        total = adv_loss + self.lambda_perceptual * perc_loss + \
                self.lambda_ssim * ssim_loss + self.lambda_l1 * l1_loss
        return total, {
            "adversarial": adv_loss.item(),
            "perceptual": perc_loss.item(),
            "ssim": ssim_loss.item(),
            "l1": l1_loss.item()
        }

    def discriminator_loss(self, disc_real, disc_fake):
        real_loss = self.bce(disc_real, torch.ones_like(disc_real))
        fake_loss = self.bce(disc_fake, torch.zeros_like(disc_fake))
        return (real_loss + fake_loss) / 2
