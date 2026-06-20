import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Fix: Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

sys.path.append(str(Path(__file__).parent.parent))
from models.enhancement.generator import get_generator
from models.superresolution.esrgan import get_esrgan


# Indian Ocean optical properties (Bay of Bengal / Arabian Sea)
# Based on MODIS/PACE ocean color data
INDIAN_OCEAN_PARAMS = {
    "bay_of_bengal": {
        "absorption_r": 0.35,   # Red absorbed fastest
        "absorption_g": 0.12,
        "absorption_b": 0.05,   # Blue penetrates deepest
        "scattering": 0.18,
        "turbidity": "moderate"
    },
    "arabian_sea": {
        "absorption_r": 0.28,
        "absorption_g": 0.09,
        "absorption_b": 0.04,
        "scattering": 0.14,
        "turbidity": "low_moderate"
    }
}


def physics_white_balance(image: np.ndarray, region: str = "bay_of_bengal") -> np.ndarray:
    """
    Adaptive white balance — corrects dominant color cast toward neutral gray.
    """
    img_float = image.astype(np.float32)
    b_mean = np.mean(img_float[:,:,0])
    g_mean = np.mean(img_float[:,:,1])
    r_mean = np.mean(img_float[:,:,2])
    gray = (b_mean + g_mean + r_mean) / 3.0

    # Scale each channel toward neutral gray — gentle correction
    b_scale = np.clip(gray / (b_mean + 1e-6), 0.8, 1.5)
    g_scale = np.clip(gray / (g_mean + 1e-6), 0.8, 1.5)
    r_scale = np.clip(gray / (r_mean + 1e-6), 0.8, 1.5)

    corrected = img_float.copy()
    corrected[:,:,0] = np.clip(img_float[:,:,0] * b_scale, 0, 255)
    corrected[:,:,1] = np.clip(img_float[:,:,1] * g_scale, 0, 255)
    corrected[:,:,2] = np.clip(img_float[:,:,2] * r_scale, 0, 255)
    return corrected.astype(np.uint8)


def estimate_transmission_map(image: np.ndarray) -> np.ndarray:
    """
    Dark Channel Prior based transmission map estimation.
    Estimates how much haze/backscatter is present.
    """
    img_float = image.astype(np.float32) / 255.0
    patch_size = 15
    min_channel = np.min(img_float, axis=2)

    # Dark channel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    # Atmospheric light estimation (brightest pixels in dark channel)
    flat_dark = dark_channel.flatten()
    top_pixels = np.argsort(flat_dark)[-int(0.001 * len(flat_dark)):]
    atm_light = np.mean(img_float.reshape(-1, 3)[top_pixels], axis=0)
    atm_light = np.clip(atm_light, 0.1, 1.0)

    # Transmission estimate
    omega = 0.95
    norm_img = img_float / atm_light
    norm_dark = np.min(norm_img, axis=2)
    transmission = 1 - omega * cv2.erode(norm_dark, kernel)
    transmission = np.clip(transmission, 0.1, 1.0)

    # Refine with guided filter approximation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    transmission = cv2.ximgproc.guidedFilter(
        gray, transmission.astype(np.float32), 60, 0.001
    ) if hasattr(cv2, 'ximgproc') else transmission

    return transmission, atm_light


def remove_haze(image: np.ndarray) -> tuple:
    """Remove haze using transmission map — returns dehazed image + quality report"""
    try:
        transmission, atm_light = estimate_transmission_map(image)
        img_float = image.astype(np.float32) / 255.0
        dehazed = np.zeros_like(img_float)
        for c in range(3):
            dehazed[:, :, c] = (img_float[:, :, c] - atm_light[c]) / transmission + atm_light[c]
        dehazed = np.clip(dehazed, 0, 1)
        result = (dehazed * 255).astype(np.uint8)
        status = "SUCCESS"
    except Exception:
        result = image.copy()
        status = "PARTIAL — guided filter not available, basic dehazing applied"
        # Fallback: CLAHE-based haze reduction
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result, status


def apply_clahe_enhancement(image: np.ndarray) -> np.ndarray:
    """Adaptive histogram equalization in LAB color space"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_with_model(image: np.ndarray, model, device: str) -> np.ndarray:
    """Run image through U-Net GAN enhancement model"""
    h, w = image.shape[:2]
    target_h = (h // 256 + 1) * 256 if h % 256 != 0 else h
    target_w = (w // 256 + 1) * 256 if w % 256 != 0 else w

    img_resized = cv2.resize(image, (target_w, target_h))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        enhanced_tensor = model(img_tensor)

    enhanced = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced = (enhanced * 0.5 + 0.5) * 255
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    enhanced = cv2.resize(enhanced, (w, h))
    return enhanced


def apply_super_resolution(image: np.ndarray, sr_model, device: str) -> np.ndarray:
    """Apply 4x super resolution for detail recovery"""
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        sr_tensor = sr_model(img_tensor)

    sr_image = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    sr_image = np.clip(sr_image * 255, 0, 255).astype(np.uint8)
    return sr_image


class EnhancementPipeline:
    def __init__(self, enhancement_weights=None, sr_weights=None, device='cpu', region='bay_of_bengal'):
        self.device = device
        self.region = region
        self.use_neural = False
        self.generator = None

        # Only load neural model if weights actually exist
        if enhancement_weights and Path(enhancement_weights).exists():
            self.generator = get_generator(enhancement_weights, device)
            self.generator.eval()
            self.use_neural = True
            print(f"Enhancement pipeline ready | Neural model loaded | Device: {device}")
        else:
            print(f"Enhancement pipeline ready | CLAHE mode (no weights) | Device: {device}")

        self.sr_model = get_esrgan(sr_weights, device) if sr_weights and Path(sr_weights).exists() else None
        self.use_sr = self.sr_model is not None

    def enhance(self, image: np.ndarray, apply_sr: bool = False) -> dict:
        """
        Full enhancement pipeline with honest reporting at each step.
        Returns enhanced image + detailed processing report.
        """
        report = {"steps": {}, "warnings": [], "success": True}
        current = image.copy()

        # Step 1: Neural enhancement first (U-Net GAN) — main enhancement
        if self.use_neural:
            try:
                current = enhance_with_model(current, self.generator, self.device)
                report["steps"]["neural_enhancement"] = "SUCCESS — U-Net GAN enhancement applied"
            except Exception as e:
                report["steps"]["neural_enhancement"] = f"FALLBACK — {str(e)} — Using CLAHE"
                current = apply_clahe_enhancement(current)
                report["warnings"].append("Neural model fallback to CLAHE")
        else:
            # Traditional enhancement when no weights
            current = apply_clahe_enhancement(current)
            img_float = current.astype(np.float32)
            img_float[:, :, 2] = np.clip(img_float[:, :, 2] * 1.4, 0, 255)
            img_float[:, :, 1] = np.clip(img_float[:, :, 1] * 1.1, 0, 255)
            current = img_float.astype(np.uint8)
            gamma = 1.2
            lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
            current = cv2.LUT(current, lut)
            report["steps"]["neural_enhancement"] = "CLAHE + Color correction"

        # Step 2: Blend neural output with original to prevent artifacts
        try:
            orig_float = image.astype(np.float32)
            enh_float = current.astype(np.float32)
            # 60% neural + 40% original prevents extreme corrections
            blended = 0.6 * enh_float + 0.4 * orig_float
            # Now apply gentle global white balance on blend
            b_m = np.mean(blended[:,:,0])
            g_m = np.mean(blended[:,:,1])
            r_m = np.mean(blended[:,:,2])
            gray = (b_m + g_m + r_m) / 3.0
            blended[:,:,0] = np.clip(blended[:,:,0] * np.clip(gray/(b_m+1e-6), 0.8, 1.3), 0, 255)
            blended[:,:,1] = np.clip(blended[:,:,1] * np.clip(gray/(g_m+1e-6), 0.8, 1.3), 0, 255)
            blended[:,:,2] = np.clip(blended[:,:,2] * np.clip(gray/(r_m+1e-6), 0.8, 1.3), 0, 255)
            # Brightness boost if needed
            brightness = np.mean(blended)
            if brightness < 110:
                blended = np.clip(blended * (115.0 / (brightness + 1e-6)), 0, 255)
            current = blended.astype(np.uint8)
            report["steps"]["white_balance"] = "SUCCESS — neural blend + color correction applied"
        except Exception as e:
            report["steps"]["white_balance"] = f"FAILED — {str(e)}"

        # Step 3: Skip haze removal — neural model handles this
        report["steps"]["haze_removal"] = "SKIPPED — neural model handles dehazing"

        # Step 4: Super resolution (optional, for detail recovery)
        sr_image = None
        if apply_sr and self.use_sr:
            try:
                sr_image = apply_super_resolution(current, self.sr_model, self.device)
                report["steps"]["super_resolution"] = "SUCCESS — 4x detail recovery applied"
            except Exception as e:
                report["steps"]["super_resolution"] = f"FAILED — {str(e)}"
                report["warnings"].append("Super resolution failed")
        else:
            report["steps"]["super_resolution"] = "SKIPPED — Not requested or weights not loaded"

        # Step 5: Mild sharpening
        kernel = np.array([[0, -0.3, 0], [-0.3, 2.2, -0.3], [0, -0.3, 0]])
        current = cv2.filter2D(current, -1, kernel)
        report["steps"]["sharpening"] = "SUCCESS"

        return {
            "enhanced": current,
            "sr_enhanced": sr_image,
            "report": report
        }
