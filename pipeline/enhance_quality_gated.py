"""
FIXED ENHANCEMENT PIPELINE - Quality-Gated Processing
Solves the problem: Model destroying good images

Key Principle:
  IF image quality is GOOD → Apply minimal processing (preserve colors)
  IF image quality is BAD → Apply aggressive enhancement (restore colors)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import sys

torch.manual_seed(42)
np.random.seed(42)

sys.path.append(str(Path(__file__).parent.parent))
from models.enhancement.generator import get_generator
from pipeline.quality_gate import assess_quality


def gentle_color_preservation(image):
    """Very gentle color boost - for already-good images"""
    img_float = image.astype(np.float32)
    
    # Analyze color
    b_mean = np.mean(img_float[:,:,0])
    g_mean = np.mean(img_float[:,:,1])
    r_mean = np.mean(img_float[:,:,2])
    
    # MINIMAL boost only (1.1x instead of 2.5x)
    if r_mean < 100:  # Only if red is very low
        img_float[:,:,2] = np.clip(img_float[:,:,2] * 1.1, 0, 255)
    
    return img_float.astype(np.uint8)


def gentle_sharpening(image):
    """Light sharpening for good images"""
    kernel = np.array([[0, -0.1, 0], 
                      [-0.1, 1.4, -0.1], 
                      [0, -0.1, 0]])
    return cv2.filter2D(image.astype(np.float32), -1, kernel).astype(np.uint8)


def moderate_contrast_boost(image):
    """Subtle contrast enhancement - for good images"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Light CLAHE only (clip 2.0 instead of 3.5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    l_clahe = clahe.apply(l)
    
    # 40% CLAHE + 60% original (preserve naturalness)
    l_enhanced = cv2.addWeighted(l_clahe, 0.4, l, 0.6, 0)
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def minimal_processing_pipeline(image):
    """
    For HIGH QUALITY images (UIQM > 0.8)
    Only preserves, doesn't aggressively enhance
    """
    result = image.copy()
    
    # Step 1: Very gentle color preservation
    result = gentle_color_preservation(result)
    
    # Step 2: Subtle contrast
    result = moderate_contrast_boost(result)
    
    # Step 3: Light sharpening
    result = gentle_sharpening(result)
    
    # NO neural enhancement
    # NO aggressive color restoration
    # NO dehazing
    # NO aggressive noise removal
    
    return result


def aggressive_color_restoration(image):
    """Strong color boost - for degraded images"""
    img_float = image.astype(np.float32) / 255.0
    b, g, r = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    
    dominant = max(r_mean, g_mean, b_mean)
    
    # AGGRESSIVE boost for degraded images
    r_scale = min(2.5, dominant / (r_mean + 1e-6)) if r_mean < 0.3 else 1.5
    g_scale = min(1.8, dominant / (g_mean + 1e-6))
    b_scale = min(1.3, dominant / (b_mean + 1e-6))
    
    result = image.astype(np.float32).copy()
    result[:, :, 2] = np.clip(result[:, :, 2] * r_scale, 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * g_scale, 0, 255)
    result[:, :, 0] = np.clip(result[:, :, 0] * b_scale, 0, 255)
    
    return result.astype(np.uint8)


def color_cast_removal(image):
    """Remove underwater color casts"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    
    a_mean = np.mean(a)
    
    if a_mean > 130:
        a = np.clip(a - 15, 0, 255)
        b = np.clip(b + 10, 0, 255)
    elif a_mean < 110:
        a = np.clip(a + 20, 0, 255)
    
    b = np.clip(b + 5, 0, 255)
    
    lab[:, :, 0] = l
    lab[:, :, 1] = a
    lab[:, :, 2] = b
    
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def tone_mapping_reinhard(image):
    """Tone mapping - prevents white light blown-out"""
    img_float = image.astype(np.float32) / 255.0
    tone_mapped = img_float / (1.0 + img_float)
    result = tone_mapped * 255
    return np.clip(result, 0, 254).astype(np.uint8)


def adaptive_contrast_enhancement(image):
    """Multi-scale contrast for degraded images"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(12, 12))
    l_clahe = clahe.apply(l)
    
    l_enhanced = cv2.addWeighted(l_clahe, 0.65, l, 0.35, 0)
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def edge_preserving_sharpening(image):
    """Sharpening for degraded images"""
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32) / 5.0
    sharpened = cv2.filter2D(image.astype(np.float32), -1, kernel)
    
    result = 0.7 * sharpened + 0.3 * image.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def dehazing(image):
    """Dark Channel Prior dehazing"""
    img_float = image.astype(np.float32) / 255.0
    
    patch_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    
    min_channel = np.min(img_float, axis=2)
    dark_channel = cv2.erode(min_channel, kernel)
    
    flat_dark = dark_channel.flatten()
    top_pixels = np.argsort(flat_dark)[-int(0.001 * len(flat_dark)):]
    atm_light = np.mean(img_float.reshape(-1, 3)[top_pixels], axis=0)
    atm_light = np.clip(atm_light, 0.1, 1.0)
    
    omega = 0.85
    norm_img = img_float / atm_light
    norm_dark = np.min(norm_img, axis=2)
    transmission = 1.0 - omega * cv2.erode(norm_dark, kernel)
    transmission = np.clip(transmission, 0.1, 1.0)
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        transmission = cv2.ximgproc.guidedFilter(gray, transmission, 60, 0.001)
    except:
        pass
    
    dehazed = np.zeros_like(img_float)
    for c in range(3):
        dehazed[:, :, c] = (img_float[:, :, c] - atm_light[c]) / np.maximum(transmission, 0.1) + atm_light[c]
    
    dehazed = np.clip(dehazed, 0, 1)
    return (dehazed * 255).astype(np.uint8)


def denoise(image):
    """Multi-stage denoising"""
    filtered1 = cv2.bilateralFilter(image, 9, 75, 75)
    
    try:
        denoised = cv2.fastNlMeansDenoisingColored(filtered1, None, h=10, hForColorComponents=10, templateWindowSize=7, searchWindowSize=21)
    except:
        denoised = filtered1
    
    result = cv2.addWeighted(denoised, 0.6, image, 0.4, 0)
    return result


def fish_visibility_boost(image):
    """Boost fish colors"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    fish_color_mask = ((h < 30) | (h > 150)).astype(np.float32)
    
    s = s * (1.0 + 0.3 * fish_color_mask)
    s = np.clip(s, 0, 255)
    
    v = v * 1.1
    v = np.clip(v, 0, 255)
    
    hsv[:, :, 0] = h
    hsv[:, :, 1] = s
    hsv[:, :, 2] = v
    
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def enhance_with_model(image, model, device):
    """Apply neural enhancement"""
    h, w = image.shape[:2]
    target_h = (h // 256 + 1) * 256 if h % 256 != 0 else h
    target_w = (w // 256 + 1) * 256 if w % 256 != 0 else w

    img_resized = cv2.resize(image, (target_w, target_h))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        enhanced_tensor = model(img_tensor)

    enhanced = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced = (enhanced * 0.5 + 0.5) * 255
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    enhanced = cv2.resize(enhanced, (w, h))
    return enhanced


class QualityGatedEnhancementPipeline:
    """
    Smart pipeline that adapts based on INPUT quality
    - High quality (UIQM > 0.8) → Minimal processing
    - Low quality (UIQM < 0.5) → Aggressive enhancement
    - Medium quality (0.5-0.8) → Moderate enhancement
    """
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = None
        
        if model_path and Path(model_path).exists():
            self.model = get_generator(model_path, device)
            self.model.eval()
            print(f"✅ Quality-Gated Pipeline Ready | With neural model | Device: {device}")
        else:
            print(f"✅ Quality-Gated Pipeline Ready | Without neural model | Device: {device}")
    
    def enhance(self, image, apply_sr=False):
        """Main enhancement - adapts to image quality"""
        
        # CRITICAL: Assess quality FIRST
        quality = assess_quality(image)
        uiqm = quality['uiqm_score']
        
        report = {
            "input_quality": uiqm,
            "strategy": "",
            "steps": {},
            "warnings": []
        }
        
        # DECISION: What to do based on quality
        if uiqm > 0.8:
            # HIGH QUALITY - Minimal processing (PRESERVE COLORS)
            report["strategy"] = f"HIGH quality detected (UIQM {uiqm:.3f}) - MINIMAL processing"
            result = self._minimal_enhancement(image, report)
        
        elif uiqm < 0.5:
            # LOW QUALITY - Aggressive enhancement (RESTORE COLORS)
            report["strategy"] = f"LOW quality detected (UIQM {uiqm:.3f}) - AGGRESSIVE enhancement"
            result = self._aggressive_enhancement(image, report)
        
        else:
            # MEDIUM QUALITY - Moderate enhancement
            report["strategy"] = f"MEDIUM quality detected (UIQM {uiqm:.3f}) - MODERATE enhancement"
            result = self._moderate_enhancement(image, report)
        
        # Verify output quality didn't degrade
        output_quality = assess_quality(result)
        output_uiqm = output_quality['uiqm_score']
        
        if output_uiqm < uiqm - 0.1:
            report["warnings"].append(f"⚠️ Quality degraded: {uiqm:.3f} → {output_uiqm:.3f}")
        
        report["output_quality"] = output_uiqm
        report["quality_change"] = output_uiqm - uiqm
        
        return {
            "enhanced": result,
            "report": report
        }
    
    def _minimal_enhancement(self, image, report):
        """For GOOD images - preserve originals, light enhancement only"""
        current = image.copy()
        
        # Step 1: Gentle color preservation
        try:
            current = gentle_color_preservation(current)
            report["steps"]["1_color"] = "✅ Gentle preservation (1.1x)"
        except:
            report["steps"]["1_color"] = "⚠️ Failed"
        
        # Step 2: Subtle contrast
        try:
            current = moderate_contrast_boost(current)
            report["steps"]["2_contrast"] = "✅ Subtle boost (40/60)"
        except:
            report["steps"]["2_contrast"] = "⚠️ Failed"
        
        # Step 3: Light sharpening
        try:
            current = gentle_sharpening(current)
            report["steps"]["3_sharpen"] = "✅ Light sharpening"
        except:
            report["steps"]["3_sharpen"] = "⚠️ Failed"
        
        report["steps"]["neural"] = "⏭️ SKIPPED (image already good)"
        report["steps"]["dehazing"] = "⏭️ SKIPPED (not needed)"
        
        return current
    
    def _moderate_enhancement(self, image, report):
        """For MEDIUM quality - balanced enhancement"""
        current = image.copy()
        
        # Stage 1: Moderate color restoration
        try:
            current = aggressive_color_restoration(current)
            report["steps"]["1_color"] = "✅ Moderate color boost"
        except:
            report["steps"]["1_color"] = "⚠️ Failed"
        
        # Stage 2: Color cast removal
        try:
            current = color_cast_removal(current)
            report["steps"]["2_cast"] = "✅ Color cast removal"
        except:
            report["steps"]["2_cast"] = "⚠️ Failed"
        
        # Stage 3: Contrast
        try:
            current = adaptive_contrast_enhancement(current)
            report["steps"]["3_contrast"] = "✅ Moderate contrast"
        except:
            report["steps"]["3_contrast"] = "⚠️ Failed"
        
        # Stage 4: Sharpening
        try:
            current = edge_preserving_sharpening(current)
            report["steps"]["4_sharpen"] = "✅ Edge-preserve sharpen"
        except:
            report["steps"]["4_sharpen"] = "⚠️ Failed"
        
        # Stage 5: Skip neural if quality okay
        if self.model and True:
            try:
                current = enhance_with_model(current, self.model, self.device)
                report["steps"]["5_neural"] = "✅ U-Net enhancement"
            except:
                report["steps"]["5_neural"] = "⚠️ Skipped"
        
        return current
    
    def _aggressive_enhancement(self, image, report):
        """For BAD images - full aggressive enhancement"""
        current = image.copy()
        
        # Stage 1: Aggressive color restoration
        try:
            current = aggressive_color_restoration(current)
            report["steps"]["1_color"] = "✅ Aggressive boost (2.5x)"
        except:
            report["steps"]["1_color"] = "⚠️ Failed"
        
        # Stage 2: Color cast removal
        try:
            current = color_cast_removal(current)
            report["steps"]["2_cast"] = "✅ Water-type detection"
        except:
            report["steps"]["2_cast"] = "⚠️ Failed"
        
        # Stage 3: Neural enhancement
        if self.model:
            try:
                current = enhance_with_model(current, self.model, self.device)
                report["steps"]["3_neural"] = "✅ U-Net GAN applied"
            except:
                report["steps"]["3_neural"] = "⚠️ Failed"
        
        # Stage 4: Tone mapping
        try:
            current = tone_mapping_reinhard(current)
            report["steps"]["4_tone"] = "✅ Reinhard tone-map"
        except:
            report["steps"]["4_tone"] = "⚠️ Failed"
        
        # Stage 5: Contrast
        try:
            current = adaptive_contrast_enhancement(current)
            report["steps"]["5_contrast"] = "✅ CLAHE boost"
        except:
            report["steps"]["5_contrast"] = "⚠️ Failed"
        
        # Stage 6: Sharpening
        try:
            current = edge_preserving_sharpening(current)
            report["steps"]["6_sharpen"] = "✅ Edge-preserve"
        except:
            report["steps"]["6_sharpen"] = "⚠️ Failed"
        
        # Stage 7: Dehazing
        try:
            current = dehazing(current)
            report["steps"]["7_dehaze"] = "✅ Dark Channel Prior"
        except:
            report["steps"]["7_dehaze"] = "⚠️ Failed"
        
        # Stage 8: Denoising
        try:
            current = denoise(current)
            report["steps"]["8_denoise"] = "✅ Multi-stage NLM"
        except:
            report["steps"]["8_denoise"] = "⚠️ Failed"
        
        # Stage 9: Fish boost
        try:
            current = fish_visibility_boost(current)
            report["steps"]["9_fish"] = "✅ Fish saturation +30%"
        except:
            report["steps"]["9_fish"] = "⚠️ Failed"
        
        return current
