import cv2
import numpy as np
from enum import Enum


class QualityLevel(Enum):
    UNRECOVERABLE = "unrecoverable"
    VERY_POOR = "very_poor"
    POOR = "poor"
    MODERATE = "moderate"
    GOOD = "good"
    HIGH = "high"


def compute_uiqm(image: np.ndarray) -> float:
    """
    Underwater Image Quality Measure (UIQM).
    Combines colorfulness, sharpness, and contrast.
    No reference image needed — works on real-world footage.
    """
    img = image.astype(np.float32) / 255.0
    # OpenCV is BGR: index 2=R, 1=G, 0=B
    r, g, b = img[:, :, 2], img[:, :, 1], img[:, :, 0]

    # UICM - Underwater Image Colorfulness Measure
    rg = r - g
    yb = 0.5 * (r + g) - b
    uicm = -0.0268 * np.sqrt(np.mean(rg**2) + np.mean(yb**2)) + \
            0.1586 * np.sqrt(np.std(rg)**2 + np.std(yb)**2)

    # UISM - Underwater Image Sharpness Measure
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    uism = np.mean(np.sqrt(sobelx**2 + sobely**2)) / 255.0

    # UIConM - Underwater Image Contrast Measure
    uiconm = np.std(gray.astype(np.float32) / 255.0)

    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    return float(np.clip(uiqm, 0, 1))


def compute_degradation_types(image: np.ndarray) -> dict:
    """
    Identify what types of degradation are present.
    Used for honest reporting of what enhancement can/cannot fix.
    """
    img_float = image.astype(np.float32) / 255.0
    b, g, r = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    degradations = {}

    # Color shift (blue-green dominance — typical underwater)
    r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
    color_shift_score = max(0, (g_mean + b_mean) / 2 - r_mean)
    degradations["color_shift"] = {
        "present": color_shift_score > 0.05,
        "severity": "HIGH" if color_shift_score > 0.15 else "MODERATE" if color_shift_score > 0.05 else "LOW",
        "score": round(float(color_shift_score), 3),
        "description": f"Red channel loss: R={r_mean:.2f} G={g_mean:.2f} B={b_mean:.2f}"
    }

    # Haze/backscatter — low contrast + washed out
    contrast = np.std(gray.astype(np.float32) / 255.0)
    degradations["haze"] = {
        "present": contrast < 0.15,
        "severity": "HIGH" if contrast < 0.08 else "MODERATE" if contrast < 0.15 else "LOW",
        "score": round(float(1 - contrast), 3),
        "description": f"Contrast score: {contrast:.3f}"
    }

    # Blur — low sharpness
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    degradations["blur"] = {
        "present": laplacian_var < 100,
        "severity": "HIGH" if laplacian_var < 20 else "MODERATE" if laplacian_var < 100 else "LOW",
        "score": round(float(max(0, 1 - laplacian_var / 500)), 3),
        "description": f"Sharpness variance: {laplacian_var:.1f}"
    }

    # Low light
    brightness = np.mean(gray)
    degradations["low_light"] = {
        "present": brightness < 80,
        "severity": "HIGH" if brightness < 40 else "MODERATE" if brightness < 80 else "LOW",
        "score": round(float(max(0, 1 - brightness / 128)), 3),
        "description": f"Mean brightness: {brightness:.1f}/255"
    }

    # Noise — high frequency variation
    noise_estimate = np.std(gray.astype(np.float32) - cv2.GaussianBlur(gray, (5, 5), 0))
    degradations["noise"] = {
        "present": noise_estimate > 10,
        "severity": "HIGH" if noise_estimate > 25 else "MODERATE" if noise_estimate > 10 else "LOW",
        "score": round(float(min(1, noise_estimate / 50)), 3),
        "description": f"Noise estimate: {noise_estimate:.2f}"
    }

    return degradations


def assess_quality(image: np.ndarray) -> dict:
    """
    Full quality assessment — determines if image is processable.
    This is the first honest gate before any AI processing.
    """
    uiqm = compute_uiqm(image)
    degradations = compute_degradation_types(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Determine quality level
    if uiqm < 0.10 and laplacian_var < 10:
        level = QualityLevel.UNRECOVERABLE
        can_process = False
        message = "Image quality too poor for reliable analysis. Enhancement cannot recover sufficient detail."
        recommendation = "Reposition camera, increase lighting, or use sonar as primary sensor."
    elif uiqm < 0.20:
        level = QualityLevel.VERY_POOR
        can_process = True
        message = "Very poor image quality. Enhancement will be applied but results may be unreliable."
        recommendation = "Results should be cross-verified with additional sensors."
    elif uiqm < 0.35:
        level = QualityLevel.POOR
        can_process = True
        message = "Poor image quality. Enhancement applied. Detection confidence will be reduced."
        recommendation = "Treat detections as preliminary — manual verification advised."
    elif uiqm < 0.55:
        level = QualityLevel.MODERATE
        can_process = True
        message = "Moderate image quality. Enhancement will significantly improve results."
        recommendation = "Detections are reasonably reliable."
    elif uiqm < 0.75:
        level = QualityLevel.GOOD
        can_process = True
        message = "Good image quality. High confidence detections expected."
        recommendation = "Proceed with standard operator review."
    else:
        level = QualityLevel.HIGH
        can_process = True
        message = "High image quality. Excellent detection conditions."
        recommendation = "High confidence results."

    return {
        "uiqm_score": round(uiqm, 4),
        "quality_level": level.value,
        "can_process": can_process,
        "message": message,
        "recommendation": recommendation,
        "degradations": degradations,
        "image_shape": image.shape,
        "active_degradations": [k for k, v in degradations.items() if v["present"]]
    }
