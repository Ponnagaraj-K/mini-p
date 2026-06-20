import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pipeline.quality_gate import compute_uiqm


def compute_psnr(original: np.ndarray, enhanced: np.ndarray) -> float:
    """PSNR — requires reference image. Only valid on paired test data."""
    try:
        orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        enh_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        if orig_rgb.shape != enh_rgb.shape:
            enh_rgb = cv2.resize(enh_rgb, (orig_rgb.shape[1], orig_rgb.shape[0]))
        return round(float(psnr(orig_rgb, enh_rgb, data_range=255)), 4)
    except Exception:
        return None


def compute_ssim(original: np.ndarray, enhanced: np.ndarray) -> float:
    """SSIM — requires reference image. Measures structural similarity."""
    try:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        if orig_gray.shape != enh_gray.shape:
            enh_gray = cv2.resize(enh_gray, (orig_gray.shape[1], orig_gray.shape[0]))
        score, _ = ssim(orig_gray, enh_gray, full=True)
        return round(float(score), 4)
    except Exception:
        return None


def compute_uciqe(image: np.ndarray) -> float:
    """
    Underwater Color Image Quality Evaluation (UCIQE).
    No-reference metric — complementary to UIQM.
    """
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]
    chroma = np.sqrt(a**2 + b**2)
    sigma_c = np.std(chroma)
    con_l = np.std(l)
    mu_s = np.mean(chroma) / (np.mean(l) + 1e-6)
    uciqe = 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s
    return round(float(np.clip(uciqe / 50, 0, 1)), 4)


def compute_all_metrics(
    original: np.ndarray,
    enhanced: np.ndarray,
    reference: np.ndarray = None
) -> dict:
    """
    Compute all quality metrics.
    - UIQM/UCIQE: always available (no-reference)
    - PSNR/SSIM: only when reference clean image is available
    """
    metrics = {
        "uiqm_before": compute_uiqm(original),
        "uiqm_after": compute_uiqm(enhanced),
        "uciqe_before": compute_uciqe(original),
        "uciqe_after": compute_uciqe(enhanced),
        "uiqm_improvement": None,
        "uciqe_improvement": None,
        "psnr": None,
        "ssim": None,
        "reference_available": reference is not None
    }

    metrics["uiqm_improvement"] = round(
        metrics["uiqm_after"] - metrics["uiqm_before"], 4
    )
    metrics["uciqe_improvement"] = round(
        metrics["uciqe_after"] - metrics["uciqe_before"], 4
    )

    if reference is not None:
        metrics["psnr"] = compute_psnr(reference, enhanced)
        metrics["ssim"] = compute_ssim(reference, enhanced)
        metrics["note"] = "PSNR/SSIM computed against clean reference image"
    else:
        metrics["note"] = (
            "PSNR/SSIM unavailable — no clean reference image. "
            "UIQM/UCIQE are primary metrics for real-world footage."
        )

    # Enhancement effectiveness
    uiqm_gain = metrics["uiqm_improvement"]
    metrics["enhancement_verdict"] = (
        "EXCELLENT improvement" if uiqm_gain > 0.25 else
        "GOOD improvement" if uiqm_gain > 0.12 else
        "MODERATE improvement" if uiqm_gain > 0.05 else
        "MINIMAL improvement — image may be heavily degraded" if uiqm_gain > 0 else
        "NO improvement — check enhancement pipeline"
    )

    return metrics
