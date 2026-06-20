"""
Synthetic Indian Ocean Underwater Dataset Generator
Applies physics-based degradation to clear images to create paired training data.
Based on real Indian Ocean optical properties (Bay of Bengal / Arabian Sea).
Run this on Kaggle to generate training pairs from EUVP/UIEB clean images.
"""
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm


# Indian Ocean optical parameters — derived from MODIS ocean color data
DEGRADATION_PROFILES = {
    "bay_of_bengal_clear": {
        "absorption": [0.05, 0.12, 0.35],   # BGR
        "scattering": 0.12,
        "turbidity": 0.15,
        "depth_range": (2, 8),
        "noise_std": 5
    },
    "bay_of_bengal_turbid": {
        "absorption": [0.06, 0.18, 0.45],
        "scattering": 0.28,
        "turbidity": 0.35,
        "depth_range": (1, 5),
        "noise_std": 12
    },
    "arabian_sea_clear": {
        "absorption": [0.04, 0.09, 0.28],
        "scattering": 0.09,
        "turbidity": 0.10,
        "depth_range": (3, 12),
        "noise_std": 4
    },
    "arabian_sea_deep": {
        "absorption": [0.04, 0.10, 0.32],
        "scattering": 0.11,
        "turbidity": 0.20,
        "depth_range": (8, 20),
        "noise_std": 8
    },
    "coastal_murky": {
        "absorption": [0.08, 0.20, 0.50],
        "scattering": 0.40,
        "turbidity": 0.55,
        "depth_range": (0.5, 3),
        "noise_std": 18
    }
}


def apply_beer_lambert_absorption(image: np.ndarray, params: dict) -> np.ndarray:
    """Simulate wavelength-dependent light absorption"""
    img = image.astype(np.float32) / 255.0
    depth = np.random.uniform(*params["depth_range"])
    absorption = params["absorption"]

    for c in range(3):
        img[:, :, c] *= np.exp(-absorption[c] * depth)

    return np.clip(img * 255, 0, 255).astype(np.uint8)


def apply_backscattering(image: np.ndarray, scattering: float, turbidity: float) -> np.ndarray:
    """Simulate light backscattering — creates haze/veil effect"""
    img = image.astype(np.float32) / 255.0
    h, w = img.shape[:2]

    # Atmospheric/water light (blueish for underwater)
    water_light = np.array([0.7, 0.85, 1.0], dtype=np.float32)

    # Transmission map — decreases with distance/turbidity
    transmission = np.random.uniform(1 - turbidity - 0.1, 1 - turbidity + 0.1)
    transmission = np.clip(transmission, 0.3, 0.95)

    # Spatially varying transmission
    grad_h = np.linspace(transmission, transmission * 0.85, h)
    t_map = np.tile(grad_h[:, np.newaxis], (1, w))

    hazy = np.zeros_like(img)
    for c in range(3):
        hazy[:, :, c] = img[:, :, c] * t_map + water_light[c] * scattering * (1 - t_map)

    return np.clip(hazy * 255, 0, 255).astype(np.uint8)


def apply_color_cast(image: np.ndarray, depth: float = 5.0) -> np.ndarray:
    """Apply blue-green color cast typical of underwater"""
    img = image.astype(np.float32)
    # Reduce red channel, slightly reduce green, keep blue
    img[:, :, 2] *= max(0.3, 1.0 - depth * 0.06)   # Red (BGR channel 2)
    img[:, :, 1] *= max(0.7, 1.0 - depth * 0.02)   # Green
    img[:, :, 0] *= max(0.85, 1.0 - depth * 0.005) # Blue
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_noise(image: np.ndarray, noise_std: float) -> np.ndarray:
    """Add realistic underwater sensor noise"""
    noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_blur(image: np.ndarray, turbidity: float) -> np.ndarray:
    """Apply motion/scattering blur based on turbidity"""
    if turbidity > 0.3:
        kernel_size = int(turbidity * 5) * 2 + 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), turbidity * 2)
    return image


def degrade_image(image: np.ndarray, profile_name: str) -> np.ndarray:
    """Apply full degradation pipeline using a specific water profile"""
    params = DEGRADATION_PROFILES[profile_name]
    degraded = apply_beer_lambert_absorption(image, params)
    degraded = apply_color_cast(degraded, np.random.uniform(*params["depth_range"]))
    degraded = apply_backscattering(degraded, params["scattering"], params["turbidity"])
    degraded = apply_blur(degraded, params["turbidity"])
    degraded = apply_noise(degraded, params["noise_std"])
    return degraded


def generate_dataset(
    clean_images_dir: str,
    output_dir: str,
    target_size: tuple = (256, 256),
    augment_per_image: int = 3
):
    """
    Generate paired dataset from clean images.
    Each clean image → N degraded versions using different water profiles.
    
    Args:
        clean_images_dir: Path to folder with clean/reference images
        output_dir: Output folder
        target_size: Resize all images to this size
        augment_per_image: Number of degraded versions per clean image
    """
    clean_dir = Path(clean_images_dir)
    out_dir = Path(output_dir)
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)
    (out_dir / "degraded").mkdir(parents=True, exist_ok=True)

    profiles = list(DEGRADATION_PROFILES.keys())
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = [
        f for f in clean_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} clean images")
    print(f"Generating {len(image_files) * augment_per_image} paired samples...")

    count = 0
    for img_path in tqdm(image_files, desc="Generating dataset"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, target_size)

        for i in range(augment_per_image):
            profile = profiles[i % len(profiles)]
            degraded = degrade_image(img, profile)
            suffix = f"_{i:02d}_{profile}"
            base_name = img_path.stem + suffix

            cv2.imwrite(str(out_dir / "clean" / f"{base_name}.jpg"), img)
            cv2.imwrite(str(out_dir / "degraded" / f"{base_name}.jpg"), degraded)
            count += 1

    print(f"Dataset generated: {count} pairs saved to {output_dir}")
    print(f"Clean images: {out_dir / 'clean'}")
    print(f"Degraded images: {out_dir / 'degraded'}")
    return count


if __name__ == "__main__":
    # Kaggle usage:
    # clean images from EUVP/UIEB datasets already on Kaggle
    generate_dataset(
        clean_images_dir="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_imagenet/trainB",
        output_dir="/kaggle/working/IOD-Syn",
        target_size=(256, 256),
        augment_per_image=5
    )
