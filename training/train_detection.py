"""
YOLOv8 Detection Fine-tuning Script — Kaggle Ready
Fine-tunes YOLOv8n on underwater object detection datasets.
Uses RUOD/Brackish/DeepFish datasets available on Kaggle.
Estimated training time: 2-3 hours for 100 epochs on T4 GPU.
"""
import os
import sys
import yaml
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    os.system("pip install ultralytics -q")
    from ultralytics import YOLO


def create_dataset_yaml(
    train_path: str,
    val_path: str,
    output_path: str = "/kaggle/working/underwater_dataset.yaml"
) -> str:
    """
    Create YOLO format dataset config.
    Classes match our knowledge base detection categories.
    """
    dataset_config = {
        "path": "/kaggle/working",
        "train": train_path,
        "val": val_path,
        "nc": 11,
        "names": [
            "submarine",        # 0 — PRIMARY THREAT
            "mine",             # 1 — CRITICAL THREAT
            "diver",            # 2 — HIGH THREAT
            "uuv",              # 3 — HIGH THREAT (unmanned underwater vehicle)
            "drone_underwater", # 4 — MEDIUM THREAT
            "fish_school",      # 5 — NO THREAT
            "marine_mammal",    # 6 — NO THREAT
            "coral_structure",  # 7 — NO THREAT
            "debris",           # 8 — LOW THREAT
            "pipeline",         # 9 — LOW THREAT
            "invasive_species"  # 10 — INFO
        ]
    }

    with open(output_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"Dataset YAML created: {output_path}")
    print(f"Classes: {dataset_config['names']}")
    return output_path


def prepare_brackish_dataset(
    brackish_path: str,
    output_path: str = "/kaggle/working/underwater_data"
) -> tuple:
    """
    Prepare Brackish dataset (available on Kaggle) for YOLO training.
    Maps Brackish classes to our class schema.
    """
    # Brackish class mapping → our class IDs
    brackish_class_map = {
        "fish": 5,          # fish_school
        "crab": 8,          # debris (closest match for non-threat)
        "shrimp": 5,        # fish_school
        "jellyfish": 6,     # marine_mammal (closest)
        "starfish": 10,     # invasive_species
        "small_fish": 5,
    }

    train_dir = Path(output_path) / "images" / "train"
    val_dir = Path(output_path) / "images" / "val"
    train_label_dir = Path(output_path) / "labels" / "train"
    val_label_dir = Path(output_path) / "labels" / "val"

    for d in [train_dir, val_dir, train_label_dir, val_label_dir]:
        d.mkdir(parents=True, exist_ok=True)

    brackish_src = Path(brackish_path)
    all_images = list(brackish_src.rglob("*.jpg")) + list(brackish_src.rglob("*.png"))

    print(f"Found {len(all_images)} images in Brackish dataset")
    split_idx = int(len(all_images) * 0.85)
    train_imgs, val_imgs = all_images[:split_idx], all_images[split_idx:]

    def copy_and_remap(images, img_dir, lbl_dir):
        for img_path in images:
            shutil.copy2(str(img_path), str(img_dir / img_path.name))
            # Look for corresponding label
            label_path = img_path.with_suffix(".txt")
            if not label_path.exists():
                label_path = img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name

            if label_path.exists():
                with open(label_path, "r") as f:
                    lines = f.readlines()
                remapped = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        orig_class = int(parts[0])
                        new_class = brackish_class_map.get(str(orig_class), 5)
                        remapped.append(f"{new_class} {' '.join(parts[1:])}")
                with open(str(lbl_dir / img_path.with_suffix(".txt").name), "w") as f:
                    f.write("\n".join(remapped))

    copy_and_remap(train_imgs, train_dir, train_label_dir)
    copy_and_remap(val_imgs, val_dir, val_label_dir)

    return str(train_dir), str(val_dir)


def train_detection(config: dict):
    """Fine-tune YOLOv8n for underwater object detection"""
    print("Loading YOLOv8n base model...")
    model = YOLO("yolov8n.pt")

    print(f"Starting fine-tuning: {config['epochs']} epochs")
    print(f"Dataset: {config['data_yaml']}")

    results = model.train(
        data=config["data_yaml"],
        epochs=config["epochs"],
        imgsz=config["image_size"],
        batch=config["batch_size"],
        lr0=config["lr"],
        device=0 if config["use_gpu"] else "cpu",
        project="/kaggle/working/detection_training",
        name="underwater_yolo",
        save=True,
        save_period=10,         # Save every 10 epochs
        patience=20,            # Early stopping patience
        augment=True,           # Use built-in augmentation
        mosaic=1.0,             # Mosaic augmentation — good for small objects
        mixup=0.1,
        copy_paste=0.1,         # Good for detecting objects in cluttered scenes
        degrees=10.0,           # Rotation augmentation
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,              # Underwater color variation
        hsv_v=0.4,
        verbose=True,
        plots=True,
        val=True,
    )

    # Best model path
    best_model = "/kaggle/working/detection_training/underwater_yolo/weights/best.pt"
    print(f"\nTraining complete!")
    print(f"Best model: {best_model}")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    return best_model


def validate_model(model_path: str, data_yaml: str):
    """Run validation and print honest performance report"""
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, verbose=True)

    print("\n" + "="*60)
    print("DETECTION MODEL VALIDATION REPORT")
    print("="*60)
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:   {metrics.box.mr:.4f}")
    print("\nPer-class AP:")
    class_names = [
        "submarine", "mine", "diver", "uuv", "drone_underwater",
        "fish_school", "marine_mammal", "coral_structure",
        "debris", "pipeline", "invasive_species"
    ]
    for i, (name, ap) in enumerate(zip(class_names, metrics.box.ap50)):
        print(f"  {name:20s}: {ap:.4f}")
    print("="*60)

    return metrics


if __name__ == "__main__":
    import torch
    use_gpu = torch.cuda.is_available()
    print(f"GPU available: {use_gpu}")

    # Prepare dataset
    # On Kaggle, add the Brackish dataset: kaggle.com/datasets/aalborguniversity/brackish-dataset
    train_path, val_path = prepare_brackish_dataset(
        brackish_path="/kaggle/input/brackish-dataset",
        output_path="/kaggle/working/underwater_data"
    )

    # Create YAML config
    data_yaml = create_dataset_yaml(train_path, val_path)

    config = {
        "data_yaml": data_yaml,
        "epochs": 100,
        "batch_size": 16,       # Fits T4 GPU
        "image_size": 640,      # Standard YOLO input size
        "lr": 0.001,
        "use_gpu": use_gpu,
    }

    # Train
    best_model = train_detection(config)

    # Validate
    validate_model(best_model, data_yaml)

    print("\nDownload from Kaggle Output:")
    print(f"  {best_model}")
