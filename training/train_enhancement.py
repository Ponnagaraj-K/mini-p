"""
Enhancement Model Training Script — Kaggle Ready
Trains FUnIE-GAN style U-Net + PatchGAN on EUVP/UIEB + synthetic IOD dataset.
Paste this directly into a Kaggle notebook with GPU T4 enabled.
Estimated training time: 4-6 hours for 100 epochs on T4 GPU.
"""
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path (adjust for Kaggle)
sys.path.append("/kaggle/working/underwater-enhancement")

from models.enhancement.generator import UNetGenerator
from models.enhancement.discriminator import PatchGANDiscriminator
from models.enhancement.losses import EnhancementLoss


class UnderwaterPairedDataset(Dataset):
    """Paired dataset: degraded underwater image + clean reference"""
    def __init__(self, degraded_dir: str, clean_dir: str, image_size: int = 256):
        self.degraded_dir = Path(degraded_dir)
        self.clean_dir = Path(clean_dir)
        self.image_size = image_size
        extensions = [".jpg", ".jpeg", ".png"]
        self.files = sorted([
            f.name for f in self.degraded_dir.iterdir()
            if f.suffix.lower() in extensions
        ])
        print(f"Dataset loaded: {len(self.files)} pairs from {degraded_dir}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        degraded = cv2.imread(str(self.degraded_dir / fname))
        clean_path = self.clean_dir / fname
        if not clean_path.exists():
            # Try matching by stem
            stem = Path(fname).stem.rsplit("_", 2)[0]
            matches = list(self.clean_dir.glob(f"{stem}*"))
            clean_path = matches[0] if matches else self.clean_dir / fname

        clean = cv2.imread(str(clean_path))

        if degraded is None or clean is None:
            # Return zeros if file is missing
            zero = torch.zeros(3, self.image_size, self.image_size)
            return zero, zero

        degraded = cv2.resize(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB),
                              (self.image_size, self.image_size))
        clean = cv2.resize(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB),
                           (self.image_size, self.image_size))

        # Random horizontal flip augmentation
        if np.random.random() > 0.5:
            degraded = np.fliplr(degraded).copy()
            clean = np.fliplr(clean).copy()

        return self.transform(degraded), self.transform(clean)


def train(config: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Models
    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)
    criterion = EnhancementLoss().to(device)

    # Optimizers
    opt_g = optim.Adam(generator.parameters(), lr=config["lr"], betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=config["lr"] * 0.5, betas=(0.5, 0.999))

    # LR schedulers
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=config["epochs"])
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=config["epochs"])

    # Dataset
    dataset = UnderwaterPairedDataset(
        config["degraded_dir"], config["clean_dir"], config["image_size"]
    )
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True
    )

    # Resume from checkpoint
    start_epoch = 0
    best_g_loss = float("inf")
    if config.get("resume_from") and Path(config["resume_from"]).exists():
        ckpt = torch.load(config["resume_from"], map_location=device)
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt["epoch"] + 1
        best_g_loss = ckpt.get("best_g_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    history = {"g_loss": [], "d_loss": [], "epoch": []}
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training: {config['epochs']} epochs | "
          f"{len(dataset)} samples | batch size {config['batch_size']}")

    for epoch in range(start_epoch, config["epochs"]):
        generator.train()
        discriminator.train()
        epoch_g_loss, epoch_d_loss = 0.0, 0.0

        for batch_idx, (degraded, clean) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            degraded, clean = degraded.to(device), clean.to(device)

            # Train Discriminator
            enhanced = generator(degraded).detach()
            disc_real = discriminator(degraded, clean)
            disc_fake = discriminator(degraded, enhanced)
            d_loss = criterion.discriminator_loss(disc_real, disc_fake)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # Train Generator
            enhanced = generator(degraded)
            disc_fake = discriminator(degraded, enhanced)
            g_loss, loss_components = criterion.generator_loss(disc_fake, enhanced, clean)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        # Epoch stats
        avg_g = epoch_g_loss / len(dataloader)
        avg_d = epoch_d_loss / len(dataloader)
        scheduler_g.step()
        scheduler_d.step()

        history["g_loss"].append(avg_g)
        history["d_loss"].append(avg_d)
        history["epoch"].append(epoch + 1)

        print(f"Epoch [{epoch+1}/{config['epochs']}] "
              f"G_loss: {avg_g:.4f} | D_loss: {avg_d:.4f} | "
              f"LR: {scheduler_g.get_last_lr()[0]:.6f}")

        # Save checkpoint every N epochs
        if (epoch + 1) % config["save_every"] == 0:
            ckpt_path = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "best_g_loss": best_g_loss,
                "config": config
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        # Save best model
        if avg_g < best_g_loss:
            best_g_loss = avg_g
            torch.save(generator.state_dict(), save_dir / "best_generator.pt")
            print(f"Best model saved (G_loss: {best_g_loss:.4f})")

    # Save final model
    torch.save(generator.state_dict(), save_dir / "final_generator.pt")
    print(f"\nTraining complete. Best G_loss: {best_g_loss:.4f}")
    print(f"Models saved to: {save_dir}")

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["g_loss"], label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["d_loss"], label="Discriminator Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Discriminator Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(save_dir / "training_curves.png"))
    print("Training curves saved.")

    return str(save_dir / "best_generator.pt")


if __name__ == "__main__":
    config = {
        # Kaggle paths — adjust based on your dataset setup
        "degraded_dir": "/kaggle/working/IOD-Syn/degraded",
        "clean_dir": "/kaggle/working/IOD-Syn/clean",
        "save_dir": "/kaggle/working/models",
        "resume_from": None,        # Set path to resume training

        # Training hyperparameters
        "epochs": 100,
        "batch_size": 8,            # Fits T4 16GB GPU
        "lr": 0.0002,
        "image_size": 256,
        "save_every": 10,           # Save checkpoint every N epochs
    }

    # Step 1: Generate synthetic dataset if not already done
    if not Path(config["degraded_dir"]).exists():
        print("Generating synthetic dataset first...")
        sys.path.append("/kaggle/working/underwater-enhancement")
        from training.synthetic_data import generate_dataset
        generate_dataset(
            clean_images_dir="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_imagenet/trainB",
            output_dir="/kaggle/working/IOD-Syn",
            target_size=(256, 256),
            augment_per_image=5
        )

    # Step 2: Train
    best_model_path = train(config)
    print(f"\nBest model: {best_model_path}")
    print("Download this file from Kaggle Output to use in your dashboard.")
