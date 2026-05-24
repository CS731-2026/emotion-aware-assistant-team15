import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CustomImageDataset


def set_seed(seed: int = 42) -> None:
    """Make the training run more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def safe_arch_name(arch: str) -> str:
    return arch.replace("/", "_").replace(":", "_")


def _cuda_works() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.empty((1,), device="cuda")
        x = x + 1
        torch.cuda.synchronize()
        return True
    except Exception as e:
        print(f"[Warning] CUDA appears unusable: {e}")
        return False


def select_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not _cuda_works():
            raise RuntimeError("Requested --device=cuda but CUDA is not usable in this environment.")
        return torch.device("cuda")
    if device == "auto":
        return torch.device("cuda" if _cuda_works() else "cpu")
    raise ValueError(f"Unsupported device: {device}")



def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """

    If the server cannot download pretrained weights, it falls back to random
    initialization so that the pipeline can still run.
    """
    try:
        model = timm.create_model(
            arch,
            pretrained=pretrained,
            num_classes=num_classes,
        )
    except Exception as e:
        print(f"[Warning] Failed to load pretrained model: {e}")
        print("[Fallback] Loading model without pretrained weights.")
        model = timm.create_model(
            arch,
            pretrained=False,
            num_classes=num_classes,
        )
    return model


def compute_class_weights(labels, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced emotion classes.
    """
    counts = Counter(labels)
    total = sum(counts.values())

    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        weight = total / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32).to(device)


class Trainer:
    def __init__(
        self,
        arch: str,
        batch_size: int,
        lr: float,
        epochs: int,
        num_workers: int,
        data_root: str = "data/processed",
        save_dir: str = "checkpoints",
        pretrained: bool = True,
        seed: int = 42,
        device: str = "auto",
    ):
        set_seed(seed)

        self.project_root = Path(__file__).resolve().parents[1]
        self.emotion_dir = Path(__file__).resolve().parent
        self.data_root = Path(data_root)
        if not self.data_root.is_absolute():
            self.data_root = self.project_root / self.data_root

        self.train_dir = self.data_root / "train"
        self.val_dir = self.data_root / "val"
        self.test_dir = self.data_root / "test"

        self.arch = arch
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.num_workers = num_workers
        self.pretrained = pretrained
        self.seed = seed

        # Always save checkpoints under emotion_recognition/checkpoints,
        # even if this script is launched from the project root.
        self.save_dir = Path(save_dir)
        if not self.save_dir.is_absolute():
            self.save_dir = self.emotion_dir / self.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = select_device(device)
        print("Using device:", self.device)
        print("Project root:", self.project_root)
        print("Data root:", self.data_root)
        print("Checkpoint dir:", self.save_dir)

        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.03,
            ),
            transforms.ToTensor(),
            # ImageNet normalization works better when pretrained=True.
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        self.train_dataset = CustomImageDataset(self.train_dir, transform=self.train_transform)
        self.val_dataset = CustomImageDataset(self.val_dir, transform=self.eval_transform)
        self.test_dataset = CustomImageDataset(self.test_dir, transform=self.eval_transform)

        assert self.train_dataset.classes == self.val_dataset.classes == self.test_dataset.classes, (
            "Class order mismatch among train/val/test."
        )

        self.classes = self.train_dataset.classes
        self.num_classes = len(self.classes)

        self.class_map_path = self.save_dir / "class_map.json"
        with open(self.class_map_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "classes": self.classes,
                    "class_to_idx": self.train_dataset.class_to_idx,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
        )

        print("Building model:", self.arch)
        self.model = build_model(
            self.arch,
            self.num_classes,
            pretrained=self.pretrained,
        ).to(self.device)

        class_weights = compute_class_weights(
            self.train_dataset.labels,
            self.num_classes,
            self.device,
        )
        print("Class weights:", class_weights.detach().cpu().tolist())

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )

        self.log_path = self.save_dir / f"{safe_arch_name(self.arch)}_training_log.csv"
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr"])

    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for images, labels in tqdm(self.train_loader, desc="Training", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / max(total_samples, 1)

    def evaluate(self, loader: DataLoader, desc: str = "Evaluating") -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        correct = 0

        with torch.no_grad():
            for images, labels in tqdm(loader, desc=desc, leave=False):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                correct += (preds == labels).sum().item()

        avg_loss = total_loss / max(total_samples, 1)
        acc = 100.0 * correct / max(total_samples, 1)
        return avg_loss, acc

    def save_checkpoint(self, name: str, epoch: Optional[int] = None, val_acc: Optional[float] = None) -> None:
        path = self.save_dir / name
        payload: Dict[str, Any] = {
            "arch": self.arch,
            "num_classes": self.num_classes,
            "classes": self.classes,
            "class_to_idx": self.train_dataset.class_to_idx,
            "epoch": epoch,
            "val_acc": val_acc,
            "pretrained_requested": self.pretrained,
            "seed": self.seed,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        torch.save(payload, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: Path) -> None:
        """
        Load the new checkpoint format. Also supports old full-model checkpoints
        as a fallback for compatibility.
        """
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        except Exception as e:
            print(f"[Warning] weights_only=True load failed: {e}")
            print("[Warning] Falling back to weights_only=False. Only use trusted checkpoints.")
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, nn.Module):
            self.model = checkpoint.to(self.device)
            return

        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

        model_state = checkpoint.get("model_state_dict")
        if model_state is None:
            raise ValueError("Checkpoint dict missing 'model_state_dict'.")

        ckpt_arch = checkpoint.get("arch")
        ckpt_num_classes = checkpoint.get("num_classes")
        rebuild_model = False

        if ckpt_arch and ckpt_arch != self.arch:
            print(f"[Info] Rebuilding model from checkpoint arch: {ckpt_arch}")
            self.arch = ckpt_arch
            rebuild_model = True

        if ckpt_num_classes and ckpt_num_classes != self.num_classes:
            print(f"[Info] Rebuilding model with checkpoint num_classes: {ckpt_num_classes}")
            self.num_classes = ckpt_num_classes
            rebuild_model = True

        if rebuild_model:
            self.model = build_model(self.arch, self.num_classes, pretrained=False).to(self.device)

        self.model.load_state_dict(model_state)
        self.model.to(self.device)

    def run(self) -> None:
        best_val_acc = -1.0

        for epoch in range(1, self.epochs + 1):
            print(f"\n===== Epoch {epoch}/{self.epochs} =====")

            train_loss = self.train_one_epoch()
            val_loss, val_acc = self.evaluate(self.val_loader, desc="Validation")
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"Val Acc:    {val_acc:.2f}%")
            print(f"LR:         {current_lr:.6g}")

            epoch_ckpt = f"{safe_arch_name(self.arch)}_epoch_{epoch}.pt"
            self.save_checkpoint(epoch_ckpt, epoch=epoch, val_acc=val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint("best.pt", epoch=epoch, val_acc=val_acc)
                self.save_checkpoint(f"{safe_arch_name(self.arch)}_best.pt", epoch=epoch, val_acc=val_acc)
                print(f"New best validation accuracy: {best_val_acc:.2f}%")

            self.scheduler.step(val_acc)

            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, val_acc, current_lr])

        print("\n===== Final Test Evaluation with Best Model =====")
        best_model_path = self.save_dir / "best.pt"
        self.load_checkpoint(best_model_path)

        test_loss, test_acc = self.evaluate(self.test_loader, desc="Test")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Acc:  {test_acc:.2f}%")
        print(f"Best checkpoint saved at: {best_model_path}")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        type=str,
        default="convnextv2_pico.fcmae_ft_in1k",
        help="timm model name",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--data-root", default="data/processed", help="dataset root containing train/val/test folders")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained weights. Useful when the server cannot download weights.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(
        data_root=args.data_root,
        arch=args.arch,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        pretrained=not args.no_pretrained,
        seed=args.seed,
        device=args.device,
    )
    trainer.run()
