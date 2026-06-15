"""
Task 2 - Simple CNN ball-counting script
Computer Vision Project - 8-ball pool dataset

This script does everything in one run:
1. Reads a Roboflow YOLO dataset with train/valid/test folders.
2. Counts the number of balls in each image from the YOLO label file.
3. Trains these CNN architectures:
   - ResNet18
   - EfficientNet-B0
   - MobileNetV3-Large
   - DenseNet121
4. Trains each architecture twice:
   - with ImageNet pretrained weights
   - from scratch
5. Saves one .pth checkpoint per model.
6. Tests every saved model on the test split.
7. Writes all JSON outputs inside OUTPUT_DIR.

Expected dataset structure:

DATASET_ROOT/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
  data.yaml

Each YOLO label file is expected to have this format per line:

    class_id x_center y_center width height

The coordinates are normalized YOLO values. For this specific Roboflow
version, the uploaded example shows that class 2 is NOT a ball annotation;
it marks many small table/rail points. Therefore the target ball count is
computed by counting only the class ids listed in BALL_CLASS_IDS.
"""

# ============================================================
# EDIT ONLY THESE VARIABLES
# ============================================================

DATASET_ROOT = "./8-ball-pool-dataset"   # folder containing train/valid/test
OUTPUT_DIR = "./task2_output"            # all .pth and .json files go here

EPOCHS = 20
BATCH_SIZE = 16
IMAGE_SIZE = 224
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
SEED = 42
MAX_BALLS = 16

# Label format control.
# Your uploaded label example contains 26 rows, but only 8 are balls:
# ball classes: 0, 1, 3, 4
# ignored class: 2
# If your data.yaml says different class ids, edit only this variable.
BALL_CLASS_IDS = {0, 1, 3, 4}

# If BALL_CLASS_IDS is None, the script will count every valid YOLO row.
# For your current dataset format, keep BALL_CLASS_IDS = {0, 1, 3, 4}.

# If True, pretrained models train only the final regression layer.
# If False, the whole pretrained model is fine-tuned.
FREEZE_PRETRAINED_BACKBONE = False

# ============================================================
# SCRIPT STARTS HERE
# ============================================================

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class BallCountDataset(Dataset):
    def __init__(self, dataset_root, split, transform=None):
        self.dataset_root = Path(dataset_root)
        self.split_dir = self.find_split_dir(split)
        self.images_dir = self.split_dir / "images"
        self.labels_dir = self.split_dir / "labels"
        self.transform = transform

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing images folder: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Missing labels folder: {self.labels_dir}")

        self.image_paths = sorted(
            p for p in self.images_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in: {self.images_dir}")

    def find_split_dir(self, split):
        if split == "valid":
            possible_names = ["valid", "val", "validation"]
        else:
            possible_names = [split]

        for name in possible_names:
            split_dir = self.dataset_root / name
            if split_dir.exists():
                return split_dir

        raise FileNotFoundError(
            f"Could not find split '{split}' inside {self.dataset_root}"
        )

    def __len__(self):
        return len(self.image_paths)

    def count_balls(self, image_path):
        """
        Counts balls from a YOLO label file.

        YOLO row format:
            class_id x_center y_center width height

        In the example you uploaded, class 2 appears many times around the
        table/rail and should not be counted as a ball. Therefore, by default,
        this function counts only classes in BALL_CLASS_IDS = {0, 1, 3, 4}.
        """
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            return 0

        ball_count = 0

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue

                if BALL_CLASS_IDS is None:
                    ball_count += 1
                elif class_id in BALL_CLASS_IDS:
                    ball_count += 1

        return ball_count

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        count = self.count_balls(image_path)

        if self.transform is not None:
            image = self.transform(image)

        count = torch.tensor([float(count)], dtype=torch.float32)
        return image, count, str(image_path)


# ------------------------------------------------------------
# Image transforms
# ------------------------------------------------------------

def get_transforms(train):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

def create_model(architecture, imagenet_pretrained):
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if imagenet_pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)

    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if imagenet_pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    elif architecture == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if imagenet_pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)

    elif architecture == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if imagenet_pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, 1)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    if imagenet_pretrained and FREEZE_PRETRAINED_BACKBONE:
        for parameter in model.parameters():
            parameter.requires_grad = False

        if architecture == "resnet18":
            for parameter in model.fc.parameters():
                parameter.requires_grad = True
        elif architecture in {"efficientnet_b0", "mobilenet_v3_large"}:
            for parameter in model.classifier.parameters():
                parameter.requires_grad = True
        elif architecture == "densenet121":
            for parameter in model.classifier.parameters():
                parameter.requires_grad = True

    return model


# ------------------------------------------------------------
# Train, validate, test
# ------------------------------------------------------------

def train_one_epoch(model, loader, loss_function, optimizer):
    model.train()
    total_loss = 0.0

    for images, counts, _ in loader:
        images = images.to(DEVICE)
        counts = counts.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, counts)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, loss_function, save_predictions=False):
    model.eval()
    total_loss = 0.0
    absolute_errors = []
    squared_errors = []
    exact_matches = []
    prediction_rows = []

    with torch.no_grad():
        for images, counts, image_paths in loader:
            images = images.to(DEVICE)
            counts = counts.to(DEVICE)

            raw_outputs = model(images)
            loss = loss_function(raw_outputs, counts)
            total_loss += loss.item() * images.size(0)

            raw_predictions = raw_outputs.cpu().numpy().reshape(-1)
            true_counts = counts.cpu().numpy().reshape(-1).astype(int)

            predicted_counts = np.rint(raw_predictions).astype(int)
            predicted_counts = np.clip(predicted_counts, 0, MAX_BALLS)

            errors = predicted_counts - true_counts
            absolute_errors.extend(np.abs(errors).tolist())
            squared_errors.extend((errors ** 2).tolist())
            exact_matches.extend((predicted_counts == true_counts).astype(float).tolist())

            if save_predictions:
                for image_path, true_count, raw_pred, pred_count in zip(
                    image_paths,
                    true_counts,
                    raw_predictions,
                    predicted_counts,
                ):
                    prediction_rows.append({
                        "image": image_path,
                        "true_count": int(true_count),
                        "total_balls": int(pred_count),
                        "raw_prediction": float(raw_pred),
                    })

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(math.sqrt(np.mean(squared_errors))),
        "accuracy": float(np.mean(exact_matches)),
    }

    if save_predictions:
        metrics["predictions"] = prediction_rows

    return metrics


# ------------------------------------------------------------
# Save/load checkpoints
# ------------------------------------------------------------

def save_checkpoint(path, model, architecture, imagenet_pretrained, epoch, val_metrics):
    torch.save({
        "architecture": architecture,
        "imagenet_pretrained": imagenet_pretrained,
        "epoch": epoch,
        "image_size": IMAGE_SIZE,
        "max_balls": MAX_BALLS,
        "ball_class_ids": sorted(BALL_CLASS_IDS) if BALL_CLASS_IDS is not None else None,
        "model_state_dict": model.state_dict(),
        "validation_metrics": val_metrics,
    }, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model = create_model(
        checkpoint["architecture"],
        checkpoint["imagenet_pretrained"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    return model, checkpoint


# ------------------------------------------------------------
# Main script
# ------------------------------------------------------------

def main():
    set_seed(SEED)

    output_dir = Path(OUTPUT_DIR)
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Output folder: {output_dir}")
    print(f"Ball class ids counted from YOLO labels: {BALL_CLASS_IDS}")

    train_dataset = BallCountDataset(
        DATASET_ROOT,
        "train",
        transform=get_transforms(train=True),
    )
    valid_dataset = BallCountDataset(
        DATASET_ROOT,
        "valid",
        transform=get_transforms(train=False),
    )
    test_dataset = BallCountDataset(
        DATASET_ROOT,
        "test",
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"Train images: {len(train_dataset)}")
    print(f"Valid images: {len(valid_dataset)}")
    print(f"Test images:  {len(test_dataset)}")

    # Quick sanity check: print the target count of one training image.
    example_image = train_dataset.image_paths[0]
    example_count = train_dataset.count_balls(example_image)
    print(f"Example label count: {example_image.name} -> {example_count} balls")

    architectures = [
        "resnet18",
        "efficientnet_b0",
        "mobilenet_v3_large",
        "densenet121",
    ]
    pretrained_options = [True, False]

    loss_function = nn.MSELoss()
    all_results = []

    for architecture in architectures:
        for imagenet_pretrained in pretrained_options:
            weight_name = "imagenet" if imagenet_pretrained else "scratch"
            model_name = f"{architecture}_{weight_name}"
            checkpoint_path = checkpoints_dir / f"{model_name}.pth"

            print("\n" + "=" * 70)
            print(f"Training {model_name}")
            print("=" * 70)

            model = create_model(architecture, imagenet_pretrained).to(DEVICE)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=LEARNING_RATE,
            )

            best_epoch = 0
            best_val_mae = float("inf")
            best_val_metrics = None

            for epoch in range(1, EPOCHS + 1):
                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    loss_function,
                    optimizer,
                )
                val_metrics = evaluate(model, valid_loader, loss_function)

                print(
                    f"Epoch {epoch:03d}/{EPOCHS} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_MAE={val_metrics['mae']:.4f} | "
                    f"val_RMSE={val_metrics['rmse']:.4f} | "
                    f"val_ACC={val_metrics['accuracy']:.4f}"
                )

                if val_metrics["mae"] < best_val_mae:
                    best_epoch = epoch
                    best_val_mae = val_metrics["mae"]
                    best_val_metrics = val_metrics
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        architecture,
                        imagenet_pretrained,
                        epoch,
                        val_metrics,
                    )

            print(f"Best validation epoch: {best_epoch}")
            print(f"Saved checkpoint: {checkpoint_path}")

            best_model, _ = load_checkpoint(checkpoint_path)
            test_metrics = evaluate(
                best_model,
                test_loader,
                loss_function,
                save_predictions=True,
            )

            predictions_path = output_dir / f"test_predictions_{model_name}.json"
            with open(predictions_path, "w", encoding="utf-8") as f:
                json.dump(test_metrics["predictions"], f, indent=2)

            result = {
                "model": model_name,
                "architecture": architecture,
                "imagenet_pretrained": imagenet_pretrained,
                "checkpoint": str(checkpoint_path),
                "best_epoch": best_epoch,
                "validation": best_val_metrics,
                "test": {
                    "loss": test_metrics["loss"],
                    "mae": test_metrics["mae"],
                    "rmse": test_metrics["rmse"],
                    "accuracy": test_metrics["accuracy"],
                },
                "test_predictions_json": str(predictions_path),
            }
            all_results.append(result)

            print(
                f"Test {model_name} | "
                f"MAE={test_metrics['mae']:.4f} | "
                f"RMSE={test_metrics['rmse']:.4f} | "
                f"ACC={test_metrics['accuracy']:.4f}"
            )
            print(f"Saved test predictions: {predictions_path}")

            del model
            del best_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save comparison for all models.
    all_results = sorted(all_results, key=lambda row: (-row["test"]["ACC"], row["test"]["MAE"], row["test"]["RMSE"]))
    comparison_path = output_dir / "model_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON, SORTED BY TEST ACCURACY, THEN MAE, THEN RMSE")
    print("=" * 70)
    for row in all_results:
        print(
            f"{row['model']:28s} | "
            f"val_MAE={row['validation']['mae']:.4f} | "
            f"test_MAE={row['test']['mae']:.4f} | "
            f"test_RMSE={row['test']['rmse']:.4f} | "
            f"test_ACC={row['test']['accuracy']:.4f}"
        )

    print("\nSaved files:")
    print(f"- Checkpoints: {checkpoints_dir}")
    print(f"- Comparison JSON: {comparison_path}")


if __name__ == "__main__":
    main()
