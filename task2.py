"""
Task 2 - More complete CNN ball-counting script
Computer Vision Project - 8-ball pool dataset

One simple script run:
1. Reads a Roboflow YOLO dataset with train/valid/test folders.
2. Counts balls from YOLO labels, ignoring non-ball classes.
3. Trains and validates four architectures:
   - ResNet18
   - EfficientNet-B0
   - MobileNetV3-Large
   - DenseNet121
4. Trains each architecture:
   - with ImageNet pretrained weights
   - from scratch
5. Saves one .pth checkpoint per model.
6. Tests every saved model.
7. Saves comparison metrics, prediction JSON files, training histories, plots,
   confusion matrices, and a best-model final output JSON inside OUTPUT_DIR.

Expected Roboflow YOLO structure:

DATASET_ROOT/
  train/
    images/
    labels/
  valid/          # the script also accepts val/ or validation/
    images/
    labels/
  test/
    images/
    labels/
  data.yaml

Each YOLO label row is expected to be:

    class_id x_center y_center width height

For the dataset example used in this project, class 2 is not a ball and should
be ignored. The target is therefore computed by counting only BALL_CLASS_IDS.
"""

# ============================================================
# EDIT ONLY THESE VARIABLES
# ============================================================

DATASET_ROOT = "./8-ball-pool-dataset"
OUTPUT_DIR = "./task2_output"

EPOCHS = 50
BATCH_SIZE = 16
IMAGE_SIZE = 224
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 42

# 8-ball pool can have cue ball + 15 object balls.
MAX_BALLS = 16

# Your uploaded example label has 26 rows, but only classes 0, 1, 3, 4 are balls.
# Class 2 appears to be table/rail/keypoint annotations, so it is ignored.
BALL_CLASS_IDS = {0, 1, 3, 4}

# Classification is usually better for this task because the output is a
# discrete count: 0, 1, 2, ..., 16.
# You can change this to "regression", but "classification" is recommended.
COUNTING_MODE = "classification"   # "classification" or "regression"

# Training improvements.
PATIENCE = 10                      # early stopping after this many bad epochs
FREEZE_PRETRAINED_BACKBONE = False # True = train only final layer for ImageNet models
SAVE_PLOTS = True

# ============================================================
# SCRIPT STARTS HERE
# ============================================================

import csv
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

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


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

    # More reproducible results. This can make training slightly slower.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------

def count_distribution(counts):
    distribution = {str(i): 0 for i in range(MAX_BALLS + 1)}
    for count in counts:
        count = int(count)
        if 0 <= count <= MAX_BALLS:
            distribution[str(count)] += 1
    return distribution


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else 0.0


def safe_std(values):
    return float(np.std(values)) if len(values) > 0 else 0.0


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class BallCountDataset(Dataset):
    def __init__(self, dataset_root, split, transform=None):
        self.dataset_root = Path(dataset_root)
        self.split_name = split
        self.split_dir = self.find_split_dir(split)
        self.images_dir = self.split_dir / "images"
        self.labels_dir = self.split_dir / "labels"
        self.transform = transform

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing images folder: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Missing labels folder: {self.labels_dir}")

        image_paths = sorted(
            p for p in self.images_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(image_paths) == 0:
            raise FileNotFoundError(f"No images found in: {self.images_dir}")

        self.samples = []
        for image_path in image_paths:
            count = self.count_balls_from_label(image_path)
            if count > MAX_BALLS:
                print(
                    f"Warning: {image_path.name} has count {count}, "
                    f"which is above MAX_BALLS={MAX_BALLS}. It will be clipped."
                )
                count = MAX_BALLS
            self.samples.append((image_path, int(count)))

        self.counts = [count for _, count in self.samples]

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

    def count_balls_from_label(self, image_path):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, count = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if COUNTING_MODE == "classification":
            target = torch.tensor(count, dtype=torch.long)
        elif COUNTING_MODE == "regression":
            target = torch.tensor([float(count)], dtype=torch.float32)
        else:
            raise ValueError("COUNTING_MODE must be 'classification' or 'regression'")

        return image, target, str(image_path), int(count)


# ------------------------------------------------------------
# Image transforms
# ------------------------------------------------------------

def get_transforms(train):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
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

def output_size():
    if COUNTING_MODE == "classification":
        return MAX_BALLS + 1
    return 1


def create_model(architecture, imagenet_pretrained):
    n_outputs = output_size()

    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if imagenet_pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_outputs)

    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if imagenet_pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_outputs)

    elif architecture == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if imagenet_pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_outputs)

    elif architecture == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if imagenet_pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, n_outputs)

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
# Metrics
# ------------------------------------------------------------

def calculate_metrics(true_counts, predicted_counts, losses=None):
    true_counts = np.array(true_counts, dtype=int)
    predicted_counts = np.array(predicted_counts, dtype=int)

    errors = predicted_counts - true_counts
    absolute_errors = np.abs(errors)
    squared_errors = errors ** 2

    if len(true_counts) == 0:
        return {
            "loss": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "accuracy": 0.0,
            "within_one_accuracy": 0.0,
            "mean_error": 0.0,
            "median_absolute_error": 0.0,
            "r2": 0.0,
        }

    ss_res = float(np.sum((true_counts - predicted_counts) ** 2))
    ss_tot = float(np.sum((true_counts - np.mean(true_counts)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "loss": safe_mean(losses) if losses is not None else 0.0,
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(math.sqrt(np.mean(squared_errors))),
        "accuracy": float(np.mean(predicted_counts == true_counts)),
        "within_one_accuracy": float(np.mean(absolute_errors <= 1)),
        "mean_error": float(np.mean(errors)),
        "median_absolute_error": float(np.median(absolute_errors)),
        "r2": float(r2),
    }


def build_confusion_matrix(true_counts, predicted_counts):
    matrix = np.zeros((MAX_BALLS + 1, MAX_BALLS + 1), dtype=int)

    for true_count, predicted_count in zip(true_counts, predicted_counts):
        true_count = int(np.clip(true_count, 0, MAX_BALLS))
        predicted_count = int(np.clip(predicted_count, 0, MAX_BALLS))
        matrix[true_count, predicted_count] += 1

    return matrix


def per_count_metrics(true_counts, predicted_counts):
    true_counts = np.array(true_counts, dtype=int)
    predicted_counts = np.array(predicted_counts, dtype=int)
    result = {}

    for count in range(MAX_BALLS + 1):
        mask = true_counts == count
        support = int(mask.sum())

        if support == 0:
            result[str(count)] = {
                "support": 0,
                "accuracy": None,
                "mae": None,
            }
        else:
            result[str(count)] = {
                "support": support,
                "accuracy": float(np.mean(predicted_counts[mask] == true_counts[mask])),
                "mae": float(np.mean(np.abs(predicted_counts[mask] - true_counts[mask]))),
            }

    return result


# ------------------------------------------------------------
# Train, validate, test
# ------------------------------------------------------------

def make_loss_function():
    if COUNTING_MODE == "classification":
        return nn.CrossEntropyLoss()
    return nn.SmoothL1Loss()


def train_one_epoch(model, loader, loss_function, optimizer):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for images, targets, _, _ in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def outputs_to_counts(outputs):
    if COUNTING_MODE == "classification":
        probabilities = torch.softmax(outputs, dim=1)
        predicted_counts = torch.argmax(probabilities, dim=1)
        confidences = torch.max(probabilities, dim=1).values
        raw_estimates = torch.sum(
            probabilities * torch.arange(MAX_BALLS + 1, device=outputs.device).float(),
            dim=1,
        )
        return (
            predicted_counts.cpu().numpy().astype(int),
            confidences.cpu().numpy().astype(float),
            raw_estimates.cpu().numpy().astype(float),
        )

    raw_outputs = outputs.cpu().numpy().reshape(-1)
    predicted_counts = np.rint(raw_outputs).astype(int)
    predicted_counts = np.clip(predicted_counts, 0, MAX_BALLS)
    confidences = np.zeros_like(raw_outputs, dtype=float)
    return predicted_counts, confidences, raw_outputs.astype(float)


def evaluate(model, loader, loss_function, save_predictions=False):
    model.eval()

    loss_values = []
    true_counts_all = []
    predicted_counts_all = []
    prediction_rows = []

    with torch.no_grad():
        for images, targets, image_paths, true_counts_original in loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)
            loss = loss_function(outputs, targets)

            batch_size = images.size(0)
            loss_values.extend([float(loss.item())] * batch_size)

            predicted_counts, confidences, raw_estimates = outputs_to_counts(outputs.cpu())

            if COUNTING_MODE == "classification":
                true_counts = targets.cpu().numpy().astype(int)
            else:
                true_counts = targets.cpu().numpy().reshape(-1).astype(int)

            true_counts_all.extend(true_counts.tolist())
            predicted_counts_all.extend(predicted_counts.tolist())

            if save_predictions:
                for image_path, true_count, predicted_count, confidence, raw_estimate in zip(
                    image_paths,
                    true_counts,
                    predicted_counts,
                    confidences,
                    raw_estimates,
                ):
                    prediction_rows.append({
                        "image": image_path,
                        "true_count": int(true_count),
                        "total_balls": int(predicted_count),
                        "absolute_error": int(abs(int(predicted_count) - int(true_count))),
                        "raw_estimate": float(raw_estimate),
                        "confidence": float(confidence),
                    })

    metrics = calculate_metrics(true_counts_all, predicted_counts_all, loss_values)
    metrics["per_count"] = per_count_metrics(true_counts_all, predicted_counts_all)
    metrics["confusion_matrix"] = build_confusion_matrix(true_counts_all, predicted_counts_all).tolist()

    if save_predictions:
        metrics["predictions"] = prediction_rows

    return metrics


# ------------------------------------------------------------
# Baselines
# ------------------------------------------------------------

def evaluate_constant_baseline(train_dataset, test_dataset):
    train_counts = np.array(train_dataset.counts, dtype=int)
    test_counts = np.array(test_dataset.counts, dtype=int)

    rounded_mean = int(np.rint(np.mean(train_counts)))
    median = int(np.median(train_counts))
    values, counts = np.unique(train_counts, return_counts=True)
    mode = int(values[np.argmax(counts)])

    baselines = {}

    for name, constant_prediction in {
        "rounded_train_mean": rounded_mean,
        "train_median": median,
        "train_mode": mode,
    }.items():
        predictions = np.full_like(test_counts, constant_prediction)
        metrics = calculate_metrics(test_counts, predictions)
        metrics["constant_prediction"] = int(constant_prediction)
        baselines[name] = metrics

    return baselines


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
        "counting_mode": COUNTING_MODE,
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
# Plots
# ------------------------------------------------------------

def plot_dataset_distribution(path, train_dataset, valid_dataset, test_dataset):
    if not MATPLOTLIB_AVAILABLE or not SAVE_PLOTS:
        return

    counts = np.arange(MAX_BALLS + 1)
    train_dist = [train_dataset.counts.count(int(c)) for c in counts]
    valid_dist = [valid_dataset.counts.count(int(c)) for c in counts]
    test_dist = [test_dataset.counts.count(int(c)) for c in counts]

    width = 0.25
    plt.figure(figsize=(12, 5))
    plt.bar(counts - width, train_dist, width=width, label="train")
    plt.bar(counts, valid_dist, width=width, label="valid")
    plt.bar(counts + width, test_dist, width=width, label="test")
    plt.xlabel("Number of balls")
    plt.ylabel("Number of images")
    plt.title("Ball-count distribution by split")
    plt.xticks(counts)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_training_history(path, history, model_name):
    if not MATPLOTLIB_AVAILABLE or not SAVE_PLOTS:
        return

    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="valid loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training loss - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    metric_path = str(path).replace("_loss.png", "_metrics.png")
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [row["val_mae"] for row in history], label="valid MAE")
    plt.plot(epochs, [row["val_accuracy"] for row in history], label="valid accuracy")
    plt.plot(epochs, [row["val_within_one_accuracy"] for row in history], label="valid within ±1")
    plt.xlabel("Epoch")
    plt.title(f"Validation metrics - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(metric_path, dpi=150)
    plt.close()


def plot_confusion_matrix(path, matrix, model_name):
    if not MATPLOTLIB_AVAILABLE or not SAVE_PLOTS:
        return

    matrix = np.array(matrix, dtype=int)

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix)
    plt.colorbar()
    plt.xlabel("Predicted count")
    plt.ylabel("True count")
    plt.title(f"Confusion matrix - {model_name}")
    plt.xticks(np.arange(MAX_BALLS + 1))
    plt.yticks(np.arange(MAX_BALLS + 1))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ------------------------------------------------------------
# Dataset statistics
# ------------------------------------------------------------

def build_dataset_statistics(train_dataset, valid_dataset, test_dataset):
    stats = {}

    for name, dataset in {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset,
    }.items():
        counts = dataset.counts
        stats[name] = {
            "num_images": len(dataset),
            "min_count": int(min(counts)),
            "max_count": int(max(counts)),
            "mean_count": safe_mean(counts),
            "std_count": safe_std(counts),
            "count_distribution": count_distribution(counts),
        }

    stats["configuration"] = {
        "dataset_root": str(DATASET_ROOT),
        "ball_class_ids": sorted(BALL_CLASS_IDS) if BALL_CLASS_IDS is not None else None,
        "counting_mode": COUNTING_MODE,
        "max_balls": MAX_BALLS,
        "image_size": IMAGE_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "freeze_pretrained_backbone": FREEZE_PRETRAINED_BACKBONE,
        "device": DEVICE,
    }

    return stats


# ------------------------------------------------------------
# Save comparison CSV
# ------------------------------------------------------------

def save_comparison_csv(path, rows):
    fieldnames = [
        "model",
        "architecture",
        "imagenet_pretrained",
        "checkpoint",
        "best_epoch",
        "val_mae",
        "val_rmse",
        "val_accuracy",
        "val_within_one_accuracy",
        "test_mae",
        "test_rmse",
        "test_accuracy",
        "test_within_one_accuracy",
        "test_r2",
        "test_mean_error",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "model": row["model"],
                "architecture": row["architecture"],
                "imagenet_pretrained": row["imagenet_pretrained"],
                "checkpoint": row["checkpoint"],
                "best_epoch": row["best_epoch"],
                "val_mae": row["validation"]["mae"],
                "val_rmse": row["validation"]["rmse"],
                "val_accuracy": row["validation"]["accuracy"],
                "val_within_one_accuracy": row["validation"]["within_one_accuracy"],
                "test_mae": row["test"]["mae"],
                "test_rmse": row["test"]["rmse"],
                "test_accuracy": row["test"]["accuracy"],
                "test_within_one_accuracy": row["test"]["within_one_accuracy"],
                "test_r2": row["test"]["r2"],
                "test_mean_error": row["test"]["mean_error"],
            })


# ------------------------------------------------------------
# Main script
# ------------------------------------------------------------

def main():
    set_seed(SEED)

    output_dir = Path(OUTPUT_DIR)
    checkpoints_dir = output_dir / "checkpoints"
    histories_dir = output_dir / "histories"
    plots_dir = output_dir / "plots"
    predictions_dir = output_dir / "predictions"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Output folder: {output_dir}")
    print(f"Counting mode: {COUNTING_MODE}")
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
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train images: {len(train_dataset)}")
    print(f"Valid images: {len(valid_dataset)}")
    print(f"Test images:  {len(test_dataset)}")

    dataset_stats = build_dataset_statistics(train_dataset, valid_dataset, test_dataset)
    dataset_stats_path = output_dir / "dataset_statistics.json"
    save_json(dataset_stats_path, dataset_stats)
    plot_dataset_distribution(
        plots_dir / "dataset_count_distribution.png",
        train_dataset,
        valid_dataset,
        test_dataset,
    )

    print(f"Saved dataset statistics: {dataset_stats_path}")

    baseline_results = evaluate_constant_baseline(train_dataset, test_dataset)
    baseline_path = output_dir / "baseline_results.json"
    save_json(baseline_path, baseline_results)
    print(f"Saved baseline results: {baseline_path}")

    architectures = [
        "resnet18",
        "efficientnet_b0",
        "mobilenet_v3_large",
        "densenet121",
    ]
    pretrained_options = [True, False]

    loss_function = make_loss_function()
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
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=2,
            )

            best_epoch = 0
            best_val_mae = float("inf")
            best_val_metrics = None
            epochs_without_improvement = 0
            history = []

            for epoch in range(1, EPOCHS + 1):
                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    loss_function,
                    optimizer,
                )
                val_metrics = evaluate(model, valid_loader, loss_function)

                scheduler.step(val_metrics["mae"])

                current_lr = optimizer.param_groups[0]["lr"]

                history.append({
                    "epoch": epoch,
                    "learning_rate": current_lr,
                    "train_loss": float(train_loss),
                    "val_loss": val_metrics["loss"],
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_within_one_accuracy": val_metrics["within_one_accuracy"],
                    "val_r2": val_metrics["r2"],
                })

                print(
                    f"Epoch {epoch:03d}/{EPOCHS} | "
                    f"lr={current_lr:.2e} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_MAE={val_metrics['mae']:.4f} | "
                    f"val_RMSE={val_metrics['rmse']:.4f} | "
                    f"val_ACC={val_metrics['accuracy']:.4f} | "
                    f"val_±1={val_metrics['within_one_accuracy']:.4f}"
                )

                if val_metrics["mae"] < best_val_mae:
                    best_epoch = epoch
                    best_val_mae = val_metrics["mae"]
                    best_val_metrics = {
                        key: value for key, value in val_metrics.items()
                        if key not in {"predictions"}
                    }
                    epochs_without_improvement = 0

                    save_checkpoint(
                        checkpoint_path,
                        model,
                        architecture,
                        imagenet_pretrained,
                        epoch,
                        best_val_metrics,
                    )
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= PATIENCE:
                    print(
                        f"Early stopping: validation MAE did not improve for "
                        f"{PATIENCE} epochs."
                    )
                    break

            history_path = histories_dir / f"training_history_{model_name}.json"
            save_json(history_path, history)

            plot_training_history(
                plots_dir / f"training_history_{model_name}_loss.png",
                history,
                model_name,
            )

            print(f"Best validation epoch: {best_epoch}")
            print(f"Saved checkpoint: {checkpoint_path}")
            print(f"Saved training history: {history_path}")

            best_model, _ = load_checkpoint(checkpoint_path)
            test_metrics = evaluate(
                best_model,
                test_loader,
                loss_function,
                save_predictions=True,
            )

            predictions_path = predictions_dir / f"test_predictions_{model_name}.json"
            save_json(predictions_path, test_metrics["predictions"])

            confusion_path = output_dir / f"confusion_matrix_{model_name}.json"
            save_json(confusion_path, test_metrics["confusion_matrix"])

            plot_confusion_matrix(
                plots_dir / f"confusion_matrix_{model_name}.png",
                test_metrics["confusion_matrix"],
                model_name,
            )

            result = {
                "model": model_name,
                "architecture": architecture,
                "imagenet_pretrained": imagenet_pretrained,
                "counting_mode": COUNTING_MODE,
                "checkpoint": str(checkpoint_path),
                "best_epoch": best_epoch,
                "validation": {
                    "loss": best_val_metrics["loss"],
                    "mae": best_val_metrics["mae"],
                    "rmse": best_val_metrics["rmse"],
                    "accuracy": best_val_metrics["accuracy"],
                    "within_one_accuracy": best_val_metrics["within_one_accuracy"],
                    "mean_error": best_val_metrics["mean_error"],
                    "median_absolute_error": best_val_metrics["median_absolute_error"],
                    "r2": best_val_metrics["r2"],
                    "per_count": best_val_metrics["per_count"],
                },
                "test": {
                    "loss": test_metrics["loss"],
                    "mae": test_metrics["mae"],
                    "rmse": test_metrics["rmse"],
                    "accuracy": test_metrics["accuracy"],
                    "within_one_accuracy": test_metrics["within_one_accuracy"],
                    "mean_error": test_metrics["mean_error"],
                    "median_absolute_error": test_metrics["median_absolute_error"],
                    "r2": test_metrics["r2"],
                    "per_count": test_metrics["per_count"],
                },
                "training_history_json": str(history_path),
                "test_predictions_json": str(predictions_path),
                "confusion_matrix_json": str(confusion_path),
            }
            all_results.append(result)

            print(
                f"Test {model_name} | "
                f"MAE={test_metrics['mae']:.4f} | "
                f"RMSE={test_metrics['rmse']:.4f} | "
                f"ACC={test_metrics['accuracy']:.4f} | "
                f"±1={test_metrics['within_one_accuracy']:.4f} | "
                f"R2={test_metrics['r2']:.4f}"
            )
            print(f"Saved test predictions: {predictions_path}")

            del model
            del best_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Sort by exact-count accuracy first, then MAE, then RMSE.
    all_results = sorted(
        all_results,
        key=lambda row: (
            -row["test"]["accuracy"],
            -row["test"]["within_one_accuracy"],
            row["test"]["mae"],
            row["test"]["rmse"],
        ),
    )

    comparison_path = output_dir / "model_comparison.json"
    comparison_csv_path = output_dir / "model_comparison.csv"
    save_json(comparison_path, all_results)
    save_comparison_csv(comparison_csv_path, all_results)

    best_result = all_results[0]
    best_predictions_path = Path(best_result["test_predictions_json"])

    with open(best_predictions_path, "r", encoding="utf-8") as f:
        best_predictions = json.load(f)

    # Final output focused on the project output: image path + total number of balls.
    final_output = [
        {
            "image": row["image"],
            "total_balls": row["total_balls"],
        }
        for row in best_predictions
    ]

    final_output_path = output_dir / "final_output_best_model.json"
    save_json(final_output_path, final_output)

    readme_path = output_dir / "summary.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("Task 2 - Ball-counting model comparison\n")
        f.write("======================================\n\n")
        f.write(f"Dataset root: {DATASET_ROOT}\n")
        f.write(f"Counting mode: {COUNTING_MODE}\n")
        f.write(f"Ball class IDs: {BALL_CLASS_IDS}\n")
        f.write(f"Best model: {best_result['model']}\n")
        f.write(f"Best checkpoint: {best_result['checkpoint']}\n")
        f.write(f"Best test MAE: {best_result['test']['mae']:.4f}\n")
        f.write(f"Best test RMSE: {best_result['test']['rmse']:.4f}\n")
        f.write(f"Best test accuracy: {best_result['test']['accuracy']:.4f}\n")
        f.write(f"Best test within ±1 accuracy: {best_result['test']['within_one_accuracy']:.4f}\n\n")
        f.write("Main generated files:\n")
        f.write(f"- {comparison_path}\n")
        f.write(f"- {comparison_csv_path}\n")
        f.write(f"- {final_output_path}\n")
        f.write(f"- {dataset_stats_path}\n")
        f.write(f"- {baseline_path}\n")
        f.write(f"- {checkpoints_dir}\n")
        f.write(f"- {histories_dir}\n")
        f.write(f"- {plots_dir}\n")
        f.write(f"- {predictions_dir}\n")

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("Sorted by test accuracy, then within ±1 accuracy, then MAE/RMSE")
    print("=" * 70)
    for row in all_results:
        print(
            f"{row['model']:28s} | "
            f"val_MAE={row['validation']['mae']:.4f} | "
            f"test_MAE={row['test']['mae']:.4f} | "
            f"test_RMSE={row['test']['rmse']:.4f} | "
            f"test_ACC={row['test']['accuracy']:.4f} | "
            f"test_±1={row['test']['within_one_accuracy']:.4f}"
        )

    print("\nBest model:")
    print(f"- {best_result['model']}")
    print(f"- checkpoint: {best_result['checkpoint']}")
    print(f"- final output JSON: {final_output_path}")

    print("\nSaved files:")
    print(f"- Checkpoints: {checkpoints_dir}")
    print(f"- Dataset statistics: {dataset_stats_path}")
    print(f"- Baselines: {baseline_path}")
    print(f"- Comparison JSON: {comparison_path}")
    print(f"- Comparison CSV: {comparison_csv_path}")
    print(f"- Summary: {readme_path}")


if __name__ == "__main__":
    main()
