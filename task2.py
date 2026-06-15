"""
Task 2 - CNN ball-counting model comparison
Computer Vision Project - 8-ball pool dataset

This is a single, simple script:
1. Reads a Roboflow YOLO dataset with train/valid/test folders.
2. Counts balls from YOLO label files, ignoring non-ball classes.
3. Trains several CNN architectures, including AlexNet, VGG, GoogLeNet, ResNet, ResNeXt, ConvNeXt, EfficientNet, MobileNet, DenseNet, ShuffleNet and SqueezeNet.
4. For every architecture, trains 4 variants:
   - classification + ImageNet pretrained
   - classification + scratch
   - logistic + ImageNet pretrained
   - logistic + scratch
5. Saves one .pth checkpoint per trained model.
6. Tests every saved model on the test split.
7. Saves JSON/CSV outputs inside OUTPUT_DIR.

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

Each YOLO label row is expected to be:

    class_id x_center y_center width height

For the dataset example used in this project, class 2 is not a ball and should
be ignored. The target count is computed by counting only BALL_CLASS_IDS.

Counting modes:
- classification: predicts one of the discrete classes 0, 1, ..., MAX_BALLS
  using CrossEntropyLoss.
- logistic: predicts one continuous value in [0, MAX_BALLS] using a sigmoid
  output scaled by MAX_BALLS and SmoothL1Loss. The final count is rounded.
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

# Each architecture below is trained 4 times:
# classification + ImageNet, classification + scratch, logistic + ImageNet,
# logistic + scratch.
COUNTING_MODES = ["classification", "logistic"]
PRETRAINED_OPTIONS = [True, False]

# More CNN architectures for a stronger quantitative comparison.
# Each architecture below is trained 4 times:
# classification + ImageNet, classification + scratch, logistic + ImageNet,
# logistic + scratch.
# To reduce runtime, remove architectures from this list.
ARCHITECTURES = [
    "alexnet",
    "vgg16",
    "googlenet",
    "resnet18",
    "resnet34",
    "resnext50_32x4d",
    "convnext_tiny",
    "efficientnet_b0",
    "mobilenet_v2",
    "mobilenet_v3_large",
    "densenet121",
    "shufflenet_v2_x1_0",
    "squeezenet1_1",
]

# Training improvements.
PATIENCE = 10                       # early stopping after this many bad epochs
FREEZE_PRETRAINED_BACKBONE = False  # True = train only final head for ImageNet models
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

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else 0.0


def safe_std(values):
    return float(np.std(values)) if len(values) > 0 else 0.0


def count_distribution(counts):
    distribution = {str(i): 0 for i in range(MAX_BALLS + 1)}
    for count in counts:
        count = int(count)
        if 0 <= count <= MAX_BALLS:
            distribution[str(count)] += 1
    return distribution


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def clean_metrics_for_saving(metrics):
    """
    Keeps model_comparison.json compact.
    No confusion matrix, no per-count/per-source details.
    """
    keys = [
        "loss",
        "mae",
        "rmse",
        "accuracy",
        "within_one_accuracy",
        "mean_error",
        "median_absolute_error",
        "r2",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


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

        return image, torch.tensor(count, dtype=torch.long), str(image_path)


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

def output_size(counting_mode):
    if counting_mode == "classification":
        return MAX_BALLS + 1
    if counting_mode == "logistic":
        return 1
    raise ValueError("counting_mode must be 'classification' or 'logistic'")


def create_model(architecture, imagenet_pretrained, counting_mode):
    n_outputs = output_size(counting_mode)

    if architecture == "alexnet":
        weights = models.AlexNet_Weights.DEFAULT if imagenet_pretrained else None
        model = models.alexnet(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_outputs)

    elif architecture == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if imagenet_pretrained else None
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_outputs)


    elif architecture == "googlenet":
        weights = models.GoogLeNet_Weights.DEFAULT if imagenet_pretrained else None
        # aux_logits=False keeps the forward output simple: one tensor, not auxiliary outputs.
        model = models.googlenet(weights=weights, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, n_outputs)

    elif architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if imagenet_pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_outputs)

    elif architecture == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if imagenet_pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_outputs)

    elif architecture == "resnext50_32x4d":
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT if imagenet_pretrained else None
        model = models.resnext50_32x4d(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_outputs)

    elif architecture == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if imagenet_pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, n_outputs)

    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if imagenet_pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_outputs)

    elif architecture == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if imagenet_pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_outputs)

    elif architecture == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if imagenet_pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_outputs)

    elif architecture == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if imagenet_pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, n_outputs)

    elif architecture == "shufflenet_v2_x1_0":
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if imagenet_pretrained else None
        model = models.shufflenet_v2_x1_0(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_outputs)

    elif architecture == "squeezenet1_1":
        weights = models.SqueezeNet1_1_Weights.DEFAULT if imagenet_pretrained else None
        model = models.squeezenet1_1(weights=weights)
        model.classifier[1] = nn.Conv2d(
            in_channels=512,
            out_channels=n_outputs,
            kernel_size=1,
        )
        model.num_classes = n_outputs

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    if imagenet_pretrained and FREEZE_PRETRAINED_BACKBONE:
        freeze_backbone_and_unfreeze_head(model, architecture)

    return model


def freeze_backbone_and_unfreeze_head(model, architecture):
    for parameter in model.parameters():
        parameter.requires_grad = False

    if architecture.startswith("resnet"):
        head = model.fc
    elif architecture.startswith("resnext"):
        head = model.fc
    elif architecture.startswith("shufflenet"):
        head = model.fc
    elif architecture.startswith("alexnet"):
        head = model.classifier
    elif architecture.startswith("vgg"):
        head = model.classifier
    elif architecture.startswith("googlenet"):
        head = model.fc
    elif architecture.startswith("convnext"):
        head = model.classifier
    elif architecture.startswith("densenet"):
        head = model.classifier
    elif architecture.startswith("efficientnet"):
        head = model.classifier
    elif architecture.startswith("mobilenet"):
        head = model.classifier
    elif architecture.startswith("squeezenet"):
        head = model.classifier
    else:
        raise ValueError(f"Unknown architecture for freezing: {architecture}")

    for parameter in head.parameters():
        parameter.requires_grad = True


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def calculate_metrics(true_counts, predicted_counts, losses=None):
    true_counts = np.array(true_counts, dtype=int)
    predicted_counts = np.array(predicted_counts, dtype=int)

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

    errors = predicted_counts - true_counts
    absolute_errors = np.abs(errors)
    squared_errors = errors ** 2

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


# ------------------------------------------------------------
# Train, validate, test
# ------------------------------------------------------------

def make_loss_function(counting_mode):
    if counting_mode == "classification":
        return nn.CrossEntropyLoss()
    if counting_mode == "logistic":
        return nn.SmoothL1Loss()
    raise ValueError("counting_mode must be 'classification' or 'logistic'")


def make_targets(counts, counting_mode):
    if counting_mode == "classification":
        return counts.to(DEVICE)
    if counting_mode == "logistic":
        return counts.float().unsqueeze(1).to(DEVICE)
    raise ValueError("counting_mode must be 'classification' or 'logistic'")


def train_one_epoch(model, loader, loss_function, optimizer, counting_mode):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for images, counts, _ in loader:
        images = images.to(DEVICE)
        targets = make_targets(counts, counting_mode)

        optimizer.zero_grad()
        outputs = model(images)

        if counting_mode == "logistic":
            outputs_for_loss = torch.sigmoid(outputs) * MAX_BALLS
            loss = loss_function(outputs_for_loss, targets)
        else:
            loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def outputs_to_counts(outputs, counting_mode):
    if counting_mode == "classification":
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

    if counting_mode == "logistic":
        scaled_outputs = torch.sigmoid(outputs).reshape(-1) * MAX_BALLS
        predicted_counts = torch.rint(scaled_outputs).clamp(0, MAX_BALLS)
        # Confidence is not a probability here, but this value is useful:
        # it is higher when the scaled output is close to an integer count.
        distance_to_integer = torch.abs(scaled_outputs - torch.rint(scaled_outputs))
        pseudo_confidence = 1.0 - distance_to_integer.clamp(0, 1)
        return (
            predicted_counts.cpu().numpy().astype(int),
            pseudo_confidence.cpu().numpy().astype(float),
            scaled_outputs.cpu().numpy().astype(float),
        )

    raise ValueError("counting_mode must be 'classification' or 'logistic'")


def evaluate(model, loader, loss_function, counting_mode, save_predictions=False):
    model.eval()

    loss_values = []
    true_counts_all = []
    predicted_counts_all = []
    prediction_rows = []

    with torch.no_grad():
        for images, counts, image_paths in loader:
            images = images.to(DEVICE)
            targets = make_targets(counts, counting_mode)

            outputs = model(images)

            if counting_mode == "logistic":
                outputs_for_loss = torch.sigmoid(outputs) * MAX_BALLS
                loss = loss_function(outputs_for_loss, targets)
            else:
                loss = loss_function(outputs, targets)

            batch_size = images.size(0)
            loss_values.extend([float(loss.item())] * batch_size)

            predicted_counts, confidences, raw_estimates = outputs_to_counts(outputs, counting_mode)
            true_counts = counts.numpy().astype(int)

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

def save_checkpoint(
    path,
    model,
    architecture,
    imagenet_pretrained,
    counting_mode,
    epoch,
    val_metrics,
    num_parameters,
    trainable_parameters,
):
    torch.save({
        "architecture": architecture,
        "imagenet_pretrained": imagenet_pretrained,
        "counting_mode": counting_mode,
        "epoch": epoch,
        "image_size": IMAGE_SIZE,
        "max_balls": MAX_BALLS,
        "ball_class_ids": sorted(BALL_CLASS_IDS) if BALL_CLASS_IDS is not None else None,
        "num_parameters": num_parameters,
        "trainable_parameters": trainable_parameters,
        "model_state_dict": model.state_dict(),
        "validation_metrics": val_metrics,
    }, path)


def load_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=DEVICE)

    model = create_model(
        checkpoint["architecture"],
        checkpoint["imagenet_pretrained"],
        checkpoint["counting_mode"],
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
        "max_balls": MAX_BALLS,
        "image_size": IMAGE_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "freeze_pretrained_backbone": FREEZE_PRETRAINED_BACKBONE,
        "counting_modes": COUNTING_MODES,
        "pretrained_options": PRETRAINED_OPTIONS,
        "architectures": ARCHITECTURES,
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
        "counting_mode",
        "imagenet_pretrained",
        "checkpoint",
        "best_epoch",
        "num_parameters",
        "trainable_parameters",
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
                "counting_mode": row["counting_mode"],
                "imagenet_pretrained": row["imagenet_pretrained"],
                "checkpoint": row["checkpoint"],
                "best_epoch": row["best_epoch"],
                "num_parameters": row["num_parameters"],
                "trainable_parameters": row["trainable_parameters"],
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
    print(f"Architectures: {ARCHITECTURES}")
    print(f"Counting modes: {COUNTING_MODES}")
    print(f"Pretrained options: {PRETRAINED_OPTIONS}")
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

    all_results = []
    total_models = len(ARCHITECTURES) * len(COUNTING_MODES) * len(PRETRAINED_OPTIONS)
    model_counter = 0

    for architecture in ARCHITECTURES:
        for counting_mode in COUNTING_MODES:
            for imagenet_pretrained in PRETRAINED_OPTIONS:
                model_counter += 1
                weight_name = "imagenet" if imagenet_pretrained else "scratch"
                model_name = f"{architecture}_{counting_mode}_{weight_name}"
                checkpoint_path = checkpoints_dir / f"{model_name}.pth"

                print("\n" + "=" * 80)
                print(f"Training model {model_counter}/{total_models}: {model_name}")
                print("=" * 80)

                model = create_model(architecture, imagenet_pretrained, counting_mode).to(DEVICE)
                num_parameters, trainable_parameters = count_parameters(model)

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
                loss_function = make_loss_function(counting_mode)

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
                        counting_mode,
                    )
                    val_metrics = evaluate(
                        model,
                        valid_loader,
                        loss_function,
                        counting_mode,
                    )

                    scheduler.step(val_metrics["mae"])
                    current_lr = optimizer.param_groups[0]["lr"]

                    history.append({
                        "epoch": epoch,
                        "learning_rate": float(current_lr),
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
                        best_val_metrics = clean_metrics_for_saving(val_metrics)
                        epochs_without_improvement = 0

                        save_checkpoint(
                            checkpoint_path,
                            model,
                            architecture,
                            imagenet_pretrained,
                            counting_mode,
                            epoch,
                            best_val_metrics,
                            num_parameters,
                            trainable_parameters,
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
                    counting_mode,
                    save_predictions=True,
                )

                predictions_path = predictions_dir / f"test_predictions_{model_name}.json"
                save_json(predictions_path, test_metrics["predictions"])

                result = {
                    "model": model_name,
                    "architecture": architecture,
                    "counting_mode": counting_mode,
                    "imagenet_pretrained": imagenet_pretrained,
                    "checkpoint": str(checkpoint_path),
                    "best_epoch": best_epoch,
                    "num_parameters": num_parameters,
                    "trainable_parameters": trainable_parameters,
                    "validation": best_val_metrics,
                    "test": clean_metrics_for_saving(test_metrics),
                    "training_history_json": str(history_path),
                    "test_predictions_json": str(predictions_path),
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

    # Sort by exact-count accuracy first, then within ±1 accuracy, then MAE/RMSE.
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

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("Sorted by test accuracy, then within ±1 accuracy, then MAE/RMSE")
    print("=" * 80)
    for row in all_results:
        print(
            f"{row['model']:45s} | "
            f"val_MAE={row['validation']['mae']:.4f} | "
            f"test_MAE={row['test']['mae']:.4f} | "
            f"test_RMSE={row['test']['rmse']:.4f} | "
            f"test_ACC={row['test']['accuracy']:.4f} | "
            f"test_±1={row['test']['within_one_accuracy']:.4f}"
        )

    best_result = all_results[0]
    print("\nBest model:")
    print(f"- {best_result['model']}")
    print(f"- checkpoint: {best_result['checkpoint']}")

    print("\nSaved files:")
    print(f"- Checkpoints: {checkpoints_dir}")
    print(f"- Dataset statistics: {dataset_stats_path}")
    print(f"- Baselines: {baseline_path}")
    print(f"- Training histories: {histories_dir}")
    print(f"- Predictions: {predictions_dir}")
    print(f"- Plots: {plots_dir}")
    print(f"- Comparison JSON: {comparison_path}")
    print(f"- Comparison CSV: {comparison_csv_path}")


if __name__ == "__main__":
    main()
