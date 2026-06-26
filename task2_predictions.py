MODEL_PATH = "./task2_output/checkpoints/resnet18_classification_imagenet.pth"
INPUT_JSON = "./task2_predictions_example/input.json"
OUTPUT_JSON = "./task2_predictions_example/output.json"

BATCH_SIZE = 16
NUM_WORKERS = 2


import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import cv2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}




def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)




def extract_image_paths(input_json_path):
    data = load_json(input_json_path)

    if not isinstance(data, dict):
        raise ValueError(
            "Invalid input JSON format. The file must be a JSON object like: "
            '{"image_path": ["image1.jpg", "image2.jpg"]}'
        )

    if set(data.keys()) != {"image_path"}:
        raise ValueError(
            "Invalid input JSON format. The JSON object must contain exactly "
            'one key: "image_path".'
        )

    image_paths = data["image_path"]

    if not isinstance(image_paths, list):
        raise ValueError(
            'Invalid input JSON format. The value of "image_path" must be a list.'
        )

    if len(image_paths) == 0:
        raise ValueError(
            'Invalid input JSON format. The "image_path" list cannot be empty.'
        )

    if not all(isinstance(path, str) for path in image_paths):
        raise ValueError(
            'Invalid input JSON format. Every element inside "image_path" must be a string.'
        )

    return image_paths


def resolve_image_path(image_path_string):
    image_path = Path(image_path_string)

    if image_path.is_absolute() and image_path.exists():
        return image_path

    input_json_parent = Path(INPUT_JSON).resolve().parent
    candidates = [
        input_json_parent / image_path_string,
        Path.cwd() / image_path_string,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find image: {image_path_string}\nSearched:\n{searched}"
    )




class LetterboxResize:

    def __init__(self, size, fill=0):
        self.size = int(size)
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        scale = self.size / max(width, height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        image = image.resize((new_width, new_height), Image.BILINEAR)
        pad_left = (self.size - new_width) // 2
        pad_top = (self.size - new_height) // 2
        pad_right = self.size - new_width - pad_left
        pad_bottom = self.size - new_height - pad_top

        return ImageOps.expand(
            image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=(self.fill, self.fill, self.fill),
        )


def automatic_table_crop(pil_image):

    rgb = np.array(pil_image.convert("RGB"))
    height, width = rgb.shape[:2]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    lower = np.array([35, 40, 35], dtype=np.uint8)
    upper = np.array([125, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_image

    candidates = []
    image_area = float(width * height)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.015 * image_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        aspect = w / float(h)
        area_ratio = (w * h) / image_area
        center_x = (x + w / 2.0) / width
        center_y = (y + h / 2.0) / height
        touches_bottom = (y + h) > 0.93 * height

        if touches_bottom:
            continue
        if not (1.0 <= aspect <= 4.5):
            continue
        if not (0.03 <= area_ratio <= 0.80):
            continue
        if not (0.15 <= center_x <= 0.85 and 0.15 <= center_y <= 0.85):
            continue

        center_distance = abs(center_x - 0.5) + abs(center_y - 0.5)
        score = area - 0.10 * image_area * center_distance
        candidates.append((score, x, y, w, h))

    if not candidates:
        return pil_image

    _, x, y, w, h = max(candidates, key=lambda row: row[0])

    pad = int(0.20 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)

    if (x2 - x1) < 50 or (y2 - y1) < 50:
        return pil_image

    return pil_image.crop((x1, y1, x2, y2))


def get_transform(image_size):
    return transforms.Compose([
        LetterboxResize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])




class InferenceImageDataset(Dataset):

    def __init__(self, image_path_strings, transform):
        self.image_path_strings = image_path_strings
        self.resolved_paths = [resolve_image_path(p) for p in image_path_strings]
        self.transform = transform

    def __len__(self):
        return len(self.image_path_strings)

    def __getitem__(self, index):
        original_string = self.image_path_strings[index]
        resolved_path = self.resolved_paths[index]

        image = Image.open(resolved_path).convert("RGB")
        image = automatic_table_crop(image)

        image = self.transform(image)
        return image, original_string




def output_size(counting_mode, max_balls):

    if counting_mode == "classification":
        return max_balls + 1
    if counting_mode == "logistic":
        return 1
    raise ValueError("counting_mode must be 'classification' or 'logistic'")


def create_model(architecture, counting_mode, max_balls):
    n_outputs = output_size(counting_mode, max_balls)

    if architecture == "alexnet":
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_outputs)

    elif architecture == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_outputs)

    elif architecture == "googlenet":
        model = models.googlenet(weights=None, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, n_outputs)

    elif architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_outputs)

    elif architecture == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_outputs)

    elif architecture == "resnext50_32x4d":
        model = models.resnext50_32x4d(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_outputs)

    elif architecture == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, n_outputs)

    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_outputs)

    elif architecture == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_outputs)

    elif architecture == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_outputs)

    elif architecture == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, n_outputs)

    elif architecture == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_outputs)

    elif architecture == "squeezenet1_1":
        model = models.squeezenet1_1(weights=None)
        model.classifier[1] = nn.Conv2d(
            in_channels=512,
            out_channels=n_outputs,
            kernel_size=1,
        )
        model.num_classes = n_outputs

    else:
        raise ValueError(f"Unknown architecture in checkpoint: {architecture}")

    return model


def forward_logits(model, images):

    outputs = model(images)
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    
    return outputs




def load_trained_model(model_path):

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=DEVICE)

    required_keys = ["architecture", "counting_mode", "model_state_dict"]
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise KeyError(
            f"Checkpoint is missing keys: {missing_keys}. "
            "This script expects a checkpoint saved by the Task 2 training script."
        )

    architecture = checkpoint["architecture"]
    counting_mode = checkpoint["counting_mode"]
    image_size = int(checkpoint.get("image_size", 512))
    max_balls = int(checkpoint.get("max_balls", 16))

    model = create_model(architecture, counting_mode, max_balls)

    state_dict = checkpoint["model_state_dict"]
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        print("Strict loading failed. Trying strict=False instead.")
        print(f"Original loading error: {error}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

    model.to(DEVICE)
    model.eval()

    model_info = {
        "architecture": architecture,
        "counting_mode": counting_mode,
        "image_size": image_size,
        "max_balls": max_balls,
        "epoch": checkpoint.get("epoch", None),
        "validation_metrics": checkpoint.get("validation_metrics", None),
    }

    return model, model_info


def outputs_to_counts(outputs, counting_mode, max_balls):

    if counting_mode == "classification":
        probabilities = torch.softmax(outputs, dim=1)
        class_indices = torch.arange(max_balls + 1, device=outputs.device).float()
        expected_values = torch.sum(probabilities * class_indices, dim=1)
        confidences = torch.max(probabilities, dim=1).values
        predicted_counts = torch.round(expected_values).clamp(0, max_balls)

        return (
            predicted_counts.cpu().numpy().astype(int),
            confidences.cpu().numpy().astype(float),
            expected_values.cpu().numpy().astype(float),
        )

    if counting_mode == "logistic":
        scaled_outputs = torch.sigmoid(outputs).reshape(-1) * max_balls
        predicted_counts = torch.round(scaled_outputs).clamp(0, max_balls)
        distance_to_integer = torch.abs(scaled_outputs - torch.round(scaled_outputs))
        pseudo_confidence = 1.0 - distance_to_integer.clamp(0, 1)

        return (
            predicted_counts.cpu().numpy().astype(int),
            pseudo_confidence.cpu().numpy().astype(float),
            scaled_outputs.cpu().numpy().astype(float),
        )

    raise ValueError("counting_mode must be 'classification' or 'logistic'")


def predict(model, model_info, image_path_strings):

    transform = get_transform(model_info["image_size"])
    dataset = InferenceImageDataset(image_path_strings, transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    results = []

    with torch.no_grad():
        for images, original_paths in loader:
            images = images.to(DEVICE)
            outputs = forward_logits(model, images)
            predicted_counts, confidences, raw_estimates = outputs_to_counts(
                outputs,
                model_info["counting_mode"],
                model_info["max_balls"],
            )

            for image_path, num_balls, confidence, raw_estimate in zip(
                original_paths,
                predicted_counts,
                confidences,
                raw_estimates,
            ):
                row = {
                    "image_path": str(image_path),
                    "num_balls": int(num_balls),
                }

                results.append(row)

    return results




def main():
    print(f"Using device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Input JSON: {INPUT_JSON}")
    print(f"Output JSON: {OUTPUT_JSON}")

    model, model_info = load_trained_model(MODEL_PATH)

    print("Loaded model:")
    for key, value in model_info.items():
        if key != "validation_metrics":
            print(f"- {key}: {value}")

    image_paths = extract_image_paths(INPUT_JSON)
    print(f"Number of input images: {len(image_paths)}")

    results = predict(model, model_info, image_paths)
    save_json(OUTPUT_JSON, results)

    print(f"Saved predictions to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
