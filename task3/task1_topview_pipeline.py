import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Display
def show(title, img):
    plt.figure(figsize=(10, 6))
    if len(img.shape) == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# IO
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image not found")
    return img


# KMEANS segmentation
def kmeans_cluster_refined(img, K=5, spatial_weight=20, blur_size=9):
    img_blur = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    img_lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2Lab)
    h, w = img_lab.shape[:2]
    color_features = img_lab.reshape((-1, 3)).astype(np.float32)

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = (x / w).reshape(-1, 1).astype(np.float32)
    y = (y / h).reshape(-1, 1).astype(np.float32)

    Z = np.hstack([color_features, x * spatial_weight, y * spatial_weight])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, centers = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    label_img = labels.reshape((h, w)).astype(np.uint8)
    return label_img.flatten(), (h, w), K


def get_cluster_masks(labels, shape, K):
    h, w = shape
    masks = []

    for i in range(K):
        m = (labels == i).astype(np.uint8).reshape(h, w) * 255

        _, comp, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if len(stats) <= 1:
            masks.append(m)
            continue

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        out = np.zeros_like(m)
        out[comp == largest] = 255

        masks.append(out)

    return masks


# Blue cluster detection
def find_blue_cluster(img, labels, K):
    best_idx, best_score = -1, -1

    for i in range(K):
        pixels = img.reshape(-1, 3)[labels == i]
        if len(pixels) == 0:
            continue

        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

        h = hsv[:, 0]
        s = hsv[:, 1]
        v = hsv[:, 2]

        mask = (h >= 90) & (h <= 140) & (s > 60) & (v > 40)

        blue_ratio = np.sum(mask) / len(mask + 1e-6)

        saturation_score = np.mean(s[mask]) / 255 if np.sum(mask) > 0 else 0
        purity = blue_ratio

        size_factor = len(pixels)

        score = purity * saturation_score * np.log1p(size_factor)

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def select_target_cluster(img, labels, masks, blue_idx):

    kernel = np.ones((5, 5), np.uint8)

    blue = cv2.morphologyEx(masks[blue_idx], cv2.MORPH_OPEN, kernel, iterations=1)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel, iterations=2)

    blue_near = cv2.dilate(blue, kernel, iterations=2)

    inv_blue = (blue == 0).astype(np.uint8)
    dist_map = cv2.distanceTransform(inv_blue, cv2.DIST_L2, 5)

    best_i, best_score = -1, -1

    for i, mask in enumerate(masks):
        if i == blue_idx:
            continue

        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num, comp, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if num > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            m = np.where(comp == largest, 255, 0).astype(np.uint8)

        size = np.sum(m) / 255
        if size < 1000:
            continue

        overlap = np.sum(cv2.bitwise_and(m, blue_near)) / 255
        proximity = overlap / (size + 1e-6)

        coords = np.column_stack(np.where(m > 0))
        dists = dist_map[coords[:, 0], coords[:, 1]]
        dist_score = 1.0 / (np.median(dists) + 1e-6)

        components = cv2.connectedComponents(m)[0]
        frag_penalty = 1.0 / (components + 1e-6)

        score = (
            0.55 * proximity +
            0.30 * dist_score +
            0.15 * frag_penalty
        )

        if score > best_score:
            best_score, best_i = score, i

    return best_i


# Line merging
def merge_lines(lines, angle_thresh=0.3, dist_thresh=210):

    merged = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        m, b = ((y2 - y1)/(x2 - x1), y1 - ((y2 - y1)/(x2 - x1))*x1) if x2 != x1 else (np.inf, x1)

        found = False
        for idx, (mr, br, x1r, y1r, x2r, y2r) in enumerate(merged):
            if (m != np.inf and mr != np.inf and abs(m - mr) < angle_thresh and abs(b - br) < dist_thresh) or \
               (m == np.inf and mr == np.inf and abs(b - br) < dist_thresh):
                xs, ys = [x1, x2, x1r, x2r], [y1, y2, y1r, y2r]
                merged[idx] = (mr, br, min(xs), min(ys), max(xs), max(ys))
                found = True
                break
        if not found:
            merged.append((m, b, x1, y1, x2, y2))
    return merged


def intersect(l1, l2):
    m1, b1, x1, y1, x2, y2 = l1
    m2, b2, x3, y3, x4, y4 = l2

    if m1 == np.inf and m2 == np.inf:
        return None

    if m1 == np.inf:
        px = b1
        py = m2 * px + b2
    elif m2 == np.inf:
        px = b2
        py = m1 * px + b1
    elif abs(m1 - m2) < 1e-4:
        return None
    else:
        px = (b2 - b1) / (m1 - m2)
        py = m1 * px + b1

    return (px, py)


def compute_corners_from_lines(lines):
    if len(lines) != 4:
        return None

    data = []
    for l in lines:
        m = l[0]
        angle = np.degrees(np.arctan(m)) if m != np.inf else 90
        data.append((angle, l))

    data.sort(key=lambda x: x[0])

    pairs = [
        abs(data[0][0] - data[1][0]),
        abs(data[1][0] - data[2][0]),
        abs(data[2][0] - data[3][0]),
        180 - abs(data[0][0] - data[3][0])
    ]

    idx = np.argmin(pairs)

    if idx == 0:
        A, B = [data[0][1], data[1][1]], [data[2][1], data[3][1]]
    elif idx == 1:
        A, B = [data[1][1], data[2][1]], [data[0][1], data[3][1]]
    elif idx == 2:
        A, B = [data[2][1], data[3][1]], [data[0][1], data[1][1]]
    else:
        A, B = [data[3][1], data[0][1]], [data[1][1], data[2][1]]

    pts = []
    for a in A:
        for b in B:
            p = intersect(a, b)
            if p:
                pts.append(p)

    return pts if len(pts) == 4 else None


def get_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def order_points(pts):
    pts.sort(key=lambda c: c[1])
    # Identify Top-Left vs Top-Right
    tl = pts[0] if get_dist(pts[0], [0, pts[0][1]]) < get_dist(pts[1], [0, pts[1][1]]) else pts[1]
    tr = pts[1] if np.array_equal(tl, pts[0]) else pts[0]
    # Identify Bottom-Left vs Bottom-Right
    bl = pts[2] if get_dist(pts[2], [0, pts[2][1]]) < get_dist(pts[3], [0, pts[3][1]]) else pts[3]
    br = pts[3] if np.array_equal(bl, pts[2]) else pts[2]
    return np.array([tl, tr, br, bl], dtype="float32")


# Warp
def warp_from_cluster(img, mask):
    h, w = img.shape[:2]
    PAD = int(0.5 * max(h, w))

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=3
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    outer = max(contours, key=cv2.contourArea)
    outer_mask = np.zeros_like(mask)
    cv2.drawContours(outer_mask, [outer], -1, 255, 3)

    padded = cv2.copyMakeBorder(outer_mask, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, 0)

    edges = cv2.Canny(padded, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=140, maxLineGap=200)

    if lines is None:
        return None

    shifted = [
        np.array([[x1 - PAD, y1 - PAD, x2 - PAD, y2 - PAD]])
        for x1, y1, x2, y2 in lines[:, 0]
    ]
    merged = merge_lines(shifted)
    merged = sorted(merged, key=lambda l: (np.hypot(l[5]-l[3], l[4]-l[2]), abs(l[0])), reverse=True)[:4]

    if len(merged) != 4:
        return None

    corners = compute_corners_from_lines(merged)
    if not corners:
        return None

    src = order_points(list(corners)) + PAD
    dst = np.array([[0, 800], [0, 0], [1600, 0], [1600, 800]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)

    padded_img = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    warped = cv2.warpPerspective(padded_img, M, (1600, 800))

    if np.count_nonzero(warped) / warped.size < 0.2:
        return None

    return warped


def extract_blue_cloth(warped_img, padding=50):
    if warped_img is None:
        return None

    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 60, 40])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No blue cloth detected in warped image.")
        return warped_img

    largest_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_cnt)

    img_h, img_w = warped_img.shape[:2]

    x_new = max(0, x - padding)
    y_new = max(0, y - padding)
    w_new = min(img_w - x_new, w + (2 * padding))
    h_new = min(img_h - y_new, h + (2 * padding))

    return warped_img[y_new: y_new + h_new, x_new: x_new + w_new]


# Ball detection / classification
BALL_NUMBER_MAP = {
    "yellow": (1, 9),
    "blue": (2, 10),
    "red": (3, 11),
    "purple": (4, 12),
    "orange": (5, 13),
    "green": (6, 14),
    "maroon": (7, 15),
}

REFERENCE_BGR_COLORS = {
    "yellow": np.uint8([[[0, 220, 255]]]),
    "blue": np.uint8([[[200, 80, 0]]]),
    "red": np.uint8([[[40, 40, 220]]]),
    "purple": np.uint8([[[135, 60, 135]]]),
    "orange": np.uint8([[[0, 140, 255]]]),
    "green": np.uint8([[[70, 150, 40]]]),
    "maroon": np.uint8([[[45, 60, 120]]]),
}

REFERENCE_LAB_COLORS = {
    "yellow":  np.array([159, 131, 158], dtype=np.float32),
    "blue":    np.array([80,  130,  98], dtype=np.float32),
    "red":     np.array([116, 158, 146], dtype=np.float32),
    "purple":  np.array([126, 122, 122], dtype=np.float32),
    "orange":  np.array([135, 149, 149], dtype=np.float32),
    "green":   np.array([126, 113, 136], dtype=np.float32),
    "maroon":  np.array([124, 133, 145], dtype=np.float32),
}

def get_playing_surface_masks(top_view_img):
    hsv = cv2.cvtColor(top_view_img, cv2.COLOR_BGR2HSV)
    h, w = top_view_img.shape[:2]

    cy1, cy2 = int(0.2 * h), int(0.8 * h)
    cx1, cx2 = int(0.2 * w), int(0.8 * w)
    center = hsv[cy1:cy2, cx1:cx2]

    blue_candidates = (
        (center[:, :, 0] >= 85) & (center[:, :, 0] <= 125) &
        (center[:, :, 1] >= 50) & (center[:, :, 2] >= 100)
    )

    if np.count_nonzero(blue_candidates) == 0:
        cloth_h, cloth_s, cloth_v = 103, 150, 220
    else:
        cloth_h = int(np.median(center[:, :, 0][blue_candidates]))
        cloth_s = int(np.median(center[:, :, 1][blue_candidates]))
        cloth_v = int(np.median(center[:, :, 2][blue_candidates]))

    lower = np.array([
        max(85, cloth_h - 10),
        max(40, cloth_s - 90),
        max(120, cloth_v - 90),
    ], dtype=np.uint8)
    upper = np.array([
        min(125, cloth_h + 10),
        255,
        255,
    ], dtype=np.uint8)

    cloth_mask = cv2.inRange(hsv, lower, upper)
    cloth_mask = cv2.morphologyEx(
        cloth_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    cloth_mask = cv2.morphologyEx(
        cloth_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )

    contours, _ = cv2.findContours(cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, cloth_h

    largest = max(contours, key=cv2.contourArea)

    filled_surface = np.zeros_like(cloth_mask)
    cv2.drawContours(filled_surface, [largest], -1, 255, thickness=-1)

    dist = cv2.distanceTransform(filled_surface, cv2.DIST_L2, 5)
    adaptive_margin = max(4.0, 0.008 * min(h, w))
    safe_inner = (dist > adaptive_margin).astype(np.uint8) * 255

    if cv2.countNonZero(safe_inner) == 0:
        relaxed_margin = max(1.0, 0.003 * min(h, w))
        safe_inner = (dist > relaxed_margin).astype(np.uint8) * 255

    if cv2.countNonZero(safe_inner) == 0:
        safe_inner = filled_surface.copy()

    return cloth_mask, filled_surface, safe_inner, cloth_h


def bbox_iou(det_a, det_b):
    ax = det_a["bbox"]["x"]
    ay = det_a["bbox"]["y"]
    aw = det_a["bbox"]["width"]
    ah = det_a["bbox"]["height"]

    bx = det_b["bbox"]["x"]
    by = det_b["bbox"]["y"]
    bw = det_b["bbox"]["width"]
    bh = det_b["bbox"]["height"]

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter

    return inter / union if union > 0 else 0.0


def center_distance(det_a, det_b):
    ax, ay = det_a["center"]
    bx, by = det_b["center"]
    return float(np.hypot(ax - bx, ay - by))


def component_to_detection(labels, stats, centroids, idx, source):
    x, y, w, h, area = stats[idx]

    component_mask = (labels == idx).astype(np.uint8)
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
    else:
        contour = np.array([
            [[x, y]],
            [[x + w - 1, y]],
            [[x + w - 1, y + h - 1]],
            [[x, y + h - 1]],
        ], dtype=np.int32)

    return {
        "source": source,
        "bbox": {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
        },
        "center": [float(centroids[idx][0]), float(centroids[idx][1])],
        "area": int(area),
        "contour": contour,
    }


def extract_component_detections(binary_mask, source, area_range, ar_max, min_long_side, suppress_tiny_round=True):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
    detections = []

    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if area < area_range[0] or area > area_range[1]:
            continue

        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio > ar_max:
            continue

        if max(w, h) < min_long_side:
            continue

        if suppress_tiny_round and area < 260 and aspect_ratio < 1.4:
            continue

        detections.append(component_to_detection(labels, stats, centroids, idx, source))

    return detections


def detect_ball_candidates(top_view_img):
    cloth_mask, filled_surface, safe_inner, cloth_hue = get_playing_surface_masks(top_view_img)
    if cloth_mask is None:
        return [], cloth_hue

    # Branch 1: balls as holes in the cloth mask
    hole_mask = cv2.subtract(filled_surface, cloth_mask)
    hole_mask = cv2.morphologyEx(
        hole_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    hole_mask = cv2.morphologyEx(
        hole_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )

    detections = extract_component_detections(
        hole_mask,
        source="hole",
        area_range=(80, 5000),
        ar_max=6.0,
        min_long_side=22,
        suppress_tiny_round=True,
    )

    H, W = top_view_img.shape[:2]
    dist_to_border = cv2.distanceTransform(filled_surface, cv2.DIST_L2, 5)
    lab = cv2.cvtColor(top_view_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    candidate_region = safe_inner > 0
    if np.count_nonzero(candidate_region) == 0:
        candidate_region = filled_surface > 0
    if np.count_nonzero(candidate_region) == 0:
        candidate_region = cloth_mask > 0
    if np.count_nonzero(candidate_region) == 0:
        return detections, cloth_hue

    reference_mask = (cloth_mask > 0) & candidate_region
    if np.count_nonzero(reference_mask) == 0:
        reference_mask = cloth_mask > 0
    if np.count_nonzero(reference_mask) == 0:
        reference_mask = filled_surface > 0
    if np.count_nonzero(reference_mask) == 0:
        reference_mask = candidate_region

    cloth_lab = np.median(lab[reference_mask], axis=0)
    delta_lab = np.linalg.norm(lab - cloth_lab, axis=2)

    delta_values = delta_lab[candidate_region]
    if delta_values.size == 0:
        return detections, cloth_hue

    adaptive_thr = max(8.0, float(np.percentile(delta_values, 89)))
    delta_mask = ((delta_lab > adaptive_thr) & candidate_region).astype(np.uint8) * 255
    delta_mask = cv2.morphologyEx(
        delta_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    delta_mask = cv2.morphologyEx(
        delta_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(delta_mask, 8)
    border_clearance = max(8.0, 0.012 * min(H, W))

    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        aspect_ratio = max(w, h) / max(1, min(w, h))
        cx, cy = centroids[idx]

        if area < 250 or area > 5000:
            continue
        if aspect_ratio > 10:
            continue
        if max(w, h) < 18:
            continue
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
            continue

        cx_i = int(np.clip(round(cx), 0, W - 1))
        cy_i = int(np.clip(round(cy), 0, H - 1))
        if dist_to_border[cy_i, cx_i] <= border_clearance:
            continue

        det = component_to_detection(labels, stats, centroids, idx, source="delta")

        duplicate = False
        for prev in detections:
            if bbox_iou(det, prev) > 0.10 or center_distance(det, prev) < 28:
                duplicate = True
                break

        if not duplicate:
            detections.append(det)

    return detections, cloth_hue


def classify_ball_candidate(top_view_img, det, cloth_hue):
    H, W = top_view_img.shape[:2]

    x = det["bbox"]["x"]
    y = det["bbox"]["y"]
    w = det["bbox"]["width"]
    h = det["bbox"]["height"]

    pad = max(2, int(0.15 * max(w, h)))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)

    roi = top_view_img[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    local_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    shifted_contour = det["contour"].copy()
    shifted_contour[:, 0, 0] -= x1
    shifted_contour[:, 0, 1] -= y1
    cv2.drawContours(local_mask, [shifted_contour], -1, 255, thickness=-1)

    local_mask = cv2.erode(
        local_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    if cv2.countNonZero(local_mask) < 40:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    mask_bool = local_mask > 0
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    white_mask = mask_bool & (sat < 60) & (val > 135)
    black_mask = mask_bool & (val < 70)
    colored_mask = mask_bool & (~white_mask) & (~black_mask) & (sat > 45)

    valid_pixels = max(1, int(np.count_nonzero(mask_bool)))
    white_ratio = np.count_nonzero(white_mask) / valid_pixels
    black_ratio = np.count_nonzero(black_mask) / valid_pixels
    colored_ratio = np.count_nonzero(colored_mask) / valid_pixels

    if np.count_nonzero(colored_mask) < 25 and white_ratio < 0.35 and black_ratio < 0.35:
        return None

    if np.count_nonzero(colored_mask) > 0:
        mean_hue = float(np.mean(hsv[:, :, 0][colored_mask]))
        if (
            det["source"] == "delta"
            and abs(mean_hue - cloth_hue) < 14
            and max(w, h) < 45
            and min(w, h) < 20
            and white_ratio < 0.35
            and black_ratio < 0.35
        ):
            return None
    else:
        mean_hue = None

    if white_ratio > 0.58 and colored_ratio < 0.35:
        number = 0
        ball_type = "cue"
        color_name = "white"

    elif black_ratio > 0.50 and colored_ratio < 0.28:
        number = 8
        ball_type = "eight"
        color_name = "black"

    elif np.count_nonzero(colored_mask) >= 20:
        bright_threshold = np.percentile(val[colored_mask], 25)
        bright_colored = colored_mask & (val >= bright_threshold)

        mean_lab = np.mean(lab[bright_colored], axis=0).astype(np.float32)
        distances = {
            name: float(np.linalg.norm(mean_lab - ref_lab))
            for name, ref_lab in REFERENCE_LAB_COLORS.items()
        }
        color_name = min(distances, key=distances.get)

        is_stripe = white_ratio > 0.18
        number = BALL_NUMBER_MAP[color_name][1] if is_stripe else BALL_NUMBER_MAP[color_name][0]
        ball_type = "stripe" if is_stripe else "solid"

    else:
        return None

    return {
        "number": int(number),
        "type": ball_type,
        "color": color_name,
        "bbox": {
            "x": int(x1),
            "y": int(y1),
            "width": int(x2 - x1),
            "height": int(y2 - y1),
        },
        "center": [int(round(det["center"][0])), int(round(det["center"][1]))],
        "source": det["source"],
        "white_ratio": round(float(white_ratio), 3),
        "black_ratio": round(float(black_ratio), 3),
        "colored_ratio": round(float(colored_ratio), 3),
    }


def detect_and_classify_balls(top_view_img, keep_internal=False):
    candidates, cloth_hue = detect_ball_candidates(top_view_img)

    detections = []
    for det in candidates:
        classified = classify_ball_candidate(top_view_img, det, cloth_hue)
        if classified is not None:
            if keep_internal:
                classified["_candidate_bbox"] = {
                    "x": int(det["bbox"]["x"]),
                    "y": int(det["bbox"]["y"]),
                    "width": int(det["bbox"]["width"]),
                    "height": int(det["bbox"]["height"]),
                }
                classified["_candidate_center"] = [float(det["center"][0]), float(det["center"][1])]
                classified["_candidate_contour"] = det["contour"].copy()
            detections.append(classified)

    final = []
    for det in sorted(detections, key=lambda d: (d["bbox"]["y"], d["bbox"]["x"])):
        duplicate = False
        for prev in final:
            if center_distance(det, prev) < 20:
                duplicate = True
                break
        if not duplicate:
            final.append(det)

    return final


def clip_bbox_to_image(bbox, img_shape):
    H, W = img_shape[:2]
    x = int(np.clip(bbox["x"], 0, max(0, W - 1)))
    y = int(np.clip(bbox["y"], 0, max(0, H - 1)))
    x2 = int(np.clip(bbox["x"] + bbox["width"], x + 1, W))
    y2 = int(np.clip(bbox["y"] + bbox["height"], y + 1, H))
    return {"x": x, "y": y, "width": x2 - x, "height": y2 - y}


def project_points_top_view_to_original(points, warp_meta, crop_bbox, original_shape):
    if points is None or warp_meta is None or crop_bbox is None:
        return None

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2).copy()
    pts[:, 0, 0] += float(crop_bbox["x"])
    pts[:, 0, 1] += float(crop_bbox["y"])

    projected = cv2.perspectiveTransform(pts, warp_meta["M_inv"])
    projected[:, 0, 0] -= float(warp_meta["pad"])
    projected[:, 0, 1] -= float(warp_meta["pad"])

    H, W = original_shape[:2]
    projected[:, 0, 0] = np.clip(projected[:, 0, 0], 0, W - 1)
    projected[:, 0, 1] = np.clip(projected[:, 0, 1], 0, H - 1)
    return projected


def build_original_cloth_model(img, blue_mask):
    if blue_mask is None:
        return None

    cloth_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    cloth_mask = cv2.morphologyEx(
        cloth_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=2,
    )

    contours, _ = cv2.findContours(cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    filled_surface = np.zeros_like(cloth_mask)
    cv2.drawContours(filled_surface, [largest], -1, 255, thickness=-1)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    reference_mask = cloth_mask > 0
    if np.count_nonzero(reference_mask) == 0:
        reference_mask = filled_surface > 0
    if np.count_nonzero(reference_mask) == 0:
        return None

    cloth_lab = np.median(lab[reference_mask], axis=0)
    return {
        "cloth_mask": cloth_mask,
        "filled_surface": filled_surface,
        "cloth_lab": cloth_lab,
    }


def refine_bbox_on_original(original_img, approx_bbox, approx_center, cloth_model):
    approx_bbox = clip_bbox_to_image(approx_bbox, original_img.shape)
    if cloth_model is None:
        return approx_bbox, [int(round(approx_center[0])), int(round(approx_center[1]))]

    H, W = original_img.shape[:2]
    x, y, w, h = (
        approx_bbox["x"],
        approx_bbox["y"],
        approx_bbox["width"],
        approx_bbox["height"],
    )

    grow = max(8, int(0.35 * max(w, h)))
    x1 = max(0, x - grow)
    y1 = max(0, y - grow)
    x2 = min(W, x + w + grow)
    y2 = min(H, y + h + grow)

    roi = original_img[y1:y2, x1:x2]
    surface_roi = cloth_model["filled_surface"][y1:y2, x1:x2]
    cloth_roi = cloth_model["cloth_mask"][y1:y2, x1:x2]

    if roi.size == 0 or cv2.countNonZero(surface_roi) == 0:
        return approx_bbox, [int(round(approx_center[0])), int(round(approx_center[1]))]

    hole_mask = cv2.subtract(surface_roi, cloth_roi)
    hole_mask = cv2.morphologyEx(
        hole_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    hole_mask = cv2.morphologyEx(
        hole_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta_lab = np.linalg.norm(lab - cloth_model["cloth_lab"], axis=2)
    candidate_region = surface_roi > 0
    delta_values = delta_lab[candidate_region]

    if delta_values.size > 0:
        adaptive_thr = max(10.0, float(np.percentile(delta_values, 87)))
    else:
        adaptive_thr = 12.0

    delta_mask = ((delta_lab > adaptive_thr) & candidate_region).astype(np.uint8) * 255
    delta_mask = cv2.morphologyEx(
        delta_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    delta_mask = cv2.morphologyEx(
        delta_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )

    combined = cv2.bitwise_or(hole_mask, delta_mask)
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, 8)
    local_cx = float(np.clip(approx_center[0] - x1, 0, max(0, x2 - x1 - 1)))
    local_cy = float(np.clip(approx_center[1] - y1, 0, max(0, y2 - y1 - 1)))
    approx_area = max(1.0, float(w * h))

    best_idx = None
    best_score = float("inf")

    for idx in range(1, num):
        bx, by, bw, bh, area = stats[idx]
        if area < max(20, 0.10 * approx_area):
            continue
        if area > 5.5 * approx_area:
            continue

        aspect_ratio = max(bw, bh) / max(1, min(bw, bh))
        if aspect_ratio > 4.5:
            continue

        ccx, ccy = centroids[idx]
        center_dist = float(np.hypot(ccx - local_cx, ccy - local_cy))
        size_penalty = 0.35 * abs(np.log((area + 1.0) / (approx_area + 1.0))) * max(w, h)

        lx = int(np.clip(round(local_cx), 0, labels.shape[1] - 1))
        ly = int(np.clip(round(local_cy), 0, labels.shape[0] - 1))
        contains_center = labels[ly, lx] == idx

        score = center_dist + size_penalty
        if not contains_center:
            score += 12.0

        if score < best_score:
            best_score = score
            best_idx = idx

    if best_idx is None:
        return approx_bbox, [int(round(approx_center[0])), int(round(approx_center[1]))]

    component = (labels == best_idx).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cx, cy, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
    else:
        cx, cy, cw, ch, _ = stats[best_idx]

    pad = 2
    rx = x1 + max(0, cx - pad)
    ry = y1 + max(0, cy - pad)
    rw = min((x2 - x1) - max(0, cx - pad), cw + 2 * pad)
    rh = min((y2 - y1) - max(0, cy - pad), ch + 2 * pad)

    refined_bbox = clip_bbox_to_image(
        {"x": rx, "y": ry, "width": rw, "height": rh},
        original_img.shape,
    )
    refined_center = [int(round(x1 + centroids[best_idx][0])), int(round(y1 + centroids[best_idx][1]))]
    return refined_bbox, refined_center


def project_and_refine_detections_to_original(detections, original_img, warp_meta, crop_bbox, cloth_model):
    projected = []

    for det in detections:
        if "_candidate_contour" in det:
            top_contour = det["_candidate_contour"]
        else:
            bx = det["bbox"]["x"]
            by = det["bbox"]["y"]
            bw = det["bbox"]["width"]
            bh = det["bbox"]["height"]
            top_contour = np.array([
                [[bx, by]],
                [[bx + bw - 1, by]],
                [[bx + bw - 1, by + bh - 1]],
                [[bx, by + bh - 1]],
            ], dtype=np.float32)

        projected_contour = project_points_top_view_to_original(
            top_contour, warp_meta, crop_bbox, original_img.shape
        )
        if projected_contour is None:
            clean = {k: v for k, v in det.items() if not k.startswith("_")}
            projected.append(clean)
            continue

        px, py, pw, ph = cv2.boundingRect(projected_contour.astype(np.float32))
        approx_bbox = {"x": int(px), "y": int(py), "width": int(pw), "height": int(ph)}

        projected_center = project_points_top_view_to_original(
            np.array([[det["center"]]], dtype=np.float32), warp_meta, crop_bbox, original_img.shape
        )
        if projected_center is None:
            approx_center = [det["center"][0], det["center"][1]]
        else:
            approx_center = [
                float(projected_center[0, 0, 0]),
                float(projected_center[0, 0, 1]),
            ]

        refined_bbox, refined_center = refine_bbox_on_original(
            original_img, approx_bbox, approx_center, cloth_model
        )

        clean = {k: v for k, v in det.items() if not k.startswith("_")}
        clean["bbox"] = refined_bbox
        clean["center"] = refined_center
        clean["bbox_source"] = "original_refined"
        projected.append(clean)

    final = []
    for det in sorted(projected, key=lambda d: (d["bbox"]["y"], d["bbox"]["x"])):
        duplicate = False
        for prev in final:
            if center_distance(det, prev) < 16:
                duplicate = True
                break
        if not duplicate:
            final.append(det)

    return final


# Ball detection / classification on the ORIGINAL image
def hue_circular_diff(h, ref_h):
    h = h.astype(np.int16)
    d = np.abs(h - int(ref_h))
    return np.minimum(d, 180 - d)


def get_original_surface_stats(original_img, cloth_model):
    if cloth_model is None:
        return None

    cloth_mask = cloth_model["cloth_mask"]
    filled_surface = cloth_model["filled_surface"]
    if cloth_mask is None or filled_surface is None or cv2.countNonZero(filled_surface) == 0:
        return None

    H, W = original_img.shape[:2]
    hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    dist_to_border = cv2.distanceTransform(filled_surface, cv2.DIST_L2, 5)
    inner_margin = max(2.0, 0.004 * min(H, W))
    safe_inner = (dist_to_border > inner_margin).astype(np.uint8) * 255
    if cv2.countNonZero(safe_inner) == 0:
        safe_inner = filled_surface.copy()

    reference_mask = (safe_inner > 0) & (cloth_mask > 0)
    if np.count_nonzero(reference_mask) == 0:
        reference_mask = cloth_mask > 0
    if np.count_nonzero(reference_mask) == 0:
        reference_mask = filled_surface > 0
    if np.count_nonzero(reference_mask) == 0:
        return None

    cloth_lab = np.median(lab[reference_mask], axis=0)
    cloth_h = int(np.median(hsv[:, :, 0][reference_mask]))
    cloth_s = int(np.median(hsv[:, :, 1][reference_mask]))
    cloth_v = int(np.median(hsv[:, :, 2][reference_mask]))

    return {
        "cloth_mask": cloth_mask,
        "filled_surface": filled_surface,
        "safe_inner": safe_inner,
        "dist_to_border": dist_to_border,
        "cloth_lab": cloth_lab,
        "cloth_h": cloth_h,
        "cloth_s": cloth_s,
        "cloth_v": cloth_v,
        "hsv": hsv,
        "lab": lab,
    }


def detect_ball_candidates_original(original_img, cloth_model):
    stats = get_original_surface_stats(original_img, cloth_model)
    if stats is None:
        return [], None

    hsv = stats["hsv"]
    lab = stats["lab"]
    filled_surface = stats["filled_surface"]
    safe_inner = stats["safe_inner"]
    cloth_lab = stats["cloth_lab"]
    cloth_h = stats["cloth_h"]
    cloth_s = stats["cloth_s"]
    cloth_v = stats["cloth_v"]
    dist_to_border = stats["dist_to_border"]

    candidate_region = filled_surface > 0
    reference_region = safe_inner > 0
    if np.count_nonzero(reference_region) == 0:
        reference_region = candidate_region
    if np.count_nonzero(reference_region) == 0:
        return [], cloth_h

    H, W = original_img.shape[:2]
    table_area = max(1, int(np.count_nonzero(candidate_region)))

    delta_lab = np.linalg.norm(lab - cloth_lab, axis=2)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue_delta = hue_circular_diff(hue, cloth_h)

    delta_values = delta_lab[reference_region]
    if delta_values.size == 0:
        return [], cloth_h

    adaptive_thr = max(10.0, float(np.percentile(delta_values, 91)))
    color_thr = max(45, int(0.45 * cloth_s))

    delta_mask = (delta_lab > adaptive_thr) & candidate_region
    white_mask = candidate_region & (sat < max(68, int(0.55 * cloth_s))) & (val > max(145, cloth_v + 8)) & (delta_lab > 6.0)
    black_mask = candidate_region & (val < min(85, max(55, cloth_v - 25))) & (delta_lab > 6.0)
    vivid_mask = candidate_region & (sat > color_thr) & ((delta_lab > max(7.0, adaptive_thr * 0.65)) | (hue_delta > 10))

    candidate_mask = (delta_mask | white_mask | black_mask | vivid_mask).astype(np.uint8) * 255
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )

    num, labels, cc_stats, centroids = cv2.connectedComponentsWithStats(candidate_mask, 8)
    min_area = max(90, int(0.00005 * table_area))
    max_area = int(0.020 * table_area)

    detections = []
    for idx in range(1, num):
        x, y, w, h, area = cc_stats[idx]
        if area < min_area or area > max_area:
            continue

        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio > 5.5:
            continue
        if max(w, h) < 12:
            continue

        cx, cy = centroids[idx]
        cx_i = int(np.clip(round(cx), 0, W - 1))
        cy_i = int(np.clip(round(cy), 0, H - 1))
        if not candidate_region[cy_i, cx_i]:
            continue
        if dist_to_border[cy_i, cx_i] < 1.0 and area < (1.7 * min_area):
            continue

        det = component_to_detection(labels, cc_stats, centroids, idx, source="original")

        duplicate = False
        for prev in detections:
            if bbox_iou(det, prev) > 0.18 or center_distance(det, prev) < 14:
                duplicate = True
                break
        if not duplicate:
            detections.append(det)

    return detections, stats


def classify_ball_candidate_original(original_img, det, surface_stats):
    if surface_stats is None:
        return None

    H, W = original_img.shape[:2]
    x = det["bbox"]["x"]
    y = det["bbox"]["y"]
    w = det["bbox"]["width"]
    h = det["bbox"]["height"]

    pad = max(2, int(0.12 * max(w, h)))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)

    roi = original_img[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    local_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    shifted_contour = det["contour"].copy()
    shifted_contour[:, 0, 0] -= x1
    shifted_contour[:, 0, 1] -= y1
    cv2.drawContours(local_mask, [shifted_contour], -1, 255, thickness=-1)

    local_mask = cv2.morphologyEx(
        local_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    dist = cv2.distanceTransform(local_mask, cv2.DIST_L2, 5)
    core_thresh = max(1.0, 0.08 * max(w, h))
    core_mask = (dist >= core_thresh).astype(np.uint8) * 255
    if cv2.countNonZero(core_mask) < 25:
        core_mask = local_mask.copy()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta_lab = np.linalg.norm(lab - surface_stats["cloth_lab"], axis=2)

    mask_bool = core_mask > 0
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    local_values = delta_lab[mask_bool]
    local_thr = max(6.0, float(np.percentile(local_values, 35))) if local_values.size > 0 else 6.0
    strong_diff = mask_bool & (delta_lab >= local_thr)
    if np.count_nonzero(strong_diff) < 20:
        strong_diff = mask_bool

    white_mask = strong_diff & (sat < max(65, int(0.55 * surface_stats["cloth_s"]))) & (val > max(145, surface_stats["cloth_v"] + 8))
    black_mask = strong_diff & (val < min(82, max(50, surface_stats["cloth_v"] - 22)))
    colored_mask = strong_diff & (~white_mask) & (~black_mask) & (sat > 45)

    valid_pixels = max(1, int(np.count_nonzero(mask_bool)))
    white_ratio = np.count_nonzero(white_mask) / valid_pixels
    black_ratio = np.count_nonzero(black_mask) / valid_pixels
    colored_ratio = np.count_nonzero(colored_mask) / valid_pixels

    if np.count_nonzero(colored_mask) < 18 and white_ratio < 0.22 and black_ratio < 0.22:
        return None

    if white_ratio > 0.62 and colored_ratio < 0.22:
        number = 0
        ball_type = "cue"
        color_name = "white"
    elif black_ratio > 0.48 and colored_ratio < 0.22:
        number = 8
        ball_type = "eight"
        color_name = "black"
    elif np.count_nonzero(colored_mask) >= 18:
        bright_threshold = np.percentile(val[colored_mask], 20)
        bright_colored = colored_mask & (val >= bright_threshold)
        if np.count_nonzero(bright_colored) < 10:
            bright_colored = colored_mask

        mean_lab = np.mean(lab[bright_colored], axis=0).astype(np.float32)
        distances = {
            name: float(np.linalg.norm(mean_lab - ref_lab))
            for name, ref_lab in REFERENCE_LAB_COLORS.items()
        }
        color_name = min(distances, key=distances.get)

        is_stripe = white_ratio > 0.15
        number = BALL_NUMBER_MAP[color_name][1] if is_stripe else BALL_NUMBER_MAP[color_name][0]
        ball_type = "stripe" if is_stripe else "solid"
    else:
        return None

    tight_x, tight_y, tight_w, tight_h = cv2.boundingRect(shifted_contour)
    tight_x1 = max(0, x1 + tight_x)
    tight_y1 = max(0, y1 + tight_y)
    tight_x2 = min(W, tight_x1 + tight_w)
    tight_y2 = min(H, tight_y1 + tight_h)

    return {
        "number": int(number),
        "type": ball_type,
        "color": color_name,
        "bbox": {
            "x": int(tight_x1),
            "y": int(tight_y1),
            "width": int(tight_x2 - tight_x1),
            "height": int(tight_y2 - tight_y1),
        },
        "center": [int(round(det["center"][0])), int(round(det["center"][1]))],
        "source": det["source"],
        "white_ratio": round(float(white_ratio), 3),
        "black_ratio": round(float(black_ratio), 3),
        "colored_ratio": round(float(colored_ratio), 3),
    }


def detect_and_classify_balls_original(original_img, cloth_model, keep_internal=False):
    candidates, surface_stats = detect_ball_candidates_original(original_img, cloth_model)

    detections = []
    for det in candidates:
        classified = classify_ball_candidate_original(original_img, det, surface_stats)
        if classified is not None:
            if keep_internal:
                classified["_candidate_bbox"] = {
                    "x": int(det["bbox"]["x"]),
                    "y": int(det["bbox"]["y"]),
                    "width": int(det["bbox"]["width"]),
                    "height": int(det["bbox"]["height"]),
                }
            detections.append(classified)

    final = []
    for det in sorted(detections, key=lambda d: (d["bbox"]["y"], d["bbox"]["x"])):
        duplicate = False
        for prev in final:
            if center_distance(det, prev) < 14:
                duplicate = True
                break
        if not duplicate:
            final.append(det)

    return final


def draw_ball_detections(img, detections):
    out = img.copy()
    for det in detections:
        x = det["bbox"]["x"]
        y = det["bbox"]["y"]
        w = det["bbox"]["width"]
        h = det["bbox"]["height"]
        number = det["number"]
        cx, cy = det["center"]

        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 2, (255, 255, 255), -1)

        text = str(number)
        text_pos = (x, max(20, y - 8))
        cv2.putText(out, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return out

def save_results_json(results, json_path):
    serializable = []
    for item in results:
        serializable.append({
            "image": item["image"],
            "total_balls": item["total_balls"],
            "balls": item["balls"],
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


# Pipeline

# v4 robust classifier

def estimate_cloth_lab_global(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    cloth_m = (
        (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 115) &
        (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 80)
    )
    if np.count_nonzero(cloth_m) > 1000:
        return np.median(lab[cloth_m], axis=0).astype(np.float32)
    return None

def neutralize_image(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    cloth_m = (
        (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 115) &
        (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 80)
    )
    if np.count_nonzero(cloth_m) < 500:
        return img_bgr.copy()
    cloth_lab = np.median(lab[cloth_m], axis=0)
    lab[:, :, 1] -= (cloth_lab[1] - 128)
    lab[:, :, 2] -= (cloth_lab[2] - 128)
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


BALL_SIG_LAB_AB = {
    # (number, is_stripe): (a, b)
    "yellow_solid":  (138, 193),
    "yellow_stripe": (138, 193),
    "blue_solid":    (72, 138),
    "blue_stripe":   (72, 138),
    "red_solid":     (160, 175),
    "red_stripe":    (160, 175),
    "purple_solid":  (135, 170),
    "purple_stripe": (135, 170),
    "orange_solid":  (147, 193),
    "orange_stripe": (147, 193),
    "green_solid":   (108, 125),
    "green_stripe":  (108, 125),
    "maroon_solid":  (136, 180),
    "maroon_stripe": (136, 180),
}

COLOR_TO_NUMBER_V7 = {
    "yellow": (1, 9),  "blue":   (2, 10), "red":    (3, 11),
    "purple": (4, 12), "orange": (5, 13), "green":  (6, 14),
    "maroon": (7, 15),
}


def _classify_by_neutralized_lab(a_n, b_n, v_n, L_n):
    if a_n < 100:
        return "blue"
    # GREEN: a* moderately below neutral (~108, b* ~125)
    if a_n < 120:
        return "green"
    # Nearest-neighbor in (a*, b*) for remaining warm colours
    refs = {
        "yellow":  (138, 193),
        "red":     (160, 175),
        "purple":  (135, 170),
        "orange":  (147, 193),
        "maroon":  (136, 180),
    }
    best = min(refs, key=lambda k: (a_n - refs[k][0]) ** 2 + (b_n - refs[k][1]) ** 2)
    if best == "purple" and v_n < 90 and a_n < 130:
        return "blue"
    return best


def recenter_ball(img_bgr, cx, cy, r, cloth_lab):
    if cloth_lab is None:
        return cx, cy, r

    H, W = img_bgr.shape[:2]
    search_pad = int(r * 1.15)
    x1, y1 = max(0, cx - search_pad), max(0, cy - search_pad)
    x2, y2 = min(W, cx + search_pad + 1), min(H, cy + search_pad + 1)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return cx, cy, r

    roi = img_bgr[y1:y2, x1:x2]
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = np.linalg.norm(roi_lab - cloth_lab, axis=2)
    fg = (delta > 25).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, 8)
    if num <= 1:
        return cx, cy, r

    lx, ly = cx - x1, cy - y1
    min_area = max(20, int(np.pi * (0.45 * r) ** 2))
    max_area = int(np.pi * (1.4 * r) ** 2)

    best_idx = -1
    best_score = float("inf")
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        ccx, ccy = centroids[i]
        d2 = (ccx - lx) ** 2 + (ccy - ly) ** 2
        if d2 < best_score:
            best_score = d2
            best_idx = i

    if best_idx == -1:
        return cx, cy, r

    ccx, ccy = centroids[best_idx]
    new_cx = int(round(ccx + x1))
    new_cy = int(round(ccy + y1))

    shift2 = (new_cx - cx) ** 2 + (new_cy - cy) ** 2
    if shift2 > (0.4 * r) ** 2:
        return cx, cy, r

    return new_cx, new_cy, r


def _classify_stripe_vs_solid(white_mask_2d, disk_full_2d, wr, cr, r):
    if wr < 0.06:
        return 'solid'
    if wr > 0.22:
        return 'stripe'

    white_u8 = (white_mask_2d.astype(np.uint8)) * 255
    # Morphological close to connect gaps in the stripe band
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white_closed = cv2.morphologyEx(white_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(white_closed, 8)
    total_white = int(np.count_nonzero(white_closed)) or 1
    if num <= 1:
        return 'solid'

    largest_area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    largest_ratio = largest_area / max(1, total_white)

    disk_area = max(1, int(np.count_nonzero(disk_full_2d)))
    largest_frac_of_disk = largest_area / disk_area

    if largest_ratio >= 0.55 and largest_frac_of_disk >= 0.08:
        return 'stripe'

    big_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    w_blob = stats[big_idx, cv2.CC_STAT_WIDTH]
    h_blob = stats[big_idx, cv2.CC_STAT_HEIGHT]
    if w_blob > 0 and h_blob > 0:
        ar = max(w_blob, h_blob) / max(1, min(w_blob, h_blob))
        if ar >= 1.8 and largest_frac_of_disk >= 0.06:
            return 'stripe'

    return 'solid'


def classify_ball_final(img_bgr, cx, cy, r, cloth_lab=None, neu_img=None):
    H, W = img_bgr.shape[:2]

    if cloth_lab is not None:
        cx, cy, r = recenter_ball(img_bgr, cx, cy, r, cloth_lab)

    pad = int(r * 1.15)
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(W, cx + pad + 1), min(H, cy + pad + 1)

    roi_bgr = img_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return None, None, False, {"path": "rejected_empty"}

    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.int32)

    hH, hW = roi_hsv.shape[:2]
    yy, xx = np.ogrid[:hH, :hW]
    lx, ly = cx - x1, cy - y1

    disk_inner = (xx - lx) ** 2 + (yy - ly) ** 2 <= (0.60 * r) ** 2
    disk_mid   = (xx - lx) ** 2 + (yy - ly) ** 2 <= (0.80 * r) ** 2
    disk_full  = (xx - lx) ** 2 + (yy - ly) ** 2 <= (0.95 * r) ** 2

    if np.count_nonzero(disk_full) < 15 or np.count_nonzero(disk_inner) < 6:
        return None, None, False, {"path": "rejected_tiny"}

    V_in = roi_hsv[:, :, 2][disk_inner].astype(np.int32)
    S_in = roi_hsv[:, :, 1][disk_inner].astype(np.int32)
    H_in = roi_hsv[:, :, 0][disk_inner].astype(np.int32)
    inner_total = V_in.size

    black_inner = (V_in < 85) & (S_in < 110)
    br_inner = np.count_nonzero(black_inner) / inner_total

    cloth_inner = (H_in >= 95) & (H_in <= 115) & (S_in >= 150) & (V_in >= 80)
    cloth_inner_ratio = np.count_nonzero(cloth_inner) / inner_total

    if br_inner >= 0.50 and cloth_inner_ratio < 0.25:
        return 8, "black", False, {
            "path": "inner_black",
            "br_inner": round(br_inner, 3),
            "cloth_inner_ratio": round(cloth_inner_ratio, 3),
        }

    white_inner = (V_in > 160) & (S_in < 70)
    wr_inner = np.count_nonzero(white_inner) / inner_total
    if wr_inner >= 0.50 and np.median(S_in) < 100 and cloth_inner_ratio < 0.25:
        return 0, "white", False, {
            "path": "inner_white",
            "wr_inner": round(wr_inner, 3),
            "median_S_in": int(np.median(S_in)),
            "cloth_inner_ratio": round(cloth_inner_ratio, 3),
        }

    if cloth_lab is None:
        ring = ((xx - lx) ** 2 + (yy - ly) ** 2 <= (1.25 * r) ** 2) & ~disk_full
        if np.count_nonzero(ring) >= 10:
            cloth_lab = np.median(roi_lab[ring], axis=0)
        else:
            cloth_lab = np.array([120, 128, 100])

    delta = np.linalg.norm(roi_lab - cloth_lab, axis=2)
    fg_mask = disk_full & (delta > 22)
    if np.count_nonzero(fg_mask) < 20:
        fg_mask = disk_full & (delta > 14)
        if np.count_nonzero(fg_mask) < 12:
            return None, None, False, {"path": "rejected_no_fg"}

    H_p = roi_hsv[:, :, 0][fg_mask].astype(np.int32)
    S_p = roi_hsv[:, :, 1][fg_mask].astype(np.int32)
    V_p = roi_hsv[:, :, 2][fg_mask].astype(np.int32)
    L_p = roi_lab[:, :, 0][fg_mask]
    a_p = roi_lab[:, :, 1][fg_mask]
    b_p = roi_lab[:, :, 2][fg_mask]
    total = H_p.size

    white_m = (S_p < 55) & (V_p > 160)
    black_m = (V_p < 65) & (S_p < 100)
    color_m = ~white_m & ~black_m & (S_p > 55)

    wr = np.count_nonzero(white_m) / total
    br = np.count_nonzero(black_m) / total
    cr = np.count_nonzero(color_m) / total

    # Fast fallbacks (very dominant white/black on fg)
    if wr >= 0.50 and cr < 0.30 and br < 0.10:
        return 0, "white", False, {
            "path": "fg_white",
            "wr": round(wr, 3), "cr": round(cr, 3), "br": round(br, 3),
        }
    if br >= 0.55 and cr < 0.20:
        return 8, "black", False, {
            "path": "fg_black",
            "br": round(br, 3), "cr": round(cr, 3),
        }

    if np.count_nonzero(color_m) < 12:
        if wr > br and wr > 0.30:
            return 0, "white", False, {"path": "fallback_white", "wr": round(wr, 3)}
        if br > 0.30:
            return 8, "black", False, {"path": "fallback_black", "br": round(br, 3)}
        return None, None, False, {
            "path": "rejected_insufficient_color",
            "wr": round(wr, 3), "br": round(br, 3), "cr": round(cr, 3),
        }

    s_c = S_p[color_m]
    v_c = V_p[color_m]
    h_c = H_p[color_m]
    a_c = a_p[color_m]
    b_c = b_p[color_m]
    L_c = L_p[color_m]

    if len(s_c) >= 25:
        thr = np.percentile(s_c, 60)
        sel = s_c >= thr
        if np.count_nonzero(sel) < 8:
            sel = np.ones_like(s_c, dtype=bool)
    else:
        sel = np.ones_like(s_c, dtype=bool)

    h_sel = h_c[sel]
    s_sel = s_c[sel]
    v_sel = v_c[sel]
    a_sel = a_c[sel]
    b_sel = b_c[sel]
    L_sel = L_c[sel]

    # Red wrap-around
    if np.percentile(h_sel, 10) < 5 and np.percentile(h_sel, 90) > 170:
        h_adj = np.where(h_sel > 90, h_sel.astype(np.int32) - 180, h_sel.astype(np.int32))
        mh = int(np.median(h_adj)) % 180
    else:
        mh = int(np.median(h_sel))
    ms = int(np.median(s_sel))
    mv = int(np.median(v_sel))
    mL = int(np.median(L_sel))
    ma = int(np.median(a_sel))
    mb = int(np.median(b_sel))

    V_mid = roi_hsv[:, :, 2][disk_mid].astype(np.int32)
    v_mid_p90 = int(np.percentile(V_mid, 90)) if V_mid.size > 0 else 0
    s_mid_med = int(np.median(roi_hsv[:, :, 1][disk_mid])) if V_mid.size > 0 else 0

    is_suspect_8 = (
        mv <= 80
        and 95 <= mh <= 115
        and 100 <= mb <= 128
        and cr >= 0.35
        and wr < 0.22
    )
    V_mid_std = float(np.std(V_mid)) if V_mid.size > 0 else 0
    if is_suspect_8 and V_mid_std > 55 and v_mid_p90 >= 150:
        is_suspect_8 = False

    if is_suspect_8:
        return 8, "black", False, {
            "path": "post_filter_black",
            "mh": mh, "mv": mv, "mb": mb,
            "v_mid_p90": v_mid_p90, "s_mid_med": s_mid_med,
            "V_mid_std": round(V_mid_std, 1),
            "cr": round(cr, 3), "wr": round(wr, 3), "br": round(br, 3),
        }

    L_mid = roi_lab[:, :, 0][disk_mid]
    L_mid_med = int(np.median(L_mid)) if L_mid.size > 0 else 0
    is_suspect_cue = (
        L_mid_med >= 190
        and wr >= 0.30
        and s_mid_med < 70
        and mv >= 170
    )
    if is_suspect_cue:
        return 0, "white", False, {
            "path": "post_filter_white",
            "L_mid_med": L_mid_med, "s_mid_med": s_mid_med,
            "wr": round(wr, 3), "mv": mv,
        }

    if (
        55 <= mh <= 75
        and mv >= 215
        and mb >= 185
        and wr < 0.15
        and cloth_inner_ratio >= 0.10
    ):
        return None, None, False, {
            "path": "rejected_fake_bright",
            "mh": mh, "mv": mv, "mb": mb, "ms": ms, "wr": round(wr, 3),
            "cloth_inner_ratio": round(cloth_inner_ratio, 3),
        }

    if (
        55 <= mh <= 75
        and mv >= 215
        and mb >= 185
        and ma < 120
        and ms >= 180
        and wr < 0.30
    ):
        return None, None, False, {
            "path": "rejected_cushion_bright",
            "mh": mh, "mv": mv, "ma": ma, "mb": mb, "ms": ms,
        }

    if cloth_inner_ratio >= 0.30 and ms < 140 and np.count_nonzero(fg_mask) < (0.45 * np.count_nonzero(disk_full)):
        return None, None, False, {
            "path": "rejected_mostly_felt",
            "cloth_inner_ratio": round(cloth_inner_ratio, 3),
            "ms": ms, "fg_frac": round(np.count_nonzero(fg_mask) / max(1, np.count_nonzero(disk_full)), 3),
        }

    white_mask_2d = np.zeros(roi_hsv.shape[:2], dtype=bool)
    white_pixel_criterion = (
        (roi_hsv[:, :, 1] < 55) & (roi_hsv[:, :, 2] > 160)
    )
    white_mask_2d = disk_full & white_pixel_criterion

    stripe_label = _classify_stripe_vs_solid(white_mask_2d, disk_full, wr, cr, r)
    is_stripe = (stripe_label == 'stripe')

    neu_features = None
    if neu_img is not None:
        neu_roi_bgr = neu_img[y1:y2, x1:x2]
        if neu_roi_bgr.size > 0:
            neu_lab = cv2.cvtColor(neu_roi_bgr, cv2.COLOR_BGR2LAB).astype(np.int32)
            neu_hsv = cv2.cvtColor(neu_roi_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
            disk_sel = disk_mid & (~white_mask_2d)
            if np.count_nonzero(disk_sel) >= 10:
                L_nv = neu_lab[:, :, 0][disk_sel]
                a_nv = neu_lab[:, :, 1][disk_sel]
                b_nv = neu_lab[:, :, 2][disk_sel]
                V_nv = neu_hsv[:, :, 2][disk_sel]
                S_nv = neu_hsv[:, :, 1][disk_sel]
                ok = (V_nv > 55) & (V_nv < 230) & (S_nv > 50)
                if np.count_nonzero(ok) >= 8:
                    neu_L = int(np.median(L_nv[ok]))
                    neu_a = int(np.median(a_nv[ok]))
                    neu_b = int(np.median(b_nv[ok]))
                    neu_V = int(np.median(V_nv[ok]))
                    neu_S = int(np.median(S_nv[ok]))
                    neu_features = (neu_L, neu_a, neu_b, neu_V, neu_S)

    color_name = None

    if neu_features is not None:
        neu_L, neu_a, neu_b, neu_V, neu_S = neu_features
        color_name = _classify_by_neutralized_lab(neu_a, neu_b, neu_V, neu_L)
    else:
        neu_L = neu_a = neu_b = neu_V = neu_S = -1

    if color_name is None:
        # RED (3/11) -- low hue, highly saturated, strong LAB a*
        if (mh <= 8 or mh >= 172) and ms >= 140 and ma >= 150:
            color_name = "red"
        # ORANGE (5/13) -- hue 8-17, saturated and bright
        elif 8 <= mh <= 17 and ms >= 130 and mv >= 165:
            color_name = "orange"
        # MAROON dark (7/15) -- low hue but dark
        elif mh <= 16 and mv < 165:
            color_name = "maroon"
        # YELLOW (1/9) -- hue 16-35, bright and saturated, b* high
        elif 16 <= mh <= 35 and mv >= 165 and mb >= 155 and ms >= 130:
            color_name = "yellow"
        # MAROON faded / yellow fallback in hue 16-35
        elif 16 <= mh <= 35:
            if is_stripe and mb >= 165 and mv >= 170:
                color_name = "yellow"
            else:
                color_name = "maroon"
        # GREEN (6/14)
        elif 35 <= mh <= 95 and ms >= 55:
            if mb >= 155 and ms < 100:
                color_name = "maroon"
            elif mh <= 70 and mv >= 200 and mb >= 175:
                if mh >= 30:
                    color_name = "yellow" if mb >= 180 else "maroon"
                else:
                    color_name = "maroon"
            else:
                color_name = "green"
        # BLUE vs PURPLE: blue has lower b* (cooler) than purple
        elif 95 <= mh <= 145:
            if mb <= 108:
                color_name = "blue"
            elif mb >= 115:
                color_name = "purple"
            else:
                # Ambiguous b* range: use saturation and brightness
                if mv < 115 or ms >= 195:
                    color_name = "blue"
                else:
                    color_name = "purple"
        else:
            REFS_LAB = {
                "yellow": (180, 140, 180), "blue":   (70, 130,  98),
                "red":    (95, 170, 150),  "purple": (80, 130, 108),
                "orange": (135, 155, 165), "green":  (115, 108, 135),
                "maroon": (125, 140, 150),
            }
            mean_lab = np.array([mL, ma, mb], dtype=np.float32)
            dists = {n: float(np.linalg.norm(mean_lab - np.array(ref))) for n, ref in REFS_LAB.items()}
            color_name = min(dists, key=dists.get)

    dbg = {
        "path": "color_classified",
        "mh": mh, "ms": ms, "mv": mv,
        "mL": mL, "ma": ma, "mb": mb,
        "neu_L": neu_L, "neu_a": neu_a, "neu_b": neu_b, "neu_V": neu_V, "neu_S": neu_S,
        "v_mid_p90": v_mid_p90, "s_mid_med": s_mid_med,
        "wr": round(wr, 3), "br": round(br, 3), "cr": round(cr, 3),
        "stripe_label": stripe_label,
        "is_stripe": is_stripe,
        "color_name": color_name,
        "recentered": True,
    }

    number = BALL_NUMBER_MAP[color_name][1 if is_stripe else 0]
    return int(number), color_name, is_stripe, dbg


def _detect_balls_on_original(img, blue_mask):
    H, W = img.shape[:2]

    filled = np.zeros_like(blue_mask)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)

    ys, xs = np.where(filled > 0)
    if len(xs) == 0:
        return []
    table_w = int(xs.max() - xs.min())
    min_r = max(10, int(table_w * 0.012))
    max_r = max(min_r + 5, int(table_w * 0.025))

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    cloth_lab_local = np.median(lab_img[filled > 0], axis=0)
    delta_lab = np.linalg.norm(lab_img - cloth_lab_local, axis=2)

    gray_blur = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (7, 7), 1.5)
    all_raw = []
    for p1, p2, dp in [(50, 12, 1.0), (60, 15, 1.0), (40, 10, 1.1),
                       (70, 18, 1.0), (50, 10, 1.2), (45, 11, 1.0)]:
        circles = cv2.HoughCircles(
            gray_blur, cv2.HOUGH_GRADIENT, dp=dp,
            minDist=int(1.8 * min_r), param1=p1, param2=p2,
            minRadius=min_r, maxRadius=max_r,
        )
        if circles is not None:
            for x, y, r in np.round(circles[0]).astype(int):
                if 0 <= y < H and 0 <= x < W and filled[y, x] > 0:
                    all_raw.append((int(x), int(y), int(r)))

    scored = []
    for x, y, r in all_raw:
        yy, xx = np.ogrid[:H, :W]
        disk_in = ((xx - x) ** 2 + (yy - y) ** 2 <= (0.8 * r) ** 2) & (filled > 0)
        if np.count_nonzero(disk_in) < 10:
            continue
        scored.append((float(np.mean(delta_lab[disk_in])), x, y, r))

    scored.sort(key=lambda c: -c[0])
    detected = []
    for sc, x, y, r in scored:
        if sc < 40:
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < (max(r, pr) * 1.6) ** 2
               for _, px, py, pr in detected):
            continue
        detected.append((sc, x, y, r))

    cloth_lab_global = estimate_cloth_lab_global(img)
    if cloth_lab_global is None:
        cloth_lab_global = cloth_lab_local.astype(np.float32)

    neu_img = neutralize_image(img)

    balls = []
    for sc, x, y, r in detected:
        # Recenter for classification AND for bbox
        nx, ny, nr = recenter_ball(img, x, y, r, cloth_lab_global)
        num, color_name, is_stripe, dbg = classify_ball_final(
            img, nx, ny, nr, cloth_lab=cloth_lab_global, neu_img=neu_img
        )
        if num is None:
            continue
        balls.append({
            "number": int(num),
            "color": color_name,
            "stripe": bool(is_stripe),
            "bbox": {
                "x": int(nx - nr), "y": int(ny - nr),
                "width": int(2 * nr), "height": int(2 * nr),
            },
            "center": [int(nx), int(ny)],
            "radius": int(nr),
            "delta_score": round(sc, 2),
        })

    final = []
    for b in sorted(balls, key=lambda x: -x.get("delta_score", 0)):
        dup = False
        for kept in final:
            dx = b["center"][0] - kept["center"][0]
            dy = b["center"][1] - kept["center"][1]
            if dx * dx + dy * dy < (max(b["radius"], kept["radius"]) * 1.2) ** 2:
                dup = True
                break
        if not dup:
            final.append(b)

    return final


def draw_ball_detections_normalized(img, balls, W, H):
    out = img.copy()
    for b in balls:
        x1 = int(b["xmin"] * W); y1 = int(b["ymin"] * H)
        x2 = int(b["xmax"] * W); y2 = int(b["ymax"] * H)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, str(b["number"]), (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, str(b["number"]), (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def run_pipeline(image_path, display=False):
    img = load_image(image_path)
    H, W = img.shape[:2]

    labels, shape, K = kmeans_cluster_refined(img, 5)
    masks = get_cluster_masks(labels, shape, K)
    blue_idx = find_blue_cluster(img, labels, K)

    if blue_idx == -1:
        return {"image_path": image_path, "num_balls": 0, "balls": [], "top_view": None, "img": img}

    blue_mask = masks[blue_idx]
    balls = _detect_balls_on_original(img, blue_mask)

    output_balls = []
    for b in balls:
        bx, by, bw, bh = b["bbox"]["x"], b["bbox"]["y"], b["bbox"]["width"], b["bbox"]["height"]
        output_balls.append({
            "number": b["number"],
            "xmin": float(max(0, bx) / W),
            "xmax": float(min(W, bx + bw) / W),
            "ymin": float(max(0, by) / H),
            "ymax": float(min(H, by + bh) / H),
        })

    target = select_target_cluster(img, labels, masks, blue_idx)
    warped = warp_from_cluster(img, masks[target]) if target != -1 else None

    if warped is not None:
        top_view = extract_blue_cloth(warped, padding=50)
    else:
        top_view = extract_blue_cloth(img, padding=50)

    if display:
        show("Original", img)
        if top_view is not None:
            show("Top View", top_view)
        print(f"Total balls: {len(output_balls)}")
        for b in output_balls:
            print(f"  Ball {b['number']}")

    return {
        "image_path": image_path,
        "num_balls": len(output_balls),
        "balls": output_balls,
        "top_view": top_view,
        "img": img,
    }


def process_images(input_json, output_json, top_view_dir=None, display=False):
    with open(input_json, "r") as f:
        data = json.load(f)
    image_paths = data.get("image_path", [])

    if top_view_dir:
        os.makedirs(top_view_dir, exist_ok=True)

    results = []
    for path in image_paths:
        print(f"Processing {path}...")
        try:
            result = run_pipeline(path, display=display)
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {"image_path": path, "num_balls": 0, "balls": [], "top_view": None, "img": None}

        results.append({
            "image_path": result["image_path"],
            "num_balls": result["num_balls"],
            "balls": result["balls"],
        })

        name = os.path.basename(path)

        if top_view_dir and result.get("top_view") is not None:
            cv2.imwrite(os.path.join(top_view_dir, name), result["top_view"])

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} results to {output_json}")
    return results


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))

    process_images(
        input_json=os.path.join(BASE, "input.json"),
        output_json=os.path.join(BASE, "output.json"),
        top_view_dir=os.path.join(BASE, "top_views"),
    )