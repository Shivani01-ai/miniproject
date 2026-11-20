"""
Suspicious Region Localization Module
Provides two complementary detection strategies:
1. Contextual Multi-Evidence (Enhanced):
   Combines local area averaging with contextual surrounding-annulus analysis,
   foreground population consistency, and aspect ratio awareness to suppress
   ordinary printed text lines without missing altered patches.
2. Baseline Area-Average (Legacy):
   Preserves the original area-average heuristic for baseline comparison.
"""

from typing import List, Dict, Tuple, Any
import io
import cv2
import numpy as np
from PIL import Image, ImageChops


def compute_ela_difference_array(image: Image.Image, quality: int = 90) -> np.ndarray:
    """
    Computes the raw absolute pixel difference array between the original image
    and its JPEG recompressed version.

    Args:
        image: Original PIL Image.
        quality: JPEG compression quality (default 90).

    Returns:
        np.ndarray: Grayscale absolute difference array as float32.
    """
    orig_rgb = image.convert("RGB")
    orig_np = np.array(orig_rgb)

    buffer = io.BytesIO()
    orig_rgb.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)
    recomp_np = np.array(recompressed)

    diff_arr = np.abs(orig_np.astype(np.float32) - recomp_np.astype(np.float32))
    gray_diff = np.mean(diff_arr, axis=2).astype(np.float32)
    return gray_diff


def merge_proximity_boxes(boxes: List[List[int]], distance: int = 25) -> List[List[int]]:
    """
    Iteratively merges bounding boxes that overlap or are within a given pixel distance.
    This prevents ordinary text lines from being fragmented into dozens of tiny disjoint boxes.

    Args:
        boxes: List of bounding boxes [x1, y1, x2, y2].
        distance: Maximum pixel distance to trigger a merge.

    Returns:
        List[List[int]]: Merged bounding boxes.
    """
    merged = boxes[:]
    for _ in range(3):  # Multiple passes for transitive chaining
        result = []
        while merged:
            curr = merged.pop(0)
            did_merge = False
            for i, other in enumerate(result):
                x_near = not (curr[0] - distance > other[2] or curr[2] + distance < other[0])
                y_near = not (curr[1] - distance > other[3] or curr[3] + distance < other[1])
                if x_near and y_near:
                    result[i] = [
                        min(curr[0], other[0]),
                        min(curr[1], other[1]),
                        max(curr[2], other[2]),
                        max(curr[3], other[3])
                    ]
                    did_merge = True
                    break
            if not did_merge:
                result.append(curr)
        merged = result
    return merged


def locate_suspicious_regions(
    image: Image.Image,
    quality: int = 90,
    sensitivity: float = 2.6,
    min_area: int = 300,
    merge_distance: int = 25,
    suppress_grid_lines: bool = True,
    method: str = "contextual"
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Identifies candidate regions requiring review using local adaptive ELA analysis.

    Supports two detection modes:
    - "contextual" (default): Evaluates local contrast against surrounding background,
      analyzes foreground population consistency, and filters ordinary 1D text strips
      that behave consistently with legitimate document print.
    - "baseline": Preserves the previous area-average algorithm for comparative benchmarking.

    Args:
        image: Original PIL Image.
        quality: JPEG reference quality for ELA recompression (default 90).
        sensitivity: Statistical threshold multiplier (k-factor above local mean, default 2.6).
        min_area: Minimum candidate region area in square pixels (default 300).
        merge_distance: Proximity threshold in pixels to merge adjacent candidate boxes (default 25).
        suppress_grid_lines: Whether to filter out thin, elongated table borders and underlines (default True).
        method: Localization strategy ("contextual" or "baseline").

    Returns:
        Tuple[Image.Image, List[Dict[str, Any]]]:
            - Marked-up PIL Image with bounding boxes drawn around candidate regions.
            - List of detected candidate regions with coordinates, dimensions, and anomaly impact scores.
    """
    orig_rgb = image.convert("RGB")
    orig_np = np.array(orig_rgb)
    img_h, img_w, _ = orig_np.shape

    # 1. Compute raw float32 ELA difference signal
    gray_diff = compute_ela_difference_array(orig_rgb, quality=quality)

    # 2. Adaptive local spatial smoothing (21x21 neighborhood)
    kernel_size = max(15, int(min(img_w, img_h) * 0.025)) | 1  # ensure odd size
    local_avg = cv2.blur(gray_diff, (kernel_size, kernel_size))

    global_mean = float(np.mean(local_avg))
    global_std = float(np.std(local_avg))

    # Safe statistical threshold: local average must exceed global baseline by sensitivity * std
    if global_std > 0.01:
        threshold_val = global_mean + (float(sensitivity) * global_std)
    else:
        threshold_val = global_mean + 10.0

    binary_mask = (local_avg >= threshold_val).astype(np.uint8) * 255

    # 3. Morphological bridge & cleanup
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

    # 4. Find candidate contours
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Ignore outer document margin borders
        if w > 0.88 * img_w or h > 0.88 * img_h:
            continue

        # Suppress long thin horizontal or vertical grid lines if enabled
        if suppress_grid_lines:
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > 6.0 and min(w, h) <= 10:
                continue

        if w * h >= min_area:
            raw_boxes.append([x, y, x + w, y + h])

    # 5. Merge nearby and overlapping boxes
    merged_boxes = merge_proximity_boxes(raw_boxes, distance=merge_distance)

    # 6. Extract candidate region measurements
    extracted_candidates = []
    for b in merged_boxes:
        x1, y1, x2, y2 = b
        w = x2 - x1
        h = y2 - y1
        area = int(w * h)

        if area < min_area:
            continue

        crop_diff = gray_diff[y1:y2, x1:x2]
        mean_err = float(np.mean(crop_diff)) if crop_diff.size > 0 else 0.0
        peak_err = float(np.max(crop_diff)) if crop_diff.size > 0 else 0.0
        aspect = max(w, h) / max(1, min(w, h))

        # Annulus (surrounding background window)
        pad = 15
        y1_p, y2_p = max(0, y1 - pad), min(img_h, y2 + pad)
        x1_p, x2_p = max(0, x1 - pad), min(img_w, x2 + pad)
        ann_mask = np.ones((y2_p - y1_p, x2_p - x1_p), dtype=bool)
        ann_mask[y1 - y1_p : y1 - y1_p + h, x1 - x1_p : x1 - x1_p + w] = False
        ann_vals = gray_diff[y1_p:y2_p, x1_p:x2_p][ann_mask]
        annulus_mean = float(np.mean(ann_vals)) if ann_vals.size else 0.1
        local_contrast = (mean_err - annulus_mean) / max(0.1, annulus_mean)

        excess_error = max(0.0, mean_err - global_mean)
        impact_score = area * excess_error

        extracted_candidates.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "X": int(x1), "Y": int(y1),
            "Width": int(w), "Height": int(h),
            "Area (px)": area,
            "Mean Error": round(mean_err, 2),
            "Peak Error": round(peak_err, 2),
            "Aspect Ratio": round(aspect, 1),
            "Local Contrast": round(local_contrast, 2),
            "Anomaly Impact": round(impact_score, 1)
        })

    # 7. Apply Contextual Multi-Evidence Filtering (if enabled)
    if method == "contextual" and len(extracted_candidates) > 0:
        all_means = np.array([c["Mean Error"] for c in extracted_candidates], dtype=np.float32)
        fg_median = float(np.median(all_means))
        fg_mad = float(np.median(np.abs(all_means - fg_median)))
        if fg_mad < 0.1:
            fg_mad = 0.4  # safe minimum deviation scale

        final_candidates = []
        for cand in extracted_candidates:
            w = cand["Width"]
            h = cand["Height"]
            area = cand["Area (px)"]
            aspect = cand["Aspect Ratio"]
            z_fg = (cand["Mean Error"] - fg_median) / fg_mad

            # Geometry check: standard 1D text line strip (e.g. single heading or table line)
            is_text_strip = (aspect >= 4.5 and min(w, h) <= 22)

            # If it is a normal 1D text strip and its error is consistent with other text (|z_fg| < 1.6),
            # it reflects ordinary printed text and is safely suppressed
            if is_text_strip and abs(z_fg) < 1.6 and area < 4000:
                continue

            final_candidates.append(cand)
    else:
        final_candidates = extracted_candidates

    # Sort candidates by anomaly impact score in descending order
    final_candidates.sort(key=lambda r: r["Anomaly Impact"], reverse=True)

    # 8. Draw bounding boxes and region badges on marked-up image
    marked_np = orig_np.copy()
    for idx, reg in enumerate(final_candidates, start=1):
        x = reg["X"]
        y = reg["Y"]
        w = reg["Width"]
        h = reg["Height"]

        # Color: Red highlight for review
        cv2.rectangle(marked_np, (x, y), (x + w, y + h), (230, 45, 45), 2)

        # Region ID label badge
        label = f"#{idx}"
        cv2.putText(
            marked_np,
            label,
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 45, 45),
            1,
            cv2.LINE_AA
        )

        reg["Region"] = f"Region #{idx}"

    # Clean internal coordinate keys before returning
    for reg in final_candidates:
        reg.pop("x1", None)
        reg.pop("y1", None)
        reg.pop("x2", None)
        reg.pop("y2", None)

    marked_image = Image.fromarray(marked_np)
    return marked_image, final_candidates
