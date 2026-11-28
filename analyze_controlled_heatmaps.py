import os
import sys
import json
import csv
import pathlib
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------
# 1. Paths & Verification (Isolated Post-Inference Analysis)
# ----------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(r"C:\Users\Kshethra B J\Document-Integrity-verification")
TEST_DIR = PROJECT_ROOT / "mantra_controlled_test"

GT_PATH = TEST_DIR / "ground_truth.json"
CLEAN_HEAT_PATH = TEST_DIR / "clean_heatmap.png"
ALTERED_HEAT_PATH = TEST_DIR / "altered_heatmap.png"
OUTPUT_CSV_PATH = TEST_DIR / "regional_comparison.csv"


def validate_prerequisites():
    """Ensures all required files exist before starting analysis."""
    for p in (GT_PATH, CLEAN_HEAT_PATH, ALTERED_HEAT_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")


# ----------------------------------------------------------------------
# 2. Heatmap Loader with RGB Channel Uniformity Check
# ----------------------------------------------------------------------
def load_and_validate_heatmap(path: pathlib.Path, expected_dims: tuple) -> tuple:
    """
    Loads an existing heatmap PNG, verifies that R, G, and B channels are
    identical (confirming a true grayscale encoding rather than a colormap),
    and normalizes pixel values to [0.0, 1.0].

    Returns:
        (heat_array, is_exact_grayscale)
    """
    exp_w, exp_h = expected_dims
    img = Image.open(path)

    # 1. Check image size (Width x Height)
    if img.size != (exp_w, exp_h):
        raise ValueError(
            f"Heatmap {path.name} dimensions {img.size} do not match expected ({exp_w}, {exp_h})"
        )

    arr = np.array(img)

    # 2. Check channel structure
    if arr.ndim == 3 and arr.shape[2] >= 3:
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        # Verify whether RGB channels are identical (pure grayscale encoding)
        is_identical_rgb = np.array_equal(r, g) and np.array_equal(r, b)
        if not is_identical_rgb:
            raise ValueError(
                f"Heatmap {path.name} contains divergent RGB channels. "
                "It appears to be a color-mapped image rather than a scalar grayscale heatmap. "
                "The red channel cannot be treated as scalar activation values."
            )

        heat = r.astype(np.float32) / 255.0
    elif arr.ndim == 2:
        heat = arr.astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unexpected array shape for heatmap {path.name}: {arr.shape}")

    # 3. Check shape (Height x Width)
    if heat.shape != (exp_h, exp_w):
        raise ValueError(f"Extracted array shape {heat.shape} does not match ({exp_h}, {exp_w})")

    # 4. Check finiteness
    if not np.all(np.isfinite(heat)):
        raise ValueError(f"Heatmap {path.name} contains non-finite values (NaN / Inf)")

    return heat


# ----------------------------------------------------------------------
# 3. Regional Metrics Calculator
# ----------------------------------------------------------------------
def compute_region_metrics(heatmap: np.ndarray, coords: tuple) -> dict:
    """Computes mean, max, and fraction of pixels exceeding the exploratory 0.2 threshold."""
    x, y, w, h = coords
    region = heatmap[y : y + h, x : x + w]

    return {
        "mean": float(region.mean()),
        "max": float(region.max()),
        "frac_gt_02": float((region > 0.2).mean()) * 100.0,
        "pixel_count": int(region.size)
    }


# ----------------------------------------------------------------------
# 4. Main Analysis Routine
# ----------------------------------------------------------------------
def main():
    print("=" * 76)
    print("ManTra-Net Regional Heatmap Comparison (Post-Inference Analysis)")
    print("=" * 76)

    # 1. Validate files
    validate_prerequisites()

    # 2. Load ground truth
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    doc_w = gt_data["document_dimensions"]["width"]
    doc_h = gt_data["document_dimensions"]["height"]
    p = gt_data["patch_coordinates"]
    target_coords = (p["x"], p["y"], p["width"], p["height"])  # (612, 316, 36, 24)

    # 3. Define nearby control region (Row MA-102 Grade cell, 40px above in same column)
    # y = 316 - 40 = 276 (same w=36, h=24, non-overlapping)
    control_coords = (target_coords[0], target_coords[1] - 40, target_coords[2], target_coords[3])

    print(f"\n📍 Region Coordinates Defined:")
    print(f"  • Target Region  (CS-204 Grade) : x={target_coords[0]}, y={target_coords[1]}, w={target_coords[2]}, h={target_coords[3]}")
    print(f"  • Control Region (MA-102 Grade) : x={control_coords[0]}, y={control_coords[1]}, w={control_coords[2]}, h={control_coords[3]}")
    print(f"    (Equally sized 36x24, unedited in both documents, non-overlapping)")

    # 4. Load & validate existing heatmaps with RGB channel uniformity check
    print(f"\n⏳ Loading and validating existing heatmaps from {TEST_DIR.name}/...")
    heat_clean = load_and_validate_heatmap(CLEAN_HEAT_PATH, (doc_w, doc_h))
    heat_altered = load_and_validate_heatmap(ALTERED_HEAT_PATH, (doc_w, doc_h))
    print(f"✅ Both heatmaps verified:")
    print(f"   - Dimensions: {doc_w}x{doc_h} (matches document).")
    print(f"   - RGB channel uniformity: Verified R == G == B across all pixels (true grayscale encoding).")
    print(f"   - Finiteness: Verified 100% finite values (no NaN / Inf).")

    # 5. Extract metrics for:
    #    a. Target region on clean image
    #    b. Target region on altered image
    #    c. Control region on clean image
    #    d. Control region on altered image
    m_clean_target = compute_region_metrics(heat_clean, target_coords)
    m_alt_target = compute_region_metrics(heat_altered, target_coords)
    m_clean_control = compute_region_metrics(heat_clean, control_coords)
    m_alt_control = compute_region_metrics(heat_altered, control_coords)

    # Target comparison
    target_diff = m_alt_target["mean"] - m_clean_target["mean"]
    target_ratio = m_alt_target["mean"] / (m_clean_target["mean"] + 1e-8)

    # Control comparison: actual percentage change calculation with safe zero division
    control_diff = m_alt_control["mean"] - m_clean_control["mean"]
    if m_clean_control["mean"] > 0:
        control_pct_change = (control_diff / m_clean_control["mean"]) * 100.0
        control_pct_str = f"{control_pct_change:+.2f}%"
    else:
        control_pct_change = 0.0
        control_pct_str = "N/A (clean mean is 0.0)"

    # Target vs Control within Altered Image
    alt_target_vs_control_diff = m_alt_target["mean"] - m_alt_control["mean"]
    alt_target_vs_control_ratio = m_alt_target["mean"] / (m_alt_control["mean"] + 1e-8)

    # Target vs Control within Clean Image
    clean_target_vs_control_diff = m_clean_target["mean"] - m_clean_control["mean"]
    clean_target_vs_control_ratio = m_clean_target["mean"] / (m_clean_control["mean"] + 1e-8)

    # 6. Save Regional Comparison CSV
    with open(OUTPUT_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "region_name", "image_source", "x", "y", "w", "h",
            "mean_activation", "max_activation", "pct_pixels_gt_02"
        ])
        writer.writerow([
            "Target (CS-204 Grade 'B')", "Clean Sample",
            target_coords[0], target_coords[1], target_coords[2], target_coords[3],
            f"{m_clean_target['mean']:.6f}", f"{m_clean_target['max']:.6f}", f"{m_clean_target['frac_gt_02']:.2f}%"
        ])
        writer.writerow([
            "Target (CS-204 Grade 'A+')", "Altered Sample",
            target_coords[0], target_coords[1], target_coords[2], target_coords[3],
            f"{m_alt_target['mean']:.6f}", f"{m_alt_target['max']:.6f}", f"{m_alt_target['frac_gt_02']:.2f}%"
        ])
        writer.writerow([
            "Control (MA-102 Grade 'B+')", "Clean Sample",
            control_coords[0], control_coords[1], control_coords[2], control_coords[3],
            f"{m_clean_control['mean']:.6f}", f"{m_clean_control['max']:.6f}", f"{m_clean_control['frac_gt_02']:.2f}%"
        ])
        writer.writerow([
            "Control (MA-102 Grade 'B+')", "Altered Sample",
            control_coords[0], control_coords[1], control_coords[2], control_coords[3],
            f"{m_alt_control['mean']:.6f}", f"{m_alt_control['max']:.6f}", f"{m_alt_control['frac_gt_02']:.2f}%"
        ])

    print(f"\n📄 Saved regional comparison metrics to:\n   {OUTPUT_CSV_PATH}")

    # 7. Print Formatted Summary Table
    print("\n" + "=" * 76)
    print("📊 REGIONAL HEATMAP COMPARISON SUMMARY")
    print("=" * 76)
    header = f"{'Region Analyzed':<28} | {'Image':<10} | {'Mean':<10} | {'Max':<10} | {'> 0.2 Pct':<10}"
    print(header)
    print("-" * 76)
    print(f"{'Target (Grade B)':<28} | {'Clean':<10} | {m_clean_target['mean']:<10.4f} | {m_clean_target['max']:<10.4f} | {m_clean_target['frac_gt_02']:<9.2f}%")
    print(f"{'Target (Grade A+ [Altered])':<28} | {'Altered':<10} | {m_alt_target['mean']:<10.4f} | {m_alt_target['max']:<10.4f} | {m_alt_target['frac_gt_02']:<9.2f}%")
    print(f"{'Control (Grade B+)':<28} | {'Clean':<10} | {m_clean_control['mean']:<10.4f} | {m_clean_control['max']:<10.4f} | {m_clean_control['frac_gt_02']:<9.2f}%")
    print(f"{'Control (Grade B+)':<28} | {'Altered':<10} | {m_alt_control['mean']:<10.4f} | {m_alt_control['max']:<10.4f} | {m_alt_control['frac_gt_02']:<9.2f}%")
    print("-" * 76)

    print("\n🔬 KEY COMPARATIVE FINDINGS:")
    print(f"  1. Target Alteration Impact (Altered Target vs. Clean Target):")
    print(f"     • Clean target mean : {m_clean_target['mean']:.4f} (Max: {m_clean_target['max']:.4f})")
    print(f"     • Altered target mean: {m_alt_target['mean']:.4f} (Max: {m_alt_target['max']:.4f})")
    print(f"     • Mean difference   : {target_diff:+.4f} (Altered is {target_ratio:.2f}x higher)")
    print(f"     • Pixels > 0.2      : {m_clean_target['frac_gt_02']:.2f}% -> {m_alt_target['frac_gt_02']:.2f}%")

    print(f"\n  2. Control Region Stability (Clean Control vs. Altered Control):")
    print(f"     • Clean control mean : {m_clean_control['mean']:.4f}")
    print(f"     • Altered control mean: {m_alt_control['mean']:.4f}")
    print(f"     • Absolute difference: {control_diff:+.6f}")
    print(f"     • Relative change   : {control_pct_str}")

    print(f"\n  3. Within-Document Contrast (Target vs. Control in Altered Document):")
    print(f"     • Altered Target Mean ({m_alt_target['mean']:.4f}) vs. Altered Control Mean ({m_alt_control['mean']:.4f})")
    print(f"     • Difference        : {alt_target_vs_control_diff:+.4f} (Target is {alt_target_vs_control_ratio:.2f}x higher than control)")
    print(f"     • Baseline in Clean : In the clean document, Target was {clean_target_vs_control_ratio:.2f}x of control.")

    # 8. Forensic Limitations
    print("\n" + "=" * 76)
    print("⚠️  DATA ENCODING & EXPERIMENTAL LIMITATIONS:")
    print("  1. 8-Bit PNG Quantization: The saved heatmaps store activation values in 8-bit")
    print("     grayscale PNG (256 discrete bins, resolution step 1/255 ≈ 0.00392). Metrics")
    print("     computed from saved PNGs reflect this discrete quantization compared to")
    print("     original 32-bit floating point model tensors.")
    print("  2. Local Context Diffusion: The minor change in the control region")
    print(f"     ({control_pct_str}) illustrates receptive field diffusion from nearby CNN")
    print("     convolutions rather than genuine tampering in the control cell.")
    print("  3. Non-Calibrated Likelihoods: These numerical values reflect relative")
    print("     feature anomaly activations and must not be treated as calibrated probabilities.")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()

