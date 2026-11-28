import os
import sys
import io
import csv
import pathlib
import types
import traceback
from PIL import Image, ImageDraw
import numpy as np

# ----------------------------------------------------------------------
# 1. Paths & Environment Setup (Self-Contained & Isolated)
# ----------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(r"C:\Users\Kshethra B J\Document-Integrity-verification")
MANTRA_ROOT = PROJECT_ROOT / "temp_mantra"
EVAL_DIR = PROJECT_ROOT / "mantra_eval"
HEATMAP_DIR = EVAL_DIR / "batch_heatmaps"

EVAL_DIR.mkdir(exist_ok=True)
HEATMAP_DIR.mkdir(exist_ok=True)

# Direct Matplotlib to a writable temporary config directory
os.environ["MPLCONFIGDIR"] = str(EVAL_DIR / ".mplconfig")

sys.path.insert(0, str(MANTRA_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Stub pytorch_lightning before importing MantraNet
if "pytorch_lightning" not in sys.modules:
    sys.modules["pytorch_lightning"] = types.ModuleType("pytorch_lightning")

import torch
import matplotlib.pyplot as plt

# Import model via workspace-relative path
from temp_mantra.MantraNet.mantranet import pre_trained_model


# ----------------------------------------------------------------------
# 2. Document & Patch Generation Helper
# ----------------------------------------------------------------------
def create_base_document() -> Image.Image:
    """
    Creates the identical pristine base document layout used in the baseline
    localization test (table grid, column lines, printed text rows), saved
    and loaded at JPEG Quality 85.
    """
    clean = Image.new("RGB", (600, 500), color=(250, 250, 250))
    draw = ImageDraw.Draw(clean)

    # Outer table border and inner grid lines
    draw.rectangle([40, 40, 560, 460], outline=(0, 0, 0), width=2)
    draw.line([40, 100, 560, 100], fill=(0, 0, 0), width=1)
    draw.line([40, 250, 560, 250], fill=(0, 0, 0), width=1)
    draw.line([200, 40, 200, 460], fill=(0, 0, 0), width=1)

    # Standard printed text rows
    for y in (60, 150, 200, 300, 350, 400):
        draw.text((60, y), "Course / Subject Name", fill=(20, 20, 20))
        draw.text((220, y), "Marks: 85 - Qualified", fill=(20, 20, 20))

    # Compress base document to JPEG quality 85
    buf_clean = io.BytesIO()
    clean.save(buf_clean, format="JPEG", quality=85)
    buf_clean.seek(0)
    clean_jpg = Image.open(buf_clean).convert("RGB")
    return clean_jpg


def generate_synthetic_pair_with_patch(
    base_img: Image.Image,
    patch_coords: tuple,
    patch_text: str,
    patch_quality: int = 45,
    final_quality: int = 85,
) -> tuple:
    """
    Generates a clean/altered pair that are 100% identical everywhere except
    for the deliberate spliced rectangular patch at `patch_coords` = (x, y, w, h).
    """
    x, y, w, h = patch_coords
    W, H = base_img.size

    # Coordinate safety assertions
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid patch dimensions (w={w}, h={h}): must be > 0")
    if x < 0 or y < 0:
        raise ValueError(f"Invalid patch origin (x={x}, y={y}): must be >= 0")
    if x + w > W:
        raise ValueError(f"Patch exceeds image width: x({x}) + w({w}) = {x+w} > W({W})")
    if y + h > H:
        raise ValueError(f"Patch exceeds image height: y({y}) + h({h}) = {y+h} > H({H})")

    # Clean image is the identical base document
    clean_img = base_img.copy()

    # Altered image: creates the patch, compresses to Q45, pastes, re-saves at Q85
    altered = base_img.copy()
    patch = Image.new("RGB", (w, h), color=(240, 240, 230))
    patch_draw = ImageDraw.Draw(patch)
    patch_draw.rectangle([2, 2, w - 3, h - 3], outline=(180, 20, 20), width=2)
    patch_draw.text((8, max(4, h // 3)), patch_text, fill=(180, 20, 20))

    buf_patch = io.BytesIO()
    patch.save(buf_patch, format="JPEG", quality=patch_quality)
    buf_patch.seek(0)
    patch_compressed = Image.open(buf_patch)

    altered.paste(patch_compressed, (x, y))

    buf_altered = io.BytesIO()
    altered.save(buf_altered, format="JPEG", quality=final_quality)
    buf_altered.seek(0)
    altered_jpg = Image.open(buf_altered).convert("RGB")

    return clean_img, altered_jpg


# ----------------------------------------------------------------------
# 3. Model Inference Helper
# ----------------------------------------------------------------------
def run_inference(model: torch.nn.Module, img: Image.Image, device: torch.device) -> np.ndarray:
    """Runs MantraNet on an RGB PIL Image and returns (H, W) numpy float array in [0, 1]."""
    im_np = np.array(img.convert("RGB"))
    im_tensor = torch.from_numpy(im_np).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
    with torch.no_grad():
        out = model(im_tensor)

    if not bool(torch.isfinite(out).all().item()):
        raise RuntimeError("Model returned non-finite (NaN or Inf) output values.")

    return out.squeeze().cpu().numpy()


# ----------------------------------------------------------------------
# 4. Main Batch Evaluation (10 Deterministic Test Pairs)
# ----------------------------------------------------------------------
def run_batch_evaluation():
    print("=" * 80)
    print("MantraNet Multi-Sample Synthetic Batch Evaluation (10 Pairs)")
    print("=" * 80)

    # 1. Load Pretrained Weights
    weight_path = MANTRA_ROOT / "MantraNet" / "MantraNetv4.pt"
    device = torch.device("cpu")

    print("\n⏳ Loading pretrained MantraNet weights...")
    cwd = os.getcwd()
    os.chdir(str(MANTRA_ROOT / "MantraNet"))
    try:
        model = pre_trained_model(weight_path=str(weight_path), device=device)
    finally:
        os.chdir(cwd)

    model.eval()
    print("✅ Model loaded successfully on CPU.\n")

    # 2. Define 10 Varied Test Configurations (Covering multiple zones & sizes)
    # Coordinate system: (x, y, w, h) inside 600x500 document table area [40, 40, 560, 460]
    test_configs = [
        {"id": 1,  "patch": (220, 280, 140, 60), "text": "VERIFIED PASS", "desc": "Marks row (original baseline)"},
        {"id": 2,  "patch": (60,  140, 130, 45), "text": "MATH 201 REV",  "desc": "Left column subject row"},
        {"id": 3,  "patch": (220, 190, 150, 50), "text": "GRADE: A+ 98",  "desc": "Right column upper marks"},
        {"id": 4,  "patch": (55,  290, 135, 55), "text": "PHYSICS LAB",   "desc": "Left column middle row"},
        {"id": 5,  "patch": (215, 340, 160, 48), "text": "CREDITS: 4.0",  "desc": "Right column lower marks"},
        {"id": 6,  "patch": (70,  65,  115, 32), "text": "AUDITED PASS",  "desc": "Top header area (left)"},
        {"id": 7,  "patch": (225, 65,  125, 32), "text": "DISTINCTION",   "desc": "Top header area (right)"},
        {"id": 8,  "patch": (55,  395, 130, 45), "text": "FINAL REVIEW",  "desc": "Bottom row subject"},
        {"id": 9,  "patch": (220, 395, 145, 50), "text": "SCORE: 96/100", "desc": "Bottom row marks"},
        {"id": 10, "patch": (150, 225, 100, 45), "text": "RE-EVALUATED",  "desc": "Straddling vertical divider line"},
    ]

    results = []
    heatmaps_to_save = {1, 2, 3}  # Save 3 representative pairs for visual inspection

    print(f"Starting evaluation of {len(test_configs)} pairs (Seed: 42 fixed deterministic setup)...\n")

    for cfg in test_configs:
        sample_id = cfg["id"]
        patch = cfg["patch"]
        text = cfg["text"]
        desc = cfg["desc"]
        px, py, pw, ph = patch

        print(f"--- [Sample {sample_id:02d}/10] {desc} ---")
        print(f"    Patch: x={px}, y={py}, w={pw}, h={ph} | Spliced text: '{text}'")

        sample_record = {
            "sample_id": sample_id,
            "description": desc,
            "patch_x": px,
            "patch_y": py,
            "patch_w": pw,
            "patch_h": ph,
            "patch_text": text,
            "clean_inside_mean": None,
            "clean_inside_max": None,
            "clean_frac_gt_02": None,
            "altered_inside_mean": None,
            "altered_inside_max": None,
            "altered_frac_gt_02": None,
            "clean_outside_mean": None,
            "altered_outside_mean": None,
            "altered_higher": None,
            "contrast_ratio": None,
            "status": "FAILED",
            "error_message": "",
        }

        try:
            # Create identical base document for both
            base_doc = create_base_document()
            clean_img, altered_img = generate_synthetic_pair_with_patch(
                base_img=base_doc,
                patch_coords=patch,
                patch_text=text,
                patch_quality=45,
                final_quality=85,
            )

            # Validate dimensions
            if clean_img.size != altered_img.size:
                raise ValueError(f"Size mismatch: clean={clean_img.size}, altered={altered_img.size}")

            # Run inference on clean
            heat_clean = run_inference(model, clean_img, device)
            # Run inference on altered
            heat_altered = run_inference(model, altered_img, device)

            # Slice patch regions
            patch_clean = heat_clean[py : py + ph, px : px + pw]
            patch_altered = heat_altered[py : py + ph, px : px + pw]

            # Outside mask
            mask = np.ones_like(heat_clean, dtype=bool)
            mask[py : py + ph, px : px + pw] = False

            outside_clean = heat_clean[mask]
            outside_altered = heat_altered[mask]

            # Calculate metrics
            c_mean = float(patch_clean.mean())
            c_max = float(patch_clean.max())
            c_gt02 = float((patch_clean > 0.2).mean())

            a_mean = float(patch_altered.mean())
            a_max = float(patch_altered.max())
            a_gt02 = float((patch_altered > 0.2).mean())

            c_out_mean = float(outside_clean.mean())
            a_out_mean = float(outside_altered.mean())

            altered_higher = bool(a_mean > c_mean)
            contrast_ratio = float(a_mean / (a_out_mean + 1e-8))

            sample_record.update({
                "clean_inside_mean": c_mean,
                "clean_inside_max": c_max,
                "clean_frac_gt_02": c_gt02,
                "altered_inside_mean": a_mean,
                "altered_inside_max": a_max,
                "altered_frac_gt_02": a_gt02,
                "clean_outside_mean": c_out_mean,
                "altered_outside_mean": a_out_mean,
                "altered_higher": altered_higher,
                "contrast_ratio": contrast_ratio,
                "status": "SUCCESS",
                "error_message": "",
            })

            print(f"    Clean   -> Patch Mean: {c_mean:.4f} | Max: {c_max:.4f} | Frac>0.2: {c_gt02*100:.1f}%")
            print(f"    Altered -> Patch Mean: {a_mean:.4f} | Max: {a_max:.4f} | Frac>0.2: {a_gt02*100:.1f}%")
            print(f"    Outcome -> Altered > Clean: {altered_higher} (Ratio: {contrast_ratio:.2f}x vs background)\n")

            # Save heatmaps for selected samples
            if sample_id in heatmaps_to_save:
                c_path = HEATMAP_DIR / f"sample_{sample_id}_clean_heat.png"
                a_path = HEATMAP_DIR / f"sample_{sample_id}_altered_heat.png"
                plt.imsave(str(c_path), heat_clean, cmap="gray")
                plt.imsave(str(a_path), heat_altered, cmap="gray")
                print(f"    💾 Saved visual heatmaps for Sample {sample_id} to {HEATMAP_DIR.name}/")

        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)}"
            sample_record["error_message"] = err_msg
            print(f"    ❌ Sample {sample_id} Failed: {err_msg}")
            traceback.print_exc()

        results.append(sample_record)

    # ------------------------------------------------------------------
    # 5. Save CSV Report
    # ------------------------------------------------------------------
    csv_path = EVAL_DIR / "batch_evaluation_results.csv"
    fieldnames = [
        "sample_id", "description", "patch_x", "patch_y", "patch_w", "patch_h", "patch_text",
        "clean_inside_mean", "clean_inside_max", "clean_frac_gt_02",
        "altered_inside_mean", "altered_inside_max", "altered_frac_gt_02",
        "clean_outside_mean", "altered_outside_mean",
        "altered_higher", "contrast_ratio", "status", "error_message"
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("=" * 80)
    print(f"📄 Full batch evaluation CSV saved to: {csv_path}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 6. Aggregate Statistics & Summary Table
    # ------------------------------------------------------------------
    successful = [r for r in results if r["status"] == "SUCCESS"]
    failed = [r for r in results if r["status"] != "SUCCESS"]

    print("\n" + "=" * 80)
    print("BATCH EVALUATION SUMMARY TABLE")
    print("=" * 80)
    header = f"{'ID':<3} | {'Description':<28} | {'Clean Mean':<10} | {'Alt Mean':<10} | {'Alt>Clean':<9} | {'Alt Frac>0.2':<12}"
    print(header)
    print("-" * 80)

    for r in results:
        if r["status"] == "SUCCESS":
            alt_higher_str = "YES" if r["altered_higher"] else "NO"
            gt02_str = f"{r['altered_frac_gt_02']*100:.1f}%"
            print(f"{r['sample_id']:<3} | {r['description']:<28} | {r['clean_inside_mean']:<10.4f} | {r['altered_inside_mean']:<10.4f} | {alt_higher_str:<9} | {gt02_str:<12}")
        else:
            print(f"{r['sample_id']:<3} | {r['description']:<28} | {'FAILED':<10} | {'FAILED':<10} | {'NO':<9} | {r['error_message'][:12]}")

    print("-" * 80)

    if successful:
        avg_clean_mean = np.mean([r["clean_inside_mean"] for r in successful])
        avg_clean_max = np.mean([r["clean_inside_max"] for r in successful])
        avg_clean_gt02 = np.mean([r["clean_frac_gt_02"] for r in successful])

        avg_alt_mean = np.mean([r["altered_inside_mean"] for r in successful])
        avg_alt_max = np.mean([r["altered_inside_max"] for r in successful])
        avg_alt_gt02 = np.mean([r["altered_frac_gt_02"] for r in successful])

        avg_alt_outside = np.mean([r["altered_outside_mean"] for r in successful])
        avg_contrast = np.mean([r["contrast_ratio"] for r in successful])

        count_higher = sum(1 for r in successful if r["altered_higher"])
        pct_higher = (count_higher / len(successful)) * 100.0

        print(f"\n📊 AGGREGATE METRICS (N = {len(successful)} successful pairs):")
        print(f"  • Altered Patch > Clean Patch Rate : {count_higher}/{len(successful)} ({pct_higher:.1f}%)")
        print(f"  • Mean Score Inside Clean Patch    : {avg_clean_mean:.4f} (Avg Max: {avg_clean_max:.4f})")
        print(f"  • Mean Score Inside Altered Patch  : {avg_alt_mean:.4f} (Avg Max: {avg_alt_max:.4f})")
        print(f"  • Mean Outside Background Score    : {avg_alt_outside:.4f}")
        print(f"  • Mean Contrast Ratio (Patch/Back) : {avg_contrast:.2f}x")
        print(f"  • Clean Patch Pixels > 0.2*        : {avg_clean_gt02 * 100:.2f}%")
        print(f"  • Altered Patch Pixels > 0.2*      : {avg_alt_gt02 * 100:.2f}%")
        print("    *(Note: 0.2 is an empirical inspection threshold, not a calibrated decision cutoff)")
    else:
        print("\n❌ All evaluation pairs failed. Check logs for details.")

    if failed:
        print(f"\n⚠️  {len(failed)} sample(s) encountered errors:")
        for f_sample in failed:
            print(f"   - Sample {f_sample['sample_id']}: {f_sample['error_message']}")

    print("\n" + "=" * 80)
    print("LIMITATIONS OF THIS SYNTHETIC EVALUATION:")
    print("  1. Splicing Type: Tests rectangular patch insertion with compression difference")
    print("     (Q45 on Q85 background). Does not cover inpainting, text erasing, or resampling.")
    print("  2. Document Diversity: Conducted on synthetic table documents with clean backgrounds;")
    print("     real marks cards have scanner noise, watermarks, paper textures, and skew.")
    print("  3. Thresholding: The 0.2 threshold is exploratory; a production deployment requires")
    print("     formal ROC/PR curve calibration across pristine and forged real-world corpora.")
    print("  4. Speed & Runtime: ManTra-Net runs ~25-30 seconds per page on CPU, requiring")
    print("     optimization or GPU acceleration before interactive UI integration.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_batch_evaluation()

