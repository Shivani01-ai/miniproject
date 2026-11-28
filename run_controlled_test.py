import os
import sys
import json
import time
import csv
import pathlib
import types
from PIL import Image, ImageDraw
import numpy as np

# ----------------------------------------------------------------------
# 1. Paths & Environment Setup (Completely Isolated)
# ----------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(r"C:\Users\Kshethra B J\Document-Integrity-verification")
TEST_DIR = PROJECT_ROOT / "mantra_controlled_test"
MANTRA_ROOT = PROJECT_ROOT / "temp_mantra"
EVAL_DIR = PROJECT_ROOT / "mantra_eval"

TEST_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)

# Direct Matplotlib cache to a writable local directory
os.environ["MPLCONFIGDIR"] = str(EVAL_DIR / ".mplconfig")

sys.path.insert(0, str(MANTRA_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Lightweight stub for unused pytorch_lightning dependency
if "pytorch_lightning" not in sys.modules:
    sys.modules["pytorch_lightning"] = types.ModuleType("pytorch_lightning")

import torch
import matplotlib.pyplot as plt

# Workspace-relative import of MantraNet model loader
from temp_mantra.MantraNet.mantranet import pre_trained_model


# ----------------------------------------------------------------------
# 2. Fictional Document Generator with Controlled Subtle Alteration
# ----------------------------------------------------------------------
def generate_fictional_documents() -> tuple:
    """
    Generates a pair of fictional academic documents:
    1. clean_sample.png: Unaltered pristine base document.
    2. altered_sample.png: Seamlessly modified Grade field ('B' -> 'A+')
       for module CS-204, preserving table dividers and background color.
    Returns: (clean_path, altered_path, target_patch_coords, gt_dict)
    """
    width, height = 800, 600
    bg_color = (252, 252, 250)
    text_color = (30, 30, 30)

    # 1. Create Pristine Clean Document
    clean = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(clean)

    # Borders & Header
    draw.rectangle([30, 30, 770, 570], outline=(40, 40, 40), width=2)
    draw.rectangle([36, 36, 764, 564], outline=(180, 180, 180), width=1)
    draw.text((230, 55), "METROPOLIS INSTITUTE OF TECHNOLOGY", fill=(20, 20, 20))
    draw.text((290, 80), "OFFICIAL STATEMENT OF GRADES", fill=(80, 80, 80))
    draw.line([50, 110, 750, 110], fill=(120, 120, 120), width=2)

    # Invented student metadata (no real personal info)
    draw.text((60, 130), "Candidate Name : Jordan Taylor", fill=text_color)
    draw.text((60, 155), "Program        : B.Sc. Computer Science", fill=text_color)
    draw.text((450, 130), "Student ID  : MIT-2024-EXAM-7741", fill=text_color)
    draw.text((450, 155), "Academic Yr : 2024 - 2025", fill=text_color)

    # Table Header (y = 200 to 230)
    table_top = 200
    draw.rectangle([60, table_top, 740, table_top + 30], fill=(235, 238, 242), outline=(100, 100, 100), width=1)
    draw.text((75,  table_top + 8), "Code", fill=(10, 10, 10))
    draw.text((160, table_top + 8), "Course Title", fill=(10, 10, 10))
    draw.text((420, table_top + 8), "Credits", fill=(10, 10, 10))
    draw.text((510, table_top + 8), "Score", fill=(10, 10, 10))
    draw.text((610, table_top + 8), "Grade", fill=(10, 10, 10))
    draw.text((680, table_top + 8), "Status", fill=(10, 10, 10))

    # Table Rows (each 40px tall: 230-270, 270-310, 310-350, 350-390)
    rows = [
        ("CS-101", "Intro to Programming",     "4.0", "88", "A",  "PASS"),
        ("MA-102", "Discrete Mathematics",     "4.0", "79", "B+", "PASS"),
        ("CS-204", "Data Structures & Algo",   "4.0", "68", "B",  "PASS"),  # Target row: y in [310, 350]
        ("EE-105", "Digital Logic Systems",    "3.0", "92", "A+", "PASS"),
    ]

    row_y = table_top + 30
    target_patch_coords = None

    for code, title, credits, score, grade, status in rows:
        draw.line([60, row_y, 740, row_y], fill=(210, 210, 210), width=1)
        draw.text((75,  row_y + 10), code, fill=text_color)
        draw.text((160, row_y + 10), title, fill=text_color)
        draw.text((435, row_y + 10), credits, fill=text_color)
        draw.text((515, row_y + 10), score, fill=text_color)
        draw.text((620, row_y + 10), grade, fill=text_color)
        draw.text((680, row_y + 10), status, fill=text_color)

        if code == "CS-204":
            # Target Grade cell precisely: x in [612, 648], y in [316, 340]
            # Dimensions: 36 x 24 px. (Safe margins: 6px below y=310 divider, 10px above y=350 divider)
            target_patch_coords = (612, row_y + 6, 36, 24)

        row_y += 40

    draw.line([60, row_y, 740, row_y], fill=(100, 100, 100), width=1)
    draw.text((60, 420), "Result: QUALIFIED FOR ADVANCED DEGREE", fill=(20, 20, 20))
    draw.text((60, 530), "[FICTIONAL SAMPLE DOCUMENT - ACADEMIC INTEGRITY RESEARCH ONLY]", fill=(130, 130, 130))

    # Save Clean Image
    clean_path = TEST_DIR / "clean_sample.png"
    clean.save(clean_path, format="PNG")

    # 2. Create Controlled Altered Copy (No border box, exact background match)
    altered = clean.copy()
    alt_draw = ImageDraw.Draw(altered)
    px, py, pw, ph = target_patch_coords

    # Clear only the original "B" with seamless document background (NO outline)
    alt_draw.rectangle([px, py, px + pw, py + ph], fill=bg_color, outline=None)
    # Render new falsified grade "A+" at identical text baseline
    alt_draw.text((620, py + 4), "A+", fill=text_color)

    altered_path = TEST_DIR / "altered_sample.png"
    altered.save(altered_path, format="PNG")

    # 3. Ground Truth Metadata
    gt_dict = {
        "experiment": "Controlled Subtle Field Integrity Test",
        "document_dimensions": {"width": width, "height": height},
        "target_module": "CS-204 (Data Structures & Algo)",
        "field_name": "Grade",
        "original_value": "B",
        "altered_value": "A+",
        "patch_coordinates": {
            "x": px,
            "y": py,
            "width": pw,
            "height": ph
        },
        "border_line_preservation": "Table dividers at y=310 and y=350 are untouched",
        "background_infill": "Exact match to canvas (252, 252, 250), outline=None",
        "font_rendering_note": (
            "Replacement text 'A+' uses the rendering engine's glyph rendering. "
            "Differences in character stroke geometry, antialiasing, or spacing "
            "between 'B' and 'A+' may influence CNN feature activation."
        )
    }

    gt_path = TEST_DIR / "ground_truth.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_dict, f, indent=2)

    return clean_path, altered_path, target_patch_coords, gt_dict


# ----------------------------------------------------------------------
# 3. Model Loading & Inference Helper
# ----------------------------------------------------------------------
def load_mantra_model(weights_dir: pathlib.Path, device: torch.device):
    """Loads pretrained ManTra-Net with working directory scoping for IMTFEv4.pt."""
    weight_path = weights_dir / "MantraNetv4.pt"
    imtfe_path = weights_dir / "IMTFEv4.pt"

    if not weight_path.exists():
        raise FileNotFoundError(f"Missing primary checkpoint: {weight_path}")
    if not imtfe_path.exists():
        raise FileNotFoundError(f"Missing IMTFE weights file: {imtfe_path}")

    cwd = os.getcwd()
    os.chdir(str(weights_dir))
    try:
        model = pre_trained_model(weight_path=str(weight_path), device=device)
    finally:
        os.chdir(cwd)

    model.eval()
    return model


def run_inference(model: torch.nn.Module, img: Image.Image, device: torch.device) -> np.ndarray:
    """Preprocesses RGB image and returns (H, W) float numpy array of anomaly likelihoods."""
    rgb = img.convert("RGB")
    im_np = np.array(rgb)
    im_tensor = torch.from_numpy(im_np).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

    with torch.no_grad():
        out = model(im_tensor)

    if not bool(torch.isfinite(out).all().item()):
        raise RuntimeError("Model produced non-finite (NaN or Inf) output values.")

    return out.squeeze().cpu().numpy()


# ----------------------------------------------------------------------
# 4. Main Controlled Test Pipeline
# ----------------------------------------------------------------------
def main():
    print("=" * 76)
    print("Controlled Fictional Document Integrity Test (ManTra-Net)")
    print("=" * 76)

    # 1. Synthesize documents
    print("\n⏳ Generating fictional clean and altered document pair...")
    clean_path, altered_path, patch_coords, gt_dict = generate_fictional_documents()
    px, py, pw, ph = patch_coords
    width = gt_dict["document_dimensions"]["width"]
    height = gt_dict["document_dimensions"]["height"]

    print(f"✅ Clean document saved   : {clean_path.name} ({width}x{height})")
    print(f"✅ Altered document saved : {altered_path.name} ({width}x{height})")
    print(f"✅ Ground truth recorded  : x={px}, y={py}, w={pw}, h={ph} (CS-204 Grade)")
    print(f"   Note: {gt_dict['font_rendering_note']}")

    # 2. Load model
    device = torch.device("cpu")
    print(f"\n⏳ Loading ManTra-Net weights on {device}...")
    try:
        model = load_mantra_model(MANTRA_ROOT / "MantraNet", device)
        print("✅ Model loaded successfully into memory.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    clean_img = Image.open(clean_path)
    altered_img = Image.open(altered_path)

    # 3. Run inference on clean document
    print("\n⏳ Running inference on clean document...")
    t0 = time.time()
    heat_clean = run_inference(model, clean_img, device)
    print(f"   Clean inference completed in {time.time() - t0:.2f}s")

    # 4. Run inference on altered document
    print("⏳ Running inference on altered document...")
    t1 = time.time()
    heat_altered = run_inference(model, altered_img, device)
    print(f"   Altered inference completed in {time.time() - t1:.2f}s")

    # 5. Strict Dimension & Integrity Validations
    expected_shape = (height, width)  # (600, 800)
    print(f"\n🔬 Validating heatmap dimensions and values...")
    if heat_clean.shape != expected_shape:
        raise ValueError(f"Clean heatmap shape {heat_clean.shape} does not match expected {expected_shape}")
    if heat_altered.shape != expected_shape:
        raise ValueError(f"Altered heatmap shape {heat_altered.shape} does not match expected {expected_shape}")
    if heat_clean.shape != heat_altered.shape:
        raise ValueError(f"Heatmap shape mismatch: clean {heat_clean.shape} vs altered {heat_altered.shape}")
    if not np.all(np.isfinite(heat_clean)):
        raise ValueError("Clean heatmap contains non-finite values (NaN / Inf)")
    if not np.all(np.isfinite(heat_altered)):
        raise ValueError("Altered heatmap contains non-finite values (NaN / Inf)")

    if px < 0 or py < 0 or px + pw > width or py + ph > height:
        raise ValueError(f"Ground-truth patch ({px}, {py}, {pw}, {ph}) exceeds image bounds ({width}, {height})")

    print(f"✅ Dimensions verified: clean {heat_clean.shape}, altered {heat_altered.shape}. All values finite.")

    # 6. Extract statistics
    patch_slice = (slice(py, py + ph), slice(px, px + pw))
    mask = np.ones_like(heat_clean, dtype=bool)
    mask[patch_slice] = False

    # Clean stats
    c_inside_mean = float(heat_clean[patch_slice].mean())
    c_inside_max = float(heat_clean[patch_slice].max())
    c_outside_mean = float(heat_clean[mask].mean())

    # Altered stats
    a_inside_mean = float(heat_altered[patch_slice].mean())
    a_inside_max = float(heat_altered[patch_slice].max())
    a_outside_mean = float(heat_altered[mask].mean())

    c_contrast = c_inside_mean / (c_outside_mean + 1e-8)
    a_contrast = a_inside_mean / (a_outside_mean + 1e-8)

    # 7. Save Heatmaps & Overlays
    clean_heat_path = TEST_DIR / "clean_heatmap.png"
    altered_heat_path = TEST_DIR / "altered_heatmap.png"
    plt.imsave(str(clean_heat_path), heat_clean, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imsave(str(altered_heat_path), heat_altered, cmap="gray", vmin=0.0, vmax=1.0)

    # Clean overlay
    clean_overlay_path = TEST_DIR / "clean_overlay.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(clean_img)
    im = ax.imshow(heat_clean, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Anomaly Likelihood")
    ax.set_title("Fictional Clean Document - ManTra-Net Heatmap Overlay", fontsize=11)
    ax.axis("off")
    fig.savefig(str(clean_overlay_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Altered overlay
    altered_overlay_path = TEST_DIR / "altered_overlay.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(altered_img)
    im = ax.imshow(heat_altered, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Anomaly Likelihood")
    ax.set_title("Fictional Altered Document - ManTra-Net Heatmap Overlay", fontsize=11)
    ax.axis("off")
    fig.savefig(str(altered_overlay_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n💾 Visual Artifacts Saved to {TEST_DIR.name}/:")
    print(f"  • Clean Heatmap   : {clean_heat_path.name}")
    print(f"  • Clean Overlay   : {clean_overlay_path.name}")
    print(f"  • Altered Heatmap : {altered_heat_path.name}")
    print(f"  • Altered Overlay : {altered_overlay_path.name}")

    # 8. Save CSV Comparison Report
    csv_path = TEST_DIR / "comparison_report.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "metric", "clean_document", "altered_document", "difference", "ratio_altered_to_clean"
        ])
        writer.writerow([
            "target_patch_mean", f"{c_inside_mean:.6f}", f"{a_inside_mean:.6f}",
            f"{a_inside_mean - c_inside_mean:.6f}", f"{a_inside_mean / (c_inside_mean + 1e-8):.4f}"
        ])
        writer.writerow([
            "target_patch_max", f"{c_inside_max:.6f}", f"{a_inside_max:.6f}",
            f"{a_inside_max - c_inside_max:.6f}", f"{a_inside_max / (c_inside_max + 1e-8):.4f}"
        ])
        writer.writerow([
            "background_mean", f"{c_outside_mean:.6f}", f"{a_outside_mean:.6f}",
            f"{a_outside_mean - c_outside_mean:.6f}", f"{a_outside_mean / (c_outside_mean + 1e-8):.4f}"
        ])
        writer.writerow([
            "contrast_ratio_patch_vs_bg", f"{c_contrast:.4f}", f"{a_contrast:.4f}",
            f"{a_contrast - c_contrast:.4f}", f"{a_contrast / (c_contrast + 1e-8):.4f}"
        ])
    print(f"  • Comparison CSV  : {csv_path.name}")

    # 9. Print Comparative Report Table
    print("\n" + "=" * 76)
    print("📊 CONTROLLED EXPERIMENT COMPARISON RESULTS")
    print("=" * 76)
    print(f"  Target Region (CS-204 Grade) : x={px}, y={py}, w={pw}, h={ph}")
    print(f"  Original Value -> Altered    : 'B' -> 'A+' (Seamless background infill)")
    print("-" * 76)
    print(f"{'Evaluation Metric':<35} | {'Clean Document':<16} | {'Altered Document':<16}")
    print("-" * 76)
    print(f"{'Inside Target Patch (Mean)':<35} | {c_inside_mean:<16.4f} | {a_inside_mean:<16.4f}")
    print(f"{'Inside Target Patch (Max)':<35} | {c_inside_max:<16.4f} | {a_inside_max:<16.4f}")
    print(f"{'Outside Target Patch (Mean)':<35} | {c_outside_mean:<16.4f} | {a_outside_mean:<16.4f}")
    print(f"{'Patch vs Background Contrast Ratio':<35} | {c_contrast:<16.2f}x | {a_contrast:<16.2f}x")
    print("-" * 76)
    print(f"  Altered Patch Mean > Clean Patch Mean : {a_inside_mean > c_inside_mean}")
    print(f"  Altered Patch Mean > Background Mean  : {a_inside_mean > a_outside_mean}")

    # 10. Scientific Caveats & Limitations for Academic Project Report
    print("\n" + "=" * 76)
    print("⚠️  SCIENTIFIC & FORENSIC LIMITATIONS FOR ACADEMIC REPORT:")
    print("  1. Uncalibrated Model Outputs: Heatmap activations represent raw network")
    print("     likelihood outputs from the final Sigmoid layer. They must NOT be")
    print("     interpreted as calibrated real-world tampering probabilities.")
    print("  2. Font & Glyph Rendering Influence: The replacement text ('A+') uses")
    print("     rendering font curves and stroke widths different from 'B'. Stroke")
    print("     geometry itself affects high-frequency CNN filter responses.")
    print("  3. Text Edge Sensitivity: High-contrast character strokes naturally")
    print("     activate local anomaly filters on both clean and altered documents,")
    print("     which explains why genuine printed text lines register non-zero heat.")
    print("  4. Absence of Physical Artifacts: This controlled test uses identical")
    print("     lossless PNG encoding. Real tampered documents often have multi-cycle")
    print("     JPEG compression, scanner sensor noise, or optical misalignment.")
    print("  5. Generalization: This single controlled experiment demonstrates model")
    print("     behavior on subtle field edits but cannot establish real-world accuracy.")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()

