import os
import sys
import time
import pathlib
import types
from PIL import Image
import numpy as np

# ----------------------------------------------------------------------
# 1. Paths & Environment Setup (Isolated & Reusable)
# ----------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(r"C:\Users\Kshethra B J\Document-Integrity-verification")
TEST_IMG_DIR = PROJECT_ROOT / "real_document_tests"
MANTRA_ROOT = PROJECT_ROOT / "temp_mantra"
EVAL_DIR = PROJECT_ROOT / "mantra_eval"
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
# 2. Image Discovery & Validation
# ----------------------------------------------------------------------
def find_target_image(img_dir: pathlib.Path) -> pathlib.Path:
    """Finds the first supported JPG, JPEG, or PNG image in `img_dir`."""
    if not img_dir.exists() or not img_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {img_dir}")

    supported_exts = {".jpg", ".jpeg", ".png"}
    candidates = sorted([
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in supported_exts
    ])

    if not candidates:
        raise FileNotFoundError(
            f"No supported image file (.jpg, .jpeg, .png) found inside: {img_dir}"
        )

    return candidates[0]


# ----------------------------------------------------------------------
# 3. Model Loading Helper
# ----------------------------------------------------------------------
def load_mantranet_model(weights_dir: pathlib.Path, device: torch.device):
    """Loads the pretrained MantraNet model on `device` with working-directory scoping."""
    weight_path = weights_dir / "MantraNetv4.pt"
    imtfe_path = weights_dir / "IMTFEv4.pt"

    if not weight_path.exists():
        raise FileNotFoundError(f"Missing primary checkpoint: {weight_path}")
    if not imtfe_path.exists():
        raise FileNotFoundError(f"Missing IMTFE weights file: {imtfe_path}")

    # mantranet.py executes relative `torch.load('IMTFEv4.pt')` during initialization
    cwd = os.getcwd()
    os.chdir(str(weights_dir))
    try:
        model = pre_trained_model(weight_path=str(weight_path), device=device)
    finally:
        os.chdir(cwd)

    model.eval()
    return model


# ----------------------------------------------------------------------
# 4. Main Inference Execution
# ----------------------------------------------------------------------
def main():
    print("=" * 76)
    print("ManTra-Net Isolated Real Document Inference")
    print("=" * 76)

    # 1. Locate target image
    try:
        target_img_path = find_target_image(TEST_IMG_DIR)
        print(f"\n📂 Target image located: {target_img_path.name}")
        print(f"   Full Path: {target_img_path}")
    except Exception as e:
        print(f"\n❌ Error finding target image: {e}")
        sys.exit(1)

    # 2. Load and validate image
    try:
        raw_img = Image.open(target_img_path)
        rgb_img = raw_img.convert("RGB")
        img_w, img_h = rgb_img.size
        print(f"   Dimensions (Width x Height): {img_w} x {img_h} pixels")
        print(f"   Original Format: {raw_img.format} | Mode: {rgb_img.mode}")
    except Exception as e:
        print(f"\n❌ Error reading image file: {e}")
        sys.exit(1)

    # 3. Load model checkpoint
    device = torch.device("cpu")
    print(f"\n⏳ Loading ManTra-Net weights on {device.type.upper()}...")
    try:
        model = load_mantranet_model(MANTRA_ROOT / "MantraNet", device)
        print("✅ Model loaded successfully into memory.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # 4. Prepare tensor input: (1, 3, H, W) in range [0, 255]
    im_np = np.array(rgb_img)
    im_tensor = torch.from_numpy(im_np).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

    print(f"\n🔬 Tensor Input Prepared:")
    print(f"   Shape: {list(im_tensor.shape)} | Dtype: {im_tensor.dtype}")
    print(f"   Input Value Range: [{im_tensor.min().item():.1f}, {im_tensor.max().item():.1f}]")

    # 5. Execute forward pass
    print(f"\n⏳ Running ManTra-Net forward pass (this may take 1-3 minutes on CPU for {img_w}x{img_h})...")
    start_time = time.time()
    try:
        with torch.no_grad():
            out_tensor = model(im_tensor)
        elapsed = time.time() - start_time
        print(f"✅ Inference finished in {elapsed:.2f} seconds.")
    except Exception as e:
        print(f"❌ Inference failed during forward pass: {e}")
        sys.exit(1)

    # 6. Validate output
    if not bool(torch.isfinite(out_tensor).all().item()):
        print("❌ Error: Output tensor contains non-finite values (NaN / Inf).")
        sys.exit(1)

    heatmap = out_tensor.squeeze().cpu().numpy()
    out_h, out_w = heatmap.shape

    # 7. Print statistics
    min_val = float(heatmap.min())
    max_val = float(heatmap.max())
    mean_val = float(heatmap.mean())
    std_val = float(heatmap.std())
    pct_gt_02 = float((heatmap > 0.2).mean()) * 100.0

    print("\n" + "=" * 76)
    print("📊 Inference Output Statistics:")
    print("=" * 76)
    print(f"  • Input Image Dimensions     : {img_w} x {img_h} (Width x Height)")
    print(f"  • Heatmap Dimensions         : {out_w} x {out_h} (Matches Input)")
    print(f"  • Minimum Probability Value  : {min_val:.6f}")
    print(f"  • Maximum Probability Value  : {max_val:.6f}")
    print(f"  • Mean Probability Value     : {mean_val:.6f}")
    print(f"  • Standard Deviation         : {std_val:.6f}")
    print(f"  • Fraction of Pixels > 0.2*  : {pct_gt_02:.2f}%")
    print("    *(0.2 is an exploratory heuristic threshold, not a calibrated decision threshold)")

    # 8. Save raw grayscale anomaly heatmap
    heat_path = EVAL_DIR / "real_document_heat.png"
    plt.imsave(str(heat_path), heatmap, cmap="gray", vmin=0.0, vmax=1.0)
    print(f"\n💾 Saved raw anomaly heatmap to:\n   {heat_path}")

    # 9. Save colorized overlay on the original document
    overlay_path = EVAL_DIR / "real_document_overlay.png"
    fig, ax = plt.subplots(figsize=(10, max(6, int(10 * img_h / img_w))))
    ax.imshow(rgb_img)
    im = ax.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ManTra-Net Anomaly Likelihood")
    ax.set_title(f"ManTra-Net Tampering Heatmap Overlay ({target_img_path.name})", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(overlay_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Saved colorized overlay to:\n   {overlay_path}")

    # 10. Exploratory test disclaimer
    print("\n" + "=" * 76)
    print("ℹ️  Interpretation & Forensic Notice:")
    print("  • This is an exploratory visual test on an uncalibrated real-world document.")
    print("  • High activations can stem from genuine strong gradients (dense bold text,")
    print("    stamps, logos, scanner sensor noise) as well as genuine tampering traces.")
    print("  • Do NOT assume the document is authentic or tampered solely from this heatmap.")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()

