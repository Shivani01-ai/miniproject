import pathlib
import sys
import os
import types

# ----------------------------------------------------------------------
# 1. Environment & Path Setup
# ----------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(r"C:\Users\Kshethra B J\Document-Integrity-verification")
MANTRA_ROOT = PROJECT_ROOT / "temp_mantra"
EVAL_DIR = PROJECT_ROOT / "mantra_eval"
EVAL_DIR.mkdir(exist_ok=True)

# Silence matplotlib cache warning by setting a writable config directory
os.environ["MPLCONFIGDIR"] = str(EVAL_DIR / ".mplconfig")

sys.path.insert(0, str(MANTRA_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------------------------
# 2. Stub pytorch_lightning before importing MantraNet
# ----------------------------------------------------------------------
if "pytorch_lightning" not in sys.modules:
    sys.modules["pytorch_lightning"] = types.ModuleType("pytorch_lightning")

# ----------------------------------------------------------------------
# 3. Import MantraNet and Synthetic Document Generator
# ----------------------------------------------------------------------
from temp_mantra.MantraNet.mantranet import pre_trained_model
from tests.test_localization_synthetic import generate_synthetic_documents

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 4. Generate Synthetic Test Documents
# ----------------------------------------------------------------------
clean_img, altered_img, gt_patch = generate_synthetic_documents()
print("\n" + "=" * 60)
print("🔎 Synthetic Document Info:")
print(f"  Clean image size   : {clean_img.size} (width, height)")
print(f"  Altered image size : {altered_img.size} (width, height)")
print(f"  Ground-truth patch : x={gt_patch[0]}, y={gt_patch[1]}, w={gt_patch[2]}, h={gt_patch[3]}")
print("=" * 60)

# ----------------------------------------------------------------------
# 5. Load Pretrained MantraNet Model
# mantranet.py executes `torch.load('IMTFEv4.pt')` relative to the current
# working directory, so we switch into temp_mantra/MantraNet while loading.
# ----------------------------------------------------------------------
weight_path = MANTRA_ROOT / "MantraNet" / "MantraNetv4.pt"
device = torch.device("cpu")

cwd = os.getcwd()
os.chdir(str(MANTRA_ROOT / "MantraNet"))
try:
    print("\n⏳ Loading MantraNet model checkpoint...")
    model = pre_trained_model(weight_path=str(weight_path), device=device)
    print("✅ Model loaded successfully on CPU.")
finally:
    os.chdir(cwd)

model.eval()

# ----------------------------------------------------------------------
# 6. Inference Helper with Full Diagnostics
# ----------------------------------------------------------------------
def run_mantra(img: Image.Image, name: str) -> np.ndarray:
    rgb = img.convert("RGB")
    im_np = np.array(rgb)  # (H, W, 3) in [0, 255]
    
    # Shape transformation: (1, 3, H, W)
    im_tensor = torch.from_numpy(im_np).unsqueeze(0).permute(0, 3, 1, 2).float()
    im_tensor = im_tensor.to(device)
    
    with torch.no_grad():
        out = model(im_tensor)  # Expected: (1, 1, H, W)
    
    is_finite = bool(torch.isfinite(out).all().item())
    min_val = float(out.min().item())
    max_val = float(out.max().item())
    
    print(f"\n🔬 Output Diagnostics [{name}]:")
    print(f"  Input Tensor Shape : {list(im_tensor.shape)}")
    print(f"  Output Tensor Shape: {list(out.shape)}")
    print(f"  Output Dtype       : {out.dtype}")
    print(f"  Min Value          : {min_val:.6f}")
    print(f"  Max Value          : {max_val:.6f}")
    print(f"  All Values Finite  : {is_finite}")
    
    if not is_finite:
        raise ValueError(f"Model produced invalid (NaN or Inf) output for {name}")
    
    return out.squeeze().cpu().numpy()

# ----------------------------------------------------------------------
# 7. Run Inference on Clean and Altered Documents
# ----------------------------------------------------------------------
print("\n⏳ Running inference on Clean image...")
heat_clean = run_mantra(clean_img, "Clean")

print("\n⏳ Running inference on Altered image...")
heat_altered = run_mantra(altered_img, "Altered")

# ----------------------------------------------------------------------
# 8. Save Output Heatmaps
# ----------------------------------------------------------------------
clean_heat_path = EVAL_DIR / "clean_heat.png"
altered_heat_path = EVAL_DIR / "altered_heat.png"
plt.imsave(str(clean_heat_path), heat_clean, cmap="gray")
plt.imsave(str(altered_heat_path), heat_altered, cmap="gray")

print("\n" + "=" * 60)
print("💾 Heatmaps Saved:")
print(f"  Clean Heatmap   → {clean_heat_path}")
print(f"  Altered Heatmap → {altered_heat_path}")

# ----------------------------------------------------------------------
# 9. Compute Patch Statistics (Identical Coordinates for Both)
# ----------------------------------------------------------------------
px, py, pw, ph = gt_patch
patch_slice = (slice(py, py + ph), slice(px, px + pw))

def log_stats(heat_map: np.ndarray, label: str):
    patch = heat_map[patch_slice]
    mask = np.ones_like(heat_map, dtype=bool)
    mask[patch_slice] = False
    outside = heat_map[mask]
    
    mean_inside = float(patch.mean())
    max_inside = float(patch.max())
    mean_outside = float(outside.mean())
    max_outside = float(outside.max())
    
    print(f"\n📊 {label} Region Statistics:")
    print(f"  • Inside Patch  [Mean]: {mean_inside:.4f} | [Max]: {max_inside:.4f}")
    print(f"  • Outside Patch [Mean]: {mean_outside:.4f} | [Max]: {max_outside:.4f}")
    print(f"  • Patch vs Background Ratio: {mean_inside / (mean_outside + 1e-8):.2f}x")

print("=" * 60)
log_stats(heat_clean, "Clean Document")
log_stats(heat_altered, "Altered Document")

# ----------------------------------------------------------------------
# 10. Repository License Status
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("⚖️  License & Distribution Status:")
print("  • The cloned repository (RonyAbecidan/ManTraNet-pytorch) does not contain")
print("    a LICENSE file in its root directory.")
print("  • The upstream model weights (ManTra-Net, Wu et al., CVPR 2019) are released")
print("    under a non-commercial academic license by USC ISI.")
print("  • Commercial redistribution or production bundling is not granted without explicit license.")
print("=" * 60 + "\n")
