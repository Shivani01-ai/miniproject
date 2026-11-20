"""
Repeatable Synthetic Regression Test for Document Localization
Compares Baseline vs. Contextual Multi-Evidence Localization on:
1. A clean synthetic document containing table borders and printed text lines.
2. An altered synthetic document containing an inserted rectangular patch.

Evaluates candidate count reduction, altered patch detection, and spatial overlap coverage.
"""

import io
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from forensics.localization import locate_suspicious_regions


def generate_synthetic_documents():
    """
    Creates a pair of synthetic test documents:
    - Clean: Table borders, column dividers, and multiple text rows saved as JPEG quality 85.
    - Altered: Identical base document with a spliced rectangular patch at (220, 280).
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

    # Save clean base as JPEG quality 85
    buf_clean = io.BytesIO()
    clean.save(buf_clean, format="JPEG", quality=85)
    buf_clean.seek(0)
    clean_jpg = Image.open(buf_clean)

    # Spliced rectangular patch at (220, 280, 140, 60)
    altered = clean_jpg.copy()
    patch = Image.new("RGB", (140, 60), color=(240, 240, 230))
    patch_draw = ImageDraw.Draw(patch)
    patch_draw.rectangle([5, 5, 135, 55], outline=(180, 20, 20), width=2)
    patch_draw.text((15, 20), "VERIFIED PASS", fill=(180, 20, 20))

    # Spliced patch saved with different compression quality (quality 45)
    buf_patch = io.BytesIO()
    patch.save(buf_patch, format="JPEG", quality=45)
    buf_patch.seek(0)
    patch_compressed = Image.open(buf_patch)

    altered.paste(patch_compressed, (220, 280))

    buf_altered = io.BytesIO()
    altered.save(buf_altered, format="JPEG", quality=85)
    buf_altered.seek(0)
    altered_jpg = Image.open(buf_altered)

    ground_truth_patch = (220, 280, 140, 60)
    return clean_jpg, altered_jpg, ground_truth_patch


def compute_patch_coverage(candidates, ground_truth):
    """Calculates maximum intersection coverage percentage with the ground truth patch."""
    if not candidates:
        return 0.0, None

    px, py, pw, ph = ground_truth
    patch_area = pw * ph

    best_coverage = 0.0
    best_candidate = None

    for c in candidates:
        cx, cy, cw, ch = c["X"], c["Y"], c["Width"], c["Height"]
        overlap_x = max(0, min(px + pw, cx + cw) - max(px, cx))
        overlap_y = max(0, min(py + ph, cy + ch) - max(py, cy))
        overlap_area = overlap_x * overlap_y
        coverage = (overlap_area / patch_area) * 100.0

        if coverage > best_coverage:
            best_coverage = coverage
            best_candidate = c

    return best_coverage, best_candidate


def run_comparative_regression():
    print("=" * 76)
    print("Document Localization Regression Test: Baseline vs. Contextual Enhanced")
    print("=" * 76)

    clean_doc, altered_doc, gt_patch = generate_synthetic_documents()

    # 1. Evaluate Baseline Method
    _, c_base = locate_suspicious_regions(clean_doc, method="baseline")
    _, a_base = locate_suspicious_regions(altered_doc, method="baseline")
    cov_base, top_match_base = compute_patch_coverage(a_base, gt_patch)

    # 2. Evaluate Contextual Multi-Evidence Method
    _, c_ctx = locate_suspicious_regions(clean_doc, method="contextual")
    _, a_ctx = locate_suspicious_regions(altered_doc, method="contextual")
    cov_ctx, top_match_ctx = compute_patch_coverage(a_ctx, gt_patch)

    # Summary table
    print("\nMetric Comparison Summary:")
    print("-" * 76)
    print(f"{'Evaluation Metric':<40} | {'Baseline':<15} | {'Contextual':<15}")
    print("-" * 76)
    print(f"{'Clean Document Candidate Count':<40} | {len(c_base):<15} | {len(c_ctx):<15}")
    print(f"{'Altered Document Candidate Count':<40} | {len(a_base):<15} | {len(a_ctx):<15}")
    print(f"{'Altered Patch Detected?':<40} | {'YES' if cov_base > 50 else 'NO':<15} | {'YES' if cov_ctx > 50 else 'NO':<15}")
    print(f"{'Ground Truth Patch Overlap':<40} | {cov_base:.1f}%{'':<10} | {cov_ctx:.1f}%{'':<10}")
    
    top_base_impact = a_base[0]['Anomaly Impact'] if a_base else 0.0
    top_ctx_impact = a_ctx[0]['Anomaly Impact'] if a_ctx else 0.0
    print(f"{'Top Candidate Anomaly Impact':<40} | {top_base_impact:<15.1f} | {top_ctx_impact:<15.1f}")

    if c_base:
        contrast_base = top_base_impact / max(1.0, c_base[0]['Anomaly Impact'])
    else:
        contrast_base = 0.0

    if c_ctx:
        contrast_ctx = top_ctx_impact / max(1.0, c_ctx[0]['Anomaly Impact'])
    else:
        contrast_ctx = top_ctx_impact

    print(f"{'Contrast vs. Clean Top Region':<40} | {contrast_base:<15.2f}x | {contrast_ctx:<15.2f}x")
    print("-" * 76)

    print("\nDetailed Findings:")
    print(f"- Ground Truth Spliced Patch: (X={gt_patch[0]}, Y={gt_patch[1]}, W={gt_patch[2]}, H={gt_patch[3]})")
    if top_match_ctx:
        print(f"- Contextual Best Match: Region '{top_match_ctx['Region']}' at (X={top_match_ctx['X']}, Y={top_match_ctx['Y']}, W={top_match_ctx['Width']}, H={top_match_ctx['Height']}) with {cov_ctx:.1f}% overlap.")
    print(f"- False Positive Reduction on Clean Document: {len(c_base)} -> {len(c_ctx)} candidates ({(1 - len(c_ctx)/max(1, len(c_base))) * 100:.1f}% reduction).")
    print("=" * 76)


if __name__ == "__main__":
    run_comparative_regression()
