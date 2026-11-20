import io
import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from forensics.localization import locate_suspicious_regions, compute_ela_difference_array

clean = Image.new("RGB", (600, 500), color=(250, 250, 250))
draw = ImageDraw.Draw(clean)
draw.rectangle([40, 40, 560, 460], outline=(0, 0, 0), width=2)
draw.line([40, 100, 560, 100], fill=(0, 0, 0), width=1)
draw.line([40, 250, 560, 250], fill=(0, 0, 0), width=1)
draw.line([200, 40, 200, 460], fill=(0, 0, 0), width=1)
for y in (60, 150, 200, 300, 350, 400):
    draw.text((60, y), "Course / Subject Name", fill=(20, 20, 20))
    draw.text((220, y), "Marks: 85 - Qualified", fill=(20, 20, 20))

buf = io.BytesIO()
clean.save(buf, format="JPEG", quality=85)
buf.seek(0)
clean_jpg = Image.open(buf)

altered_jpg = clean_jpg.copy()
patch = Image.new("RGB", (140, 60), color=(240, 240, 230))
p_draw = ImageDraw.Draw(patch)
p_draw.rectangle([5, 5, 135, 55], outline=(180, 20, 20), width=2)
p_draw.text((15, 20), "VERIFIED PASS", fill=(180, 20, 20))
p_buf = io.BytesIO()
patch.save(p_buf, format="JPEG", quality=45)
p_buf.seek(0)
patch_compressed = Image.open(p_buf)
altered_jpg.paste(patch_compressed, (220, 280))
buf2 = io.BytesIO()
altered_jpg.save(buf2, format="JPEG", quality=85)
buf2.seek(0)
altered_jpg = Image.open(buf2)

def filter_candidates(boxes):
    if not boxes:
        return []
    all_means = np.array([b['Mean Error'] for b in boxes], dtype=np.float32)
    fg_med = float(np.median(all_means))
    fg_mad = float(np.median(np.abs(all_means - fg_med)))
    if fg_mad < 0.1:
        fg_mad = 0.4
        
    filtered = []
    for b in boxes:
        x, y, w, h = b['X'], b['Y'], b['Width'], b['Height']
        area = w * h
        aspect = max(w, h) / max(1, min(w, h))
        z_fg = (b['Mean Error'] - fg_med) / fg_mad
        
        # Check if it is a standard 1D text line strip
        is_text_strip = (aspect >= 4.5 and min(w, h) <= 22)
        
        # If it is a standard text strip, check if its deviation from typical text is small
        if is_text_strip and abs(z_fg) < 1.6 and area < 4000:
            continue
            
        b_copy = dict(b)
        b_copy['Z_fg'] = round(z_fg, 2)
        b_copy['Aspect'] = round(aspect, 1)
        filtered.append(b_copy)
    return filtered

_, c_boxes = locate_suspicious_regions(clean_jpg)
_, a_boxes = locate_suspicious_regions(altered_jpg)

print(f"BASELINE: Clean boxes = {len(c_boxes)} | Altered boxes = {len(a_boxes)}")
c_filt = filter_candidates(c_boxes)
a_filt = filter_candidates(a_boxes)
print(f"ENHANCED FILTERING: Clean boxes = {len(c_filt)} | Altered boxes = {len(a_filt)}")
for af in a_filt:
    print("  -> Altered Candidate:", af['Region'], af['X'], af['Y'], af['Width'], af['Height'], "Impact:", af['Anomaly Impact'], "Aspect:", af['Aspect'], "Z_fg:", af['Z_fg'])
