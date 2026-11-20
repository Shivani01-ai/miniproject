"""
Error Level Analysis (ELA) Module
Detects compression discrepancies by re-saving the image at a known JPEG quality
and computing the absolute pixel difference between original and recompressed versions.
"""

import io
from PIL import Image, ImageChops, ImageEnhance


def perform_ela(image: Image.Image, quality: int = 90, scale: int = 15) -> Image.Image:
    """
    Performs Error Level Analysis (ELA) on a PIL Image.

    Args:
        image: Original PIL Image.
        quality: JPEG compression quality for recompression (1-100, default 90).
        scale: Multiplier to amplify differences for human visual inspection (default 15).

    Returns:
        PIL.Image.Image: Visualized ELA difference image.
    """
    # 1. Convert to RGB to ensure uniform 3-channel processing
    orig_rgb = image.convert("RGB")

    # 2. Save image temporarily in an in-memory buffer at specified JPEG quality
    buffer = io.BytesIO()
    orig_rgb.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    # 3. Reopen the recompressed JPEG image
    recompressed = Image.open(buffer)

    # 4. Calculate absolute pixel-wise difference between original and recompressed
    diff = ImageChops.difference(orig_rgb, recompressed)

    # 5. Determine safe, non-zero scaling factor
    extrema = diff.getextrema()
    # extrema is ((min_r, max_r), (min_g, max_g), (min_b, max_b))
    max_diff = max([channel_extrema[1] for channel_extrema in extrema]) if extrema else 0

    if scale is not None and scale > 0:
        safe_scale = float(scale)
    elif max_diff > 0:
        safe_scale = 255.0 / float(max_diff)
    else:
        safe_scale = 1.0

    # 6. Amplify the subtle compression differences so they are visible to the human eye
    enhancer = ImageEnhance.Brightness(diff)
    ela_image = enhancer.enhance(safe_scale)

    return ela_image

