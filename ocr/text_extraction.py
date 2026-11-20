"""
OCR Text Extraction Module
Uses EasyOCR to extract readable text lines from document images in CPU mode.
"""

from typing import List, Optional
import numpy as np
from PIL import Image


def get_easyocr_reader(languages: Optional[List[str]] = None, gpu: bool = False):
    """
    Initializes and returns an EasyOCR Reader instance.

    Args:
        languages: List of language codes to recognize (defaults to ['en']).
        gpu: Whether to use CUDA GPU acceleration (default False for CPU compatibility).

    Returns:
        easyocr.Reader: Configured EasyOCR reader instance.
    """
    import easyocr

    if languages is None:
        languages = ["en"]
    return easyocr.Reader(languages, gpu=gpu)


def extract_text(image: Image.Image, reader=None) -> str:
    """
    Extracts text from a PIL Image using an EasyOCR reader.

    Args:
        image: PIL Image to process.
        reader: Optional pre-initialized EasyOCR reader instance.
                If None, creates a default English CPU reader.

    Returns:
        str: Extracted text lines joined by newlines.
    """
    if reader is None:
        reader = get_easyocr_reader(languages=["en"], gpu=False)

    # EasyOCR expects a numpy array in RGB format
    image_np = np.array(image.convert("RGB"))

    # detail=0 returns a list of recognized string lines
    results = reader.readtext(image_np, detail=0)

    # Filter out empty or whitespace-only lines
    clean_lines = [line.strip() for line in results if line and line.strip()]
    return "\n".join(clean_lines)

