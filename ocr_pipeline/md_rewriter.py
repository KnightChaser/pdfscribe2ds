# ocr_pipeline/md_rewriter.py
from __future__ import annotations

import os
import re
import ast
from pathlib import Path
from typing import List
from PIL import Image

# Matches only image / image_caption ref+det blocks
_IMG_TAG = re.compile(
    r"<\|ref\|\>(image|image_caption)<\|/ref\|\><\|det\|\>(\[\[.*?\]\])<\|/det\|\>",
    re.DOTALL,
)

# Strip everything else: text, sub_title, title, etc.
_NON_IMG_TAG = re.compile(
    r"<\|ref\|\>(?!image|image_caption)[^<]+<\|/ref\|\><\|det\|\>\[\[.*?\]\]\<\|/det\|\>",
    re.DOTALL,
)

def _scale_box(box, W: int, H: int, pad: int = 2) -> tuple[int, int, int, int]:
    """
    Scale bounding box coordinates from 0-999 range to image dimensions.
    (DeepSeek coordinates are normalized [0...999]; scale back to pixels)

    Args:
        box (list|tuple): Bounding box in [x1, y1, x2, y2] format.
        W (int): Width of the image.
        H (int): Height of the image.
        pad (int, optional): Padding to add around the box. Defaults to 2.

    Returns:
        tuple[int, int, int, int]: Scaled bounding box (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = map(float, box)
    x1 = int(round(x1 / 999.0 * W))
    y1 = int(round(y1 / 999.0 * H))
    x2 = int(round(x2 / 999.0 * W))
    y2 = int(round(y2 / 999.0 * H))

    # Normalize coordinates
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # Apply padding
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)
    return x1, y1, x2, y2

def rewrite_md_with_embeds(
    text_output: str,
    image: Image.Image,
    output_dir: Path,
    base_img_name: str
) -> str:
    """
    Rewrite markdown text by extracting embedded images from the provided image
    based on bounding boxes specified in the text_output. Saves the extracted images
    to the output directory and updates the markdown to reference these images.

    Args:
        text_output (str): The original markdown text with embedded image tags.
        image (Image.Image): The source image from which to extract embedded images.
        output_dir (Path): Directory to save the extracted images.
        base_img_name (str): Base name for the extracted image files.

    Returns:
        str: The rewritten markdown text with updated image references.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    W, H = image.size
    pieces: List[str] = []
    last = 0
    img_counter = 1

    for m in _IMG_TAG.finditer(text_output):
        pieces.append(text_output[last:m.start()]) # Add preceding text

        label = m.group(1).strip()
        boxes = ast.literal_eval(m.group(2))
        if boxes and isinstance(boxes[0], (int, float)):
            boxes = [boxes]  # Single box case

        if label == "image":
            md_snips: List[str] = []
            for box in boxes:
                x1, y1, x2, y2 = _scale_box(box, W, H)
                if x2 <= x1 or y2 <= y1:
                    continue  # Skip invalid boxes

                crop = image.crop((x1, y1, x2, y2))
                crop_name = f"{base_img_name}_img{img_counter:03d}.png"
                crop.save(output_dir / crop_name)
                md_snips.append(f"![Image {img_counter}]({crop_name})")
                img_counter += 1
            pieces.append("\n".join(md_snips) + "\n")

        # captions: ignored, text is already in plain text, do nothing
        last = m.end()

    pieces.append(text_output[last:])  # Add remaining text
    new_md = "".join(pieces)

    # NOTE: Drop any remaining non-image tags (e.g. text, sub_titles)
    new_md = _NON_IMG_TAG.sub("", new_md)
    return new_md
