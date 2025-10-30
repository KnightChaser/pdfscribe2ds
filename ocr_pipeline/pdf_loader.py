# ocr_pipeline/pdf_loader.py
from __future__ import annotations

from pathlib import Path
from typing import List
from pdf2image import convert_from_path

def pdf_to_images(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 200,
    format: str = "png"
) -> List[Path]:
    """
    Convert a PDF into a list of image files (one per page)

    Args:
        pdf_path (Path): Path to the input PDF file.
        out_dir (Path): Directory to save the output images.
        dpi (int, optional): Dots per inch for image quality. Defaults to 200.
        format (str, optional): Image format (e.g., 'png', 'jpeg'). Defaults to 'png'.

    Returns:
        List[Path]: List of paths to the generated image files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = convert_from_path(str(pdf_path), dpi=dpi)
    image_paths: List[Path] = []

    for idx, page in enumerate(pages, start=1):
        img_name = f"page-{idx:03d}.{format}"
        img_path = out_dir / img_name
        page.save(img_path, format=format.upper())
        image_paths.append(img_path)

    return image_paths
