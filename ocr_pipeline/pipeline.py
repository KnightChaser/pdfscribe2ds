# ocr_pipeline/pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image

from .config import PipelineConfig
from .pdf_loader import pdf_to_images
from .ocr_engine import DeepSeekOCREngine
from .md_rewriter import rewrite_md_with_embeds

def run_pdf_pipeline(
    pdf_path: Path,
    output_dir: Path,
    model_name: str = "deepseek-ai/DeepSeek-OCR", # NOTE: Fixed
    dpi: int = 200,
    num_processes: int = None,
    num_threads: int = None,
) -> None:
    """
    Full PDF -> images -> DeepSeek-OCR -> Markdown pipeline.

    Args:
        pdf_path (Path): Path to the input PDF file.
        output_dir (Path): Directory to save output images and markdown files.
        model_name (str): Name of the DeepSeek-OCR model to use.
        dpi (int): Dots per inch for image quality when converting PDF to images.
        num_processes (int, optional): Number of processes for parallel PDF conversion.
        num_threads (int, optional): Number of threads for parallel image saving.
    """
    cfg = PipelineConfig(
        pdf_path=pdf_path,
        output_dir=output_dir,
        model_name=model_name,
        dpi=dpi,
    )

    # 1. PDF -> images/{page-001.png, ...}
    images_out_dir = cfg.output_dir / "images"
    image_paths: List[Path] = pdf_to_images(
        pdf_path=cfg.pdf_path,
        out_dir=images_out_dir,
        dpi=cfg.dpi,
        num_processes=num_processes,
        num_threads=num_threads,
    )

    # 2. Prepare OCR engine
    ocr = DeepSeekOCREngine(model_name=cfg.model_name)

     # 3. Per page processing
    md_out_dir = cfg.output_dir / "markdown"
    md_out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        # 2.a OCR
        raw_md = ocr.image_to_markdown(img_path)

        # 2.b rewrite with embeds
        page_stem = img_path.stem  # e.g. "page_001"
        assets_dir = md_out_dir / f"{page_stem}_assets"
        # re-open image to pass to rewriter
        img = Image.open(img_path).convert("RGB")
        cleaned_md = rewrite_md_with_embeds(
            text_output=raw_md,
            image=img,
            output_dir=assets_dir,
            base_img_name=page_stem,
        )

        # 2.c write md
        md_file = md_out_dir / f"{page_stem}.md"
        md_file.write_text(cleaned_md, encoding="utf-8")

        print(f"[OK] page {page_stem} --> {md_file}")


