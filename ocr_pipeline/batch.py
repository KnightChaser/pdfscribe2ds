# ocr_pipeline/batch.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import logging

from .pipeline import run_pdf_pipeline
from .ocr_engine import DeepSeekOCREngine
import quiet

logger = logging.getLogger("pdfscribe2ds")

def _iter_pdfs(pdf_dir: Path) -> Iterable[Path]:
    """
    Yield all PDF files in the given directory.

    Args:
        pdf_dir (Path): Directory to search for PDF files.

    Yields:
        Iterable[Path]: Paths to PDF files.
    """
    for pdf_path in pdf_dir.glob("*.pdf"):
        yield pdf_path

def run_batch(
    pdf_dir: Path,
    output_dir: Path,
    model_name: str = "deepseek-ai/DeepSeek-OCR",
    dpi: int = 200,
    num_processes: Optional[int] = None,
    num_threads: Optional[int] = None,
) -> None:
    """
    Run the single-PDF pipeline over every PDF in `pdf_dir`.

    Args:
        pdf_dir (Path): Directory containing PDF files to process.
        output_dir (Path): Directory to save output for all PDFs.
        model_name (str): Name of the DeepSeek-OCR model to use.
        dpi (int): Dots per inch for image quality when converting PDF to images.
        num_processes (int, optional): Number of processes for parallel PDF conversion.
        num_threads (int, optional): Number of threads for parallel image saving.

    Raises:
        FileNotFoundError: If `pdf_dir` does not exist.
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF dir not found: {pdf_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list(_iter_pdfs(pdf_dir))
    if not pdfs:
        logger.warning("No PDF files found in %s", pdf_dir)
        return

    logger.info("Found %d PDF(s) in %s", len(pdfs), pdf_dir)

    # Initialize OCR engine once for reuse across all PDFs
    logger.info("Initializing DeepSeek-OCR engine with model: %s", model_name)
    with quiet.quiet_stdio():
        ocr_engine = DeepSeekOCREngine(model_name=model_name)

    for idx, pdf_path in enumerate(pdfs, start=1):
        pdf_out_dir = output_dir / pdf_path.stem
        logger.info("[%d/%d] Processing %s --> %s", idx, len(pdfs), pdf_path, pdf_out_dir)

        # call the normal, single-PDF pipeline
        run_pdf_pipeline(
            pdf_path=pdf_path,
            output_dir=pdf_out_dir,
            model_name=model_name,
            dpi=dpi,
            num_processes=num_processes,
            num_threads=num_threads,
            ocr_engine=ocr_engine,
        )

    logger.info("Batch finished. Outputs --> %s", output_dir)

