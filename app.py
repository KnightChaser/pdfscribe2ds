# app.py
from pathlib import Path
import typer

from ocr_pipeline.pipeline import run_pdf_pipeline

app = typer.Typer(help="PDF --> images --> DeepSeek-OCR --> Markdown")

@app.command("pdf")
def pdf_to_md(
    pdf_path: Path = typer.Argument(..., exists=True, readable=True, help="Input PDF"),
    output_dir: Path = typer.Option(Path("./output"), help="Where to store images and markdown"),
    model_name: str = typer.Option("deepseek-ai/DeepSeek-OCR", help="DeepSeek-OCR model name"),
    dpi: int = typer.Option(200, help="DPI for pdf2image"),
    num_processes: int = typer.Option(None, help="Number of processes for parallel PDF conversion (default: CPU count // 2)"),
    num_threads: int = typer.Option(None, help="Number of threads for parallel image saving (default: 4)"),
) -> None:
    """
    Convert a PDF to per-page Markdown files + cropped assets.
    """
    run_pdf_pipeline(
        pdf_path=pdf_path,
        output_dir=output_dir,
        model_name=model_name,
        dpi=dpi,
        num_processes=num_processes,
        num_threads=num_threads,
    )
    typer.echo(f"[OK] PDF processed --> {output_dir}")

if __name__ == "__main__":
    app()
