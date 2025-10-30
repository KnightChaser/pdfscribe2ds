# ocr_pipeline/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class PipelineConfig:
    pdf_path: Path
    output_dir: Path
    model_name: str = "deepseek-ai/DeepSeek-OCR" # fixed
    dpi: int = 200
