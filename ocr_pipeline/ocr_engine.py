# ocr_pipeline/ocr_engine.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image, ImageOps

class DeepSeekOCREngine:
    """
    Thin wrapper over vLLM-powered DeepSeek-OCR for single image to markdown inference.
    """
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-OCR",
        gpu_memory_utilization: float = 0.7,
    ) -> None:
        """
        Args:
            model_name (str): Name of the DeepSeek-OCR model.
            gpu_memory_utilization (float): Fraction of GPU memory to utilize.
        """
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor],
            # NOTE: Adjust GPU memory utilization as needed. Without this configuration,
            # there may be a memory(VRAM) allocation error on GPUs with limited memory (e.g., 16GB).
            gpu_memory_utilization=gpu_memory_utilization,
        )

        # fixed prompt for now
        self.prompt = "<image>\n<|grounding|>Convert the document to markdown."

    def image_to_markdown(self, image_path: Path) -> str:
        """
        Run OCR on a single image file and return raw model text.

        Args:
            image_path (Path): Path to the input image file.

        Returns:
            str: OCR output in markdown format.
        """
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

        model_input: Dict[str, Any] = {
            "prompt": self.prompt,
            "multi_modal_data": {"image": image},
        }

        sampling_param = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822}, # <td>, </td>
            ),
            skip_special_tokens=False,
        )

        outputs = self.llm.generate(model_input, sampling_param, use_tqdm=False)  # type: ignore
        text_output = outputs[0].outputs[0].text
        return text_output

