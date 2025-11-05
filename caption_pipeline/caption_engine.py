# caption_pipeline/caption_engine.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional
from PIL import Image, ImageOps
from vllm import LLM, EngineArgs, SamplingParams

@dataclass(frozen=True)
class CaptionerConfig:
    model_name: str = "deepseek-ai/deepseek-vl2-tiny" # fixed
    max_model_len: int = 4096
    max_num_seqs: int = 1
    gpu_memory_utilization: float = 0.7
    seed: Optional[int] = None
    min_side: int = 128 # ensure min(H, W) >= this
    max_side: int = 2048 # avoid absurdly large images

    def to_engine_args(self) -> EngineArgs:
        """
        Convert the CaptionerConfig dataclass to vLLM EngineArgs
        to utilize the model on the vLLM framework stably.

        Returns:
            EngineArgs: The corresponding vLLM EngineArgs object.
        """
        return EngineArgs(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
            limit_mm_per_prompt={"image": 1},
            seed=self.seed
        )

def _prepare_image_for_vl2(img: Image.Image, min_side: int, max_side: int) -> Image.Image:
    """
    Ensure the image has sane size for the VL2 model (avoid zero patch grid),
    preserve aspect ratio, and respect an upper bound as well.

    Args:
        img (PIL.Image): The input image to prepare.
        min_side (int): Minimum size for the smaller side of the image.
        max_side (int): Maximum size for the larger side of the image.

    Returns:
        PIL.Image: The resized image.
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size

    # Upscale the image if it's too small
    m = min(w, h)
    if m < min_side and m > 0:
        scale = float(min_side) / float(m)
        nw = int(round(w * scale))
        nh = int(round(h * scale))
        nw = max(1, nw)
        nh = max(1, nh)
        img = img.resize((nw, nh), resample=Image.Resampling.BICUBIC)
        w, h = nw, nh

    # Downscale the image if it's too large
    M = max(w, h)
    if M > max_side:
        scale = float(max_side) / float(M)
        nw = int(round(w * scale))
        nh = int(round(h * scale))
        nw = max(1, nw)
        nh = max(1, nh)
        img = img.resize((nw, nh), resample=Image.Resampling.BICUBIC)

    return img

class DeepSeekVL2Captioner:
    """
    Thin captioner for DeepSeek-VL2 (tiny by default). Produces concise,
    technical-report-friendly captions for a single PIL image.
    """
    def __init__(self, cfg: CaptionerConfig = CaptionerConfig()) -> None:
        self.cfg = cfg
        engine_dict = cfg.to_engine_args()

        # vLLM's LLM constructor accepts `seed` directly
        extra = {"seed": self.cfg.seed} if cfg.seed is not None else {}
        self.llm = LLM(**(asdict(engine_dict) | extra))

        # Default caption prompt
        question="Describe the visual elements of this image exactly as they appear, and then interpret what are the given image(diagram) is meaning."
        self.prompt_template = f"<|User|>: image_1:<image>\n{question}\n\n<|Assistant|>:"

        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=256,
        )

    def caption(self, image: Image.Image, prompt_override: str | None = None) -> str:
        """
        Generate a caption for the given image using DeepSeek-VL2.

        Args:
            image (PIL.Image): The input image to caption.
            prompt_override (str | None): Optional custom prompt to use instead of the default.

        Returns:
            str: The generated caption text.
        """
        prompt = prompt_override or self.prompt_template
        safe_image = _prepare_image_for_vl2(image, self.cfg.min_side, self.cfg.max_side)

        outputs = self.llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": [safe_image]},  # list[PIL.Image]
            },
            sampling_params=self.sampling,
        )
        return outputs[0].outputs[0].text.strip()
