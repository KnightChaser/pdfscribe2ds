# caption_pipeline/batch.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import logging

import quiet
from .caption_engine import DeepSeekVL2Captioner, CaptionerConfig
from .caption_pipeline import caption_markdown_file, CaptionRewrite

logger = logging.getLogger("pdfscribe2ds")

def _iter_caption_targets(root_output_dir: Path) -> Iterable[Path]:
    """
    Yield each PDF's output directory under root_output_dir that contains markdown/.
    Example:
      root_output_dir/
        001-foo/markdown/
        002-bar/markdown/
    -> yields root_output_dir/001-foo, root_output_dir/002-bar

    Args:
        root_output_dir (Path): The root output directory containing per-PDF subdirectories.

    Returns:
        Iterable[Path]: Paths to per-PDF output directories with markdown/.
    """
    if not root_output_dir.exists():
        raise FileNotFoundError(f"Output root directory not found: {root_output_dir}")
    for p in sorted(root_output_dir.iterdir()):
        if p.is_dir() and (p / "markdown").exists():
            yield p

def run_caption_batch(
    outputs_root: Path,
    caption_model: str = "deepseek-ai/deepseek-vl2-tiny",
    gpu_mem: float = 0.7,
    seed: Optional[int] = None,
    prompt: Optional[str] = None,
    rewrite: CaptionRewrite = CaptionRewrite.APPEND,
) -> None:
    """
    Run captioning over every finished PDF output folder under outputs_root.
    Each target must contain a markdown/ directory with page-*.md files.

    Model is initialized ONCE and reused for all targets.

    Args:
        outputs_root (Path): Root directory containing per-PDF output subdirectories.
        caption_model (str): The caption model to use.
        gpu_mem (float): The GPU memory utilization fraction.
        seed (Optional[int]): Optional random seed for reproducibility.
        prompt (Optional[str]): Optional prompt to override the default captioning prompt.
        rewrite (CaptionRewrite): Whether to append or replace captions in the markdown.

    Raises:
        FileNotFoundError: If outputs_root does not exist or contains no valid targets.
    """
    targets = list(_iter_caption_targets(outputs_root))
    if not targets:
        logger.warning("No caption targets found under %s (no subdirs with markdown/)", outputs_root)
        return

    logger.info("Found %d target(s) under %s", len(targets), outputs_root)
    logger.info("Initializing captioner: %s", caption_model)

    with quiet.quiet_stdio():
        captioner = DeepSeekVL2Captioner(
            CaptionerConfig(
                model_name=caption_model,
                gpu_memory_utilization=gpu_mem,
                seed=seed,
                min_side=128,
                max_side=2048,
            )
        )

    total_changed = 0
    for idx, tgt in enumerate(targets, start=1):
        md_dir = tgt / "markdown"
        logger.info("[%d/%d] Captioning %s", idx, len(targets), tgt)

        changed = 0
        for md_file in sorted(md_dir.glob("*.md")):
            try:
                if caption_markdown_file(
                    md_file,
                    captioner=captioner,
                    prompt_override=prompt,
                    rewrite=rewrite,
                ):
                    changed += 1
            except Exception:
                logger.exception("Failed to process %s; skipping.", md_file.name)

        logger.info("Updated %d file(s) in %s", changed, tgt)
        total_changed += changed

    logger.info("Caption batch finished. %d file(s) updated in total.", total_changed)
