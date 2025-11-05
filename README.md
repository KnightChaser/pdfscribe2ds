# pdfscribe2ds

> **(WIP)** A tool to convert PDF documents to Markdown format using AI-powered OCR and optional image captioning.


`pdfscribe2ds` processes PDF files by converting them to images, extracting text and structure using `DeepSeek-OCR`, and generating Markdown files. It can also add captions to images referenced in the Markdown using `DeepSeek-VL2` models.

## Technical Aspects

- Converts PDFs to high-resolution images using `pdf2image`
- Uses `DeepSeek-OCR` model for optical character recognition and layout analysis
- Generates Markdown with embedded images and text
- Optional captioning of images with `DeepSeek-VL2` vision-language models
- Supports batch processing of multiple PDFs
- CLI interface built with `Typer`

## Setup

1. Installing vLLM

```
uv venv
source .venv/bin/activate
uv sync
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
uv pip install timm
```

Currently, you have to install vLLM nightly builds to get the features to use DeepSeek-based models. (Required VLLM: `>=0.11.1`).
Refer to the [related issue](https://github.com/vllm-project/vllm/issues/28030) for more details.
