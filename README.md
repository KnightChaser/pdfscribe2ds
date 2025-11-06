# pdfscribe2ds

> A tool to convert PDF documents to Markdown format using AI-powered OCR and optional image captioning.


`pdfscribe2ds` processes PDF files by converting them to images, extracting text and structure using `DeepSeek-OCR`, and generating Markdown files. It can also add captions to images referenced in the Markdown using `DeepSeek-VL2` models.

## Technical Aspects

- Converts PDFs to high-resolution images using `pdf2image`
- Uses `DeepSeek-OCR` model for optical character recognition and layout analysis
- Generates Markdown with embedded images and text
- Optional captioning of images with `DeepSeek-VL2` vision-language models
- Supports batch processing of multiple PDFs
- CLI interface built with `Typer`

## Installation

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
uv pip install timm
```

Currently, you have to install vLLM nightly builds to get the features to use DeepSeek-based models. (Required VLLM: `>=0.11.1`).
Refer to the [related issue](https://github.com/vllm-project/vllm/issues/28030) for more details.

## Usage

### Convert PDF to Markdown

```bash
python app.py pdf /path/to/document.pdf
```

Output: `output/markdown/page-001.md`, `output/images/page-001.png`, etc.

### Batch Process PDFs

```bash
python app.py pdf-batch /path/to/pdf/directory
```

Processes all PDFs in directory, each in its own subdirectory under `outputs/`.

### Add Captions to Images

```bash
python app.py caption /path/to/output/directory
```

Adds captions to images referenced in Markdown files using DeepSeek-VL2.

### Batch Caption

```bash
python app.py caption-batch /path/to/outputs/root
```

Captions all processed PDFs under the root directory.

## Options

Run `python app.py --help` or `python app.py <command> --help` for detailed options.


## Output Structure

```
output/
├── images/
│   ├── page-001.png
│   └── page-002.png
├── markdown/
│   ├── page-001.md
│   ├── page-002.md
│   ├── page-001_assets/
│   └── page-002_assets/
```

## Requirements

- Python >=3.12
- CUDA-compatible GPU (for vLLM)
- Sufficient VRAM for models (40GiB+ recommended, because it runs locally)