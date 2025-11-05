# pdfscribe2ds

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
