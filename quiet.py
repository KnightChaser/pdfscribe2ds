# quiet.py
import os
import warnings
import logging
import contextlib

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("RAY_DEDUP_LOGS", "1")
os.environ.setdefault("NCCL_DEBUG", "ERROR")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ["GLOG_minloglevel"] = "2"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"


# Python-level warnings & logging
warnings.filterwarnings("ignore")

def apply_library_quiet_logging() -> None:
    """
    Set logging levels for various libraries to ERROR to suppress verbose output.
    """
    for name in ("vllm", "transformers", "torch", "torch.distributed", "PIL", "pdf2image"):
        logging.getLogger(name).setLevel(logging.ERROR)

@contextlib.contextmanager
def quiet_stdio():
    apply_library_quiet_logging()
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        yield
