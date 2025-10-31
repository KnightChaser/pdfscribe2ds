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
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# Silence specific library loggers
logging.getLogger('vllm').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('torch.distributed').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('pdf2image').setLevel(logging.ERROR)

# ultra-quiet context (redirects C/C++ stderr spam)
@contextlib.contextmanager
def quiet_stdio():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
