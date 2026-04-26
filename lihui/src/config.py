"""Paths and defaults for the real-vLLM prefix-cache experiment framework.

Simulator-only hardware knobs (GPU FLOPS, PCIe bandwidth) are intentionally
absent here: we run vLLM for real, so wall-clock and transfer bytes come from
live instrumentation rather than back-of-envelope constants.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_DATA_ROOT = Path("/data3/howarli/")
_HOME = Path.home()


def ensure_hf_cache_dirs() -> None:
    """Point HF / datasets cache under /data/howarli when available.

    Unlike the simulator, we do NOT hide GPUs from this process — vLLM needs
    them. Tokenization still parallelises on CPU workers, so we only dampen
    tokenizer's own threading when multiprocessing is involved.
    """
    data_root = _DEFAULT_DATA_ROOT if _DEFAULT_DATA_ROOT.is_dir() else _HOME
    hf_home = data_root / ".cache" / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def ensure_cpu_only() -> None:
    """Shim kept for compatibility with request_generator.load_or_tokenize.

    The simulator called this to hide GPUs during tokenization; for us the
    tokenizer also runs CPU-side, but we keep any visible GPUs in case a
    tokenizer (e.g. sentencepiece-cuda) would otherwise fail. This is a no-op
    unless explicitly opted in via KV_SIM_HIDE_CUDA_FOR_TOKENIZER=1.
    """
    if os.environ.get("KV_SIM_HIDE_CUDA_FOR_TOKENIZER", "").lower() in ("1", "true", "yes"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Default tokenizer used for text datasets (overridable via env / CLI).
DEFAULT_TOKENIZER_NAME = os.environ.get("KV_SIM_TOKENIZER", "Qwen/Qwen3-0.6B")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TOKEN_CACHE_DIR = DATA_DIR / "tokenized"
PREPARED_CACHE_DIR = DATA_DIR / "prepared"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_MODEL_NAME = os.environ.get("VLLM_EXP_MODEL", "ibm-ai-platform/Bamba-9B-v2")
