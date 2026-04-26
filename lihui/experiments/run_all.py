#!/usr/bin/env python3
"""Run a sweep of real-vLLM prefix-cache experiments sequentially.

Each experiment launches a fresh vLLM in its own subprocess — vLLM holds
most of the GPU and cannot be cleanly re-initialised in-process.

Two sweep axes:
  hbm_strategy  ∈ {none, all, align}
  dram_strategy ∈ {none, vllm-native, lmcache (deferred)}
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import DEFAULT_MODEL_NAME


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          CONFIGURATION                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

DATASET          = "swesmith"
PAGE_SIZE        = 32
ORDERING         = "timestamp"
TOKENIZER: Optional[str] = None            # None = use each experiment's model tokenizer
SEED             = 0
MAX_REQUESTS     = 500

MODEL_NAME       = DEFAULT_MODEL_NAME
DTYPE            = "auto"
MAX_MODEL_LEN    = None
GPU_MEM_UTIL     = 0.9
ENFORCE_EAGER    = True                   # compile/capture overhead dominates short sweeps
TENSOR_PARALLEL  = 1
MAX_GEN_TOKENS   = 1
NVML_SAMPLE      = True
WRITE_CSV        = True
OVERWRITE_CSV    = False

# Default sweep: the three HBM modes with no DRAM tier, plus
# all × vllm-native for DRAM offload. align × vllm-native is not supported by
# vLLM because mamba block-aligned split rejects external KV connectors.
EXPERIMENTS: List[Dict[str, Any]] = []
for hbm in ["none", "all", "align"]:
    EXPERIMENTS.append(dict(hbm_strategy=hbm, dram_strategy="none"))
EXPERIMENTS.append(dict(hbm_strategy="all", dram_strategy="vllm-native",
                        cpu_capacity="4.0"))

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       END OF CONFIGURATION                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


def _cmd_for(cfg: Dict[str, Any]) -> List[str]:
    cmd = [
        sys.executable,
        str(_ROOT / "experiments" / "run_one.py"),
        "--dataset",        str(cfg.get("dataset", DATASET)),
        "--page-size",      str(cfg.get("page_size", PAGE_SIZE)),
        "--ordering",       str(cfg.get("ordering", ORDERING)),
        "--seed",           str(cfg.get("seed", SEED)),
        "--max-requests",   str(cfg.get("max_requests", MAX_REQUESTS)),
        "--model",          str(cfg.get("model", MODEL_NAME)),
        "--hbm-strategy",   str(cfg["hbm_strategy"]),
        "--dram-strategy",  str(cfg["dram_strategy"]),
        "--hbm-capacity",   str(cfg.get("hbm_capacity", "auto")),
        "--cpu-capacity",   str(cfg.get("cpu_capacity", "4.0")),
        "--dtype",          str(cfg.get("dtype", DTYPE)),
        "--gpu-memory-utilization", str(cfg.get("gpu_memory_utilization", GPU_MEM_UTIL)),
        "--tensor-parallel-size",   str(cfg.get("tensor_parallel_size", TENSOR_PARALLEL)),
        "--max-gen-tokens", str(cfg.get("max_gen_tokens", MAX_GEN_TOKENS)),
    ]
    tokenizer = cfg.get("tokenizer", TOKENIZER)
    if tokenizer is not None:
        cmd += ["--tokenizer", str(tokenizer)]
    if cfg.get("max_model_len", MAX_MODEL_LEN) is not None:
        cmd += ["--max-model-len", str(cfg.get("max_model_len", MAX_MODEL_LEN))]
    if not cfg.get("enforce_eager", ENFORCE_EAGER):
        cmd += ["--no-enforce-eager"]
    if cfg.get("nvml_sample", NVML_SAMPLE):
        cmd += ["--nvml-sample"]
    if not cfg.get("write_csv", WRITE_CSV):
        cmd += ["--no-write-csv"]
    if cfg.get("overwrite_csv", OVERWRITE_CSV):
        cmd += ["--overwrite-csv"]
    return cmd


def _label(cfg: Dict[str, Any]) -> str:
    return (
        f"{cfg.get('dataset', DATASET)} "
        f"ps={cfg.get('page_size', PAGE_SIZE)} "
        f"{cfg.get('ordering', ORDERING)} "
        f"hbm={cfg['hbm_strategy']} dram={cfg['dram_strategy']} "
        f"cap={cfg.get('hbm_capacity', 'auto')}"
    )


def main() -> None:
    print(f"=== Running {len(EXPERIMENTS)} experiment(s) ===", flush=True)
    t0 = time.perf_counter()
    failures: List[str] = []
    for i, cfg in enumerate(EXPERIMENTS, 1):
        label = _label(cfg)
        cmd = _cmd_for(cfg)
        print(f"\n[{i}/{len(EXPERIMENTS)}] {label}\n  $ {shlex.join(cmd)}", flush=True)
        t_s = time.perf_counter()
        rc = subprocess.call(cmd)
        elapsed = time.perf_counter() - t_s
        if rc != 0:
            print(f"  !! failed (exit {rc}) after {elapsed:.1f}s", flush=True)
            failures.append(label)
        else:
            print(f"  ok ({elapsed:.1f}s)", flush=True)

    total = time.perf_counter() - t0
    if failures:
        print(f"\n{len(failures)}/{len(EXPERIMENTS)} failed:", file=sys.stderr)
        for l in failures:
            print(f"  {l}", file=sys.stderr)
        sys.exit(1)
    print(f"\nAll {len(EXPERIMENTS)} experiments completed in {total:.1f}s.")


if __name__ == "__main__":
    main()
