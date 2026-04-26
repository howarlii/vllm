#!/usr/bin/env python3
"""Drive one real-vLLM prefix-cache experiment and optionally write results.

Two independent strategy axes:
  --hbm-strategy  ∈ {none, all, align}      — vLLM mamba_cache_mode
  --dram-strategy ∈ {none, vllm-native, lmcache}
                                            — KV-transfer connector choice

PCIe byte signals emitted to CSV:
  external_kv_transfer_tokens       from scheduler.connector_prefix_cache_stats
  pcie_bytes_restore_estimate       external_kv_transfer_tokens × kv_bytes_per_token
  pcie_nvml_bytes_{tx,rx,total}     NVML PCIe throughput integration (if --nvml-sample)
  pcie_*_bandwidth_gb_per_s         average and peak PCIe bandwidth (GB/s)
  pcie_total_transfer_bytes         canonical: NVML total if sampled else estimate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import DEFAULT_MODEL_NAME, RESULTS_DIR
from src.metrics import compute_run_metrics
from src.request_generator import TokenizedRequest
from src.vllm_runner import run_live
from experiments.runner import (
    persist_result_row,
    prepare_requests,
)


def _parse_capacity(spec: str):
    s = spec.strip().lower()
    if s in ("", "inf", "none", "unlimited", "auto"):
        return None
    return float(s.replace("gb", "").strip())


def _actual_hbm_capacity_spec(metrics) -> str:
    """Render the engine-created HBM KV cache capacity as concrete space."""
    capacity_bytes = int(getattr(metrics, "hbm_capacity_bytes", 0) or 0)
    if capacity_bytes > 0:
        return f"{capacity_bytes / (1024 ** 3):.3f}GB"

    capacity_tokens = int(getattr(metrics, "hbm_capacity_tokens", 0) or 0)
    bytes_per_token = int(getattr(metrics, "pcie_kv_bytes_per_token", 0) or 0)
    if capacity_tokens > 0 and bytes_per_token > 0:
        capacity_gb = capacity_tokens * bytes_per_token / (1024 ** 3)
        return f"{capacity_gb:.3f}GB"
    if capacity_tokens > 0:
        return f"{capacity_tokens}tokens"
    return "unknown"


def _truncate_to_model_context(
    reqs: list[TokenizedRequest],
    *,
    max_model_len: int | None,
    max_gen_tokens: int,
) -> list[TokenizedRequest]:
    """Clamp prepared requests to the live vLLM context window.

    Tokenization caches can be shared with simulator runs whose max input
    length is much larger than this live engine's ``max_model_len``. vLLM
    rejects token-id prompts longer than the context window, so truncate here
    after loading cached requests.
    """
    if max_model_len is None:
        return reqs
    max_input_tokens = max(1, int(max_model_len) - max(0, int(max_gen_tokens)))
    truncated = 0
    out: list[TokenizedRequest] = []
    for req in reqs:
        token_ids = req.token_ids
        if len(token_ids) > max_input_tokens:
            token_ids = token_ids[:max_input_tokens]
            truncated += 1
        out.append(TokenizedRequest(
            token_ids=token_ids,
            group_id=req.group_id,
            meta=req.meta,
        ))
    if truncated:
        print(
            f"[prep] truncated {truncated}/{len(reqs)} requests to "
            f"{max_input_tokens} input tokens for max_model_len={max_model_len}",
            flush=True,
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run one real-vLLM prefix-cache experiment")
    # Dataset / ordering (same as simulator)
    p.add_argument("--dataset", default="swesmith",
                   help="loogle | narrativeqa | sharegpt_90k_raw | swesmith | oasst1")
    p.add_argument("--page-size", type=int, default=32,
                   help="vLLM block_size requested by the experiment.")
    p.add_argument("--ordering", default="timestamp",
                   help="original | min_distance | max_distance | random | timestamp")
    p.add_argument("--sessions-per-second", type=float, default=1.0)
    p.add_argument("--words-per-min", type=float, default=90.0)
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer for request token IDs. Defaults to --model; "
                        "override only when it is vocabulary-compatible with the model.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenize-workers", type=int, default=0)
    p.add_argument("--max-requests", type=int, default=500,
                   help="0 = full dataset")

    # vLLM engine — two strategy axes
    p.add_argument("--model", default=DEFAULT_MODEL_NAME)
    p.add_argument("--hbm-strategy", default="all",
                   choices=["none", "all", "align"],
                   help="mamba_cache_mode on HBM (GPU)")
    p.add_argument("--dram-strategy", default="none",
                   choices=["none", "vllm-native", "lmcache"],
                   help="KV-transfer connector — none=no offload, "
                        "vllm-native=SimpleCPUOffloadConnector, lmcache=deferred")
    p.add_argument("--hbm-capacity", default="auto",
                   help="GB, or 'auto' to let vLLM profile free memory")
    p.add_argument("--cpu-capacity", default="4.0",
                   help="CPU offload buffer size in GB (only for --dram-strategy vllm-native)")
    p.add_argument("--dtype", default="auto")
    # NOTE: Bamba-9B-v2's default max_model_len is 262144 which does NOT fit
    # on a 24 GB card once the 18 GB model weights are loaded (only ~2 GB KV
    # space left). We cap to 8192 by default — enough for swesmith / loogle /
    # narrativeqa prompts. Override with --max-model-len for long-context runs.
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    p.add_argument("--no-enforce-eager", action="store_true",
                   help="Allow compile/capture (default is --enforce-eager for sweep speed)")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-gen-tokens", type=int, default=1,
                   help="Per-request output length — prefix cache hit happens at "
                        "prefill, so 1 is enough.")

    # PCIe byte measurement — target GPU is auto-detected from the CUDA
    # device that vLLM is using (via PCI bus id lookup in NVML), so there is
    # no --nvml-device flag.
    p.add_argument("--nvml-sample", action="store_true",
                   help="Sample NVML PCIe tx/rx counters in a background thread for "
                        "ground-truth total bytes (aggregate, includes non-KV).")
    p.add_argument("--nvml-interval-s", type=float, default=0.05)

    # Output
    p.add_argument("--no-write-csv", action="store_true",
                   help="Run without updating --out-csv or the JSON sidecar directory.")
    p.add_argument("--overwrite-csv", action="store_true",
                   help="Overwrite an existing CSV row with the same run args. "
                        "Default: keep the existing row.")
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--out-json-dir", type=Path, default=None)
    args = p.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.model
    # if args.hbm_strategy == "align" and args.dram_strategy != "none":
    #     raise SystemExit(
    #         "Unsupported strategy combination: --hbm-strategy align cannot be "
    #         "used with --dram-strategy vllm-native. vLLM's mamba block-aligned "
    #         "split path asserts that external KV connectors are not verified "
    #         "yet. Use --hbm-strategy all --dram-strategy vllm-native, or use "
    #         "--hbm-strategy align --dram-strategy none."
    #     )

    if args.out_csv is None:
        args.out_csv = RESULTS_DIR / f"results_{args.dataset}.csv"
    if args.out_json_dir is None:
        args.out_json_dir = RESULTS_DIR / f"json_{args.dataset}"

    page_size = args.page_size
    cap_gb = _parse_capacity(args.hbm_capacity)
    cpu_cap_gb = float(args.cpu_capacity)

    reqs = prepare_requests(
        args.dataset,
        args.ordering,
        tokenizer_name=args.tokenizer,
        seed=args.seed,
        tokenize_workers=args.tokenize_workers,
        max_requests=args.max_requests or None,
        sessions_per_second=args.sessions_per_second,
        words_per_min=args.words_per_min,
    )
    if not reqs:
        raise SystemExit(f"No requests loaded for dataset {args.dataset!r}")
    reqs = _truncate_to_model_context(
        reqs,
        max_model_len=args.max_model_len,
        max_gen_tokens=args.max_gen_tokens,
    )

    state = run_live(
        reqs,
        model=args.model,
        block_size=page_size,
        hbm_strategy=args.hbm_strategy,
        dram_strategy=args.dram_strategy,
        hbm_capacity_gb=cap_gb,
        cpu_capacity_gb=cpu_cap_gb,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.no_enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        max_gen_tokens=args.max_gen_tokens,
        nvml_sample=args.nvml_sample,
        nvml_interval_s=args.nvml_interval_s,
    )

    metrics = compute_run_metrics(state)
    actual_page_size = int(getattr(state, "block_size", page_size) or page_size)
    hbm_capacity_spec = _actual_hbm_capacity_spec(metrics)
    row = {
        # Dataset / request-prep args.
        "dataset": args.dataset,
        "ordering": args.ordering,
        "sessions_per_second": args.sessions_per_second,
        "words_per_min": args.words_per_min,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "tokenize_workers": args.tokenize_workers,
        "max_requests": args.max_requests,
        "requested_page_size": args.page_size,
        "page_size": actual_page_size,
        # vLLM engine args.
        "model_name": args.model,
        "hbm_strategy": args.hbm_strategy,
        "dram_strategy": args.dram_strategy,
        "hbm_capacity_spec": hbm_capacity_spec,
        "cpu_capacity_spec": str(args.cpu_capacity) if args.dram_strategy == "vllm-native" else "",
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": not args.no_enforce_eager,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_gen_tokens": args.max_gen_tokens,
        # Measurement args.
        "nvml_sample": args.nvml_sample,
        "nvml_interval_s": args.nvml_interval_s,
        "metrics": metrics.to_dict(),
    }
    persist_status = None
    if not args.no_write_csv:
        persist_status = persist_result_row(
            args.out_csv,
            args.out_json_dir,
            row,
            overwrite_existing=args.overwrite_csv,
        )

    page_size_note = (
        f" requested_ps={page_size}" if actual_page_size != page_size else ""
    )
    print(
        f"\n=== {args.dataset} ps={actual_page_size}{page_size_note} "
        f"ord={args.ordering} "
        f"hbm={args.hbm_strategy} (hbm_capacity_spec: {hbm_capacity_spec}) dram={args.dram_strategy} {(f'(cpu_capacity: {args.cpu_capacity})') if args.dram_strategy == 'vllm-native' else ''}) ===\n"
        f"  n_req={metrics.num_requests} total_in={metrics.total_input_tokens} "
        f"infer_time={metrics.inference_time_s:.3f}s\n"
        f"  hbm_hr={metrics.hbm_token_hit_rate:.4f} "
        f"dram_hr={metrics.dram_token_hit_rate:.4f}  "
        f"load/compute={metrics.load_compute_ratio}\n"
        f"  peak={metrics.peak_cached_tokens} avg={metrics.avg_cached_tokens:.1f} "
        f"(cap={metrics.hbm_capacity_tokens} tokens)\n"
        f"  dram_peak={metrics.dram_peak_cached_tokens} "
        f"dram_avg={metrics.dram_avg_cached_tokens:.1f}\n"
        f"  external_tokens={metrics.external_kv_transfer_tokens} "
        f"bpt={metrics.pcie_kv_bytes_per_token}\n"
        f"  pcie: est={metrics.pcie_bytes_restore_estimate/1e6:.1f} MB  "
        f"nvml={metrics.pcie_nvml_bytes_total/1e6:.1f} MB "
        f"(avail={metrics.nvml_available})\n"
        f"  pcie_bw: avg={metrics.pcie_avg_bandwidth_gb_per_s:.3f} GB/s "
        f"peak={metrics.pcie_peak_bandwidth_gb_per_s:.3f} GB/s\n"
        f"  → canonical pcie_total_transfer_bytes = "
        f"{metrics.pcie_total_transfer_bytes/1e6:.1f} MB"
    )
    if persist_status is not None:
        print(f"  csv={persist_status} path={args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
