# lihui/ — vLLM prefix-cache real-run experiment framework

A live-vLLM counterpart to `../LLM-prefix-caching-simulator/`.  Same
request-preparation pipeline, same dataset / ordering axes, but the
simulator's `KVCacheSimulator` is replaced with an actual vLLM engine.

## Layout

```
lihui/
├── src/
│   ├── config.py             # paths + DEFAULT_MODEL_NAME / TOKENIZER
│   ├── datasets_loader.py    # [same as simulator] HF → RawRequest
│   ├── request_generator.py  # [same as simulator] tokenize + order
│   ├── vllm_runner.py        # real LLM() driver (replaces cache_simulator.py)
│   ├── metrics.py            # trimmed RunMetrics (no FLOP/wall-clock/branch)
│   └── nvml_util.py          # background PCIe throughput sampler
├── experiments/
│   ├── runner.py             # RESULT_CSV_FIELDS + persist_result_row + prepare_requests
│   ├── run_one.py            # CLI — one experiment
│   └── run_all.py            # CLI — sweep (subprocess-sequential)
├── data/                     # symlinks into ../LLM-prefix-caching-simulator/data/data/
│   ├── mooncake_trace/       → simulator's mooncake_trace
│   ├── prepared/             → simulator's prepared (pickle cache, shared)
│   └── tokenized/            → simulator's tokenized (jsonl cache, shared)
├── results/                  # CSV + JSON output
└── legacy/                   # pre-framework ad-hoc scripts / logs / notes
```

## Strategy axes

vLLM's native HBM *eviction* is hard-wired LRU (no knob) — so the
simulator's `--hbm_strategy` flag is repurposed to select the
**mamba_cache_mode** axis on the GPU tier. The DRAM tier becomes the
connector choice.

### `--hbm-strategy` (mamba_cache_mode)

| value     | vLLM config                                                                      |
|-----------|----------------------------------------------------------------------------------|
| `none`    | `enable_prefix_caching=False` — no prefix caching, every prompt re-prefilled    |
| `all`     | `enable_prefix_caching=True`  — model must `SupportsMambaPrefixCaching`         |
| `align`   | `enable_prefix_caching=True` + `mamba_cache_mode="align"` + `enable_chunked_prefill=True` |

See `legacy/mamba_prefix_cache_notes.md` for what each mode does mechanically.

### `--dram-strategy` (KV-transfer connector)

| value          | vLLM config                                                                                                       |
|----------------|-------------------------------------------------------------------------------------------------------------------|
| `none`         | No `kv_transfer_config` — single-tier HBM only                                                                    |
| `vllm-native`  | `kv_transfer_config=KVTransferConfig(kv_connector="SimpleCPUOffloadConnector", kv_role="kv_both", cpu_bytes_to_use=…)`  |
| `lmcache`      | Deferred (upstream doesn't yet support hybrid-Mamba models) — raises `NotImplementedError`                        |

`vllm-native` requires prefix caching on, so it must be combined with
`--hbm-strategy all|align`, not `none`.

## PCIe transfer bytes

Three independent signals are emitted to the CSV. None of them
individually is perfect; cross-reference in analysis.

| CSV column                          | Source                                                                              | Covers           | Accuracy                                |
|-------------------------------------|-------------------------------------------------------------------------------------|------------------|-----------------------------------------|
| `external_kv_transfer_tokens`       | `scheduler.connector_prefix_cache_stats.hits`                                       | restore only     | **exact** in tokens                    |
| `pcie_kv_bytes_per_token`           | derived from `KVCacheSpec.page_size_bytes / block_size` at runtime                  | —                | model-exact                             |
| `pcie_bytes_restore_estimate`       | `external_kv_transfer_tokens × pcie_kv_bytes_per_token`                             | restore only     | exact modulo derivation                 |
| `pcie_nvml_bytes_{tx,rx,total}`     | `nvmlDeviceGetPcieThroughput` integrated at 50 ms intervals in a background thread | tx+rx, all traffic | ground truth (incl. weights / activations) |
| `inference_time_s`                  | wall-clock time around the serial `LLM.generate()` loop                             | full run         | measured                                |
| `pcie_avg_bandwidth_gb_per_s`       | `pcie_total_transfer_bytes / sampled_or_inference_seconds`                          | canonical        | best-effort                             |
| `pcie_peak_bandwidth_gb_per_s`      | max sampled NVML tx+rx throughput                                                   | tx+rx, all traffic | sampled peak                            |
| `pcie_total_transfer_bytes`         | NVML total if sampled, else restore estimate                                        | canonical        | best-effort                             |

Write-half (HBM→DRAM offload) has no native vLLM counter — NVML is the
only source until `SimpleCPUOffloadWorker.handle_transfers()` is
instrumented.

Enable NVML via `--nvml-sample` (or default-on in `run_all.py`).

## Model memory caveat

Bamba-9B-v2 advertises `max_model_len=262144` (256 K) but needs ≈ 33 GiB of
KV cache for one such sequence — on a 24 GB RTX 3090 only ~2 GiB is left
after the 18 GiB model weights are loaded. `run_one.py` therefore defaults
`--max-model-len=8192`, enough for every dataset we ship. Raise it only if
(a) prompts are longer AND (b) the card has more VRAM headroom.

## Quick start

```bash
cd /data3/howarli/dev/vllm
source .venv/bin/activate

# One experiment
python lihui/experiments/run_one.py \
    --dataset swesmith --page-size 32 --ordering timestamp \
    --hbm-strategy all --dram-strategy vllm-native \
    --max-requests 500 --nvml-sample

# Full sweep (3 HBM modes × 2 DRAM modes = 5 configs on swesmith)
python lihui/experiments/run_all.py
```
