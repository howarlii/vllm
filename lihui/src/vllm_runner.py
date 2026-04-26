"""Live-vLLM driver that mirrors the simulator's KVCacheSimulator loop.

Two axes:

  --hbm-strategy ∈ {none, all, align}
      Maps to vLLM's mamba_cache_mode. See lihui/legacy/mamba_prefix_cache_notes.md
      for what each value actually does. In vLLM's own source:
      `MambaModelConfig.verify_and_update_config` (vllm/model_executor/models/config.py)
      auto-selects `none` (no prefix cache), `all` (multi-block cache,
      needs SupportsMambaPrefixCaching), or `align` (single-block cache + forced
      chunked prefill). We override it explicitly so the three can be swept.

  --dram-strategy ∈ {None, vllm-native, lmcache}
      None         → no KV connector — single-tier HBM only.
      vllm-native  → SimpleCPUOffloadConnector; blocks evicted from GPU are
                     offloaded to pinned CPU memory and restored on hit.
      lmcache      → LMCache connector (deferred — doesn't support hybrid
                     models yet; raises NotImplementedError).

Metrics collected per run:
  * Native HBM prefix-cache hits (from scheduler.kv_cache_manager.prefix_cache_stats)
  * Connector DRAM prefix-cache hits (from RequestOutput.num_external_computed_tokens)
  * Per-request num_cached_tokens (total, local + external)
  * Optional: NVML PCIe tx/rx byte integration around each generate()
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from src.nvml_util import PcieSampler
from src.request_generator import TokenizedRequest


def _lazy_import_vllm():
    # Force vLLM v1 to run the EngineCore in-process. By default it spawns
    # EngineCore as a subprocess, which makes `llm.llm_engine.engine_core` on
    # the client side a thin IPC shim — we cannot reach the real scheduler /
    # KVCacheManager / block pool from here. In-process mode loses a bit of
    # async overlap but our sequential per-request driver doesn't benefit
    # from that anyway, so the trade-off is fine for experiments.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams  # type: ignore
    return LLM, SamplingParams


_VALID_HBM_STRATEGIES = ("none", "all", "align")
_VALID_DRAM_STRATEGIES = ("none", "vllm-native", "lmcache")


# ── Per-run state ───────────────────────────────────────────────────────────


@dataclass
class PerRequestTrace:
    input_tokens: int
    hit_tokens: int        # total cached (local + external)
    miss_tokens: int
    hbm_hit_tokens: int    # native GPU prefix-cache hits
    dram_hit_tokens: int   # connector / external hits


@dataclass
class RunState:
    traces: list[PerRequestTrace] = field(default_factory=list)
    hbm_cache_token_samples: list[int] = field(default_factory=list)
    dram_cache_token_samples: list[int] = field(default_factory=list)
    block_size: int = 0
    # Logical token capacity after vLLM's block/layer grouping.
    hbm_capacity_tokens: int = 0
    # Profiled KV memory budget in bytes, matching vLLM's
    # "Available KV cache memory" log line.
    hbm_capacity_bytes: int = 0

    # PCIe byte estimates, two independent sources.
    pcie_external_tokens: int = 0          # sum of connector hits (tokens)
    pcie_kv_bytes_per_token: int = 0       # derived from model config (see below)
    nvml_pcie_tx_bytes: int = 0
    nvml_pcie_rx_bytes: int = 0
    nvml_sampling_duration_s: float = 0.0
    nvml_pcie_peak_bytes_per_s: float = 0.0
    nvml_available: bool = False
    inference_time_s: float = 0.0

    # Engine knobs echoed back (for CSV identity).
    engine_hbm_strategy: str = ""
    engine_dram_strategy: str = ""
    native_prefix_queries: int = 0
    native_prefix_hits: int = 0
    connector_prefix_queries: int = 0
    connector_prefix_hits: int = 0


# ── Strategy → vLLM config mapping ──────────────────────────────────────────


def _build_llm_kwargs(
    *,
    model: str,
    block_size: int,
    hbm_strategy: str,
    dram_strategy: str,
    hbm_capacity_gb: float | None,
    cpu_capacity_gb: float,
    dtype: str,
    max_model_len: int | None,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    tensor_parallel_size: int,
    trust_remote_code: bool,
) -> dict:
    if hbm_strategy not in _VALID_HBM_STRATEGIES:
        raise ValueError(
            f"Unknown hbm_strategy {hbm_strategy!r}; "
            f"expected one of {_VALID_HBM_STRATEGIES}"
        )
    if dram_strategy not in _VALID_DRAM_STRATEGIES:
        raise ValueError(
            f"Unknown dram_strategy {dram_strategy!r}; "
            f"expected one of {_VALID_DRAM_STRATEGIES}"
        )
    if dram_strategy == "lmcache":
        raise NotImplementedError(
            "dram_strategy='lmcache' is deferred — LMCache does not yet support "
            "hybrid-architecture (Mamba) models. Add wiring when upstream lands."
        )

    kwargs: dict = {
        "model": model,
        "block_size": block_size,
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "enforce_eager": enforce_eager,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "disable_log_stats": False,
        # Bamba / hybrid-Mamba models NEED the hybrid KV manager even when a
        # connector is attached; vLLM auto-disables it otherwise. Force on.
        "disable_hybrid_kv_cache_manager": False,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len

    # HBM strategy → mamba_cache_mode + enable_prefix_caching.
    if hbm_strategy == "none":
        kwargs["enable_prefix_caching"] = False
        # 'none' mode is vLLM's default when prefix caching is off; we don't
        # need to pass mamba_cache_mode explicitly.
    elif hbm_strategy == "all":
        kwargs["enable_prefix_caching"] = True
        # 'all' is auto-selected for SupportsMambaPrefixCaching models; if the
        # model doesn't support it vLLM falls back to 'align'. We don't force
        # 'all' because vLLM's internal assert is the right gate.
    elif hbm_strategy == "align":
        kwargs["enable_prefix_caching"] = True
        kwargs["mamba_cache_mode"] = "align"
        kwargs["enable_chunked_prefill"] = True

    # DRAM strategy → kv_transfer_config.
    if dram_strategy == "vllm-native":
        # SimpleCPUOffloadConnector requires prefix caching on — enforce.
        if not kwargs.get("enable_prefix_caching"):
            raise ValueError(
                "dram_strategy='vllm-native' requires prefix caching; "
                "combine with --hbm-strategy all|align (not none)."
            )
        from vllm.config import KVTransferConfig  # type: ignore
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="SimpleCPUOffloadConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "cpu_bytes_to_use": int(cpu_capacity_gb * (1024 ** 3)),
            },
        )

    if hbm_capacity_gb is not None:
        kwargs["kv_cache_memory_bytes"] = int(hbm_capacity_gb * (1024 ** 3))

    return kwargs


# ── Scheduler stat access (best-effort; in-process only) ───────────────────


def _scheduler(llm):
    try:
        return llm.llm_engine.engine_core.engine_core.scheduler  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise RuntimeError(
            "Failed to read vLLM in-process scheduler. "
            "Ensure VLLM_ENABLE_V1_MULTIPROCESSING=0 is honored."
        ) from exc


def _block_pool(scheduler):
    try:
        return scheduler.kv_cache_manager.block_pool
    except AttributeError as exc:
        raise RuntimeError(
            "Failed to read scheduler.kv_cache_manager.block_pool"
        ) from exc


def _logical_cache_tokens_from_block_pool(block_pool, block_size: int) -> int:
    """Logical prefix-cache tokens registered in a vLLM BlockPool.

    BlockPool keys include the KV-cache group id. Hybrid models can register
    the same logical prefix block for multiple groups, so count unique
    BlockHash values rather than raw map entries or transient block occupancy.
    """
    try:
        token_granularity = int(block_pool.hash_block_size)
        cache_map = block_pool.cached_block_hash_to_block
        raw_cache = cache_map._cache

        from vllm.v1.core.kv_cache_utils import get_block_hash  # type: ignore

        logical_hashes = {get_block_hash(key) for key in raw_cache}
        return len(logical_hashes) * token_granularity
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Failed to read logical cached tokens from vLLM BlockPool"
        ) from exc


def _hbm_logical_cache_tokens(scheduler, block_size: int) -> int:
    return _logical_cache_tokens_from_block_pool(_block_pool(scheduler), block_size)


def _dram_block_pool(scheduler):
    if scheduler.connector is None:
        return None
    try:
        return scheduler.connector.scheduler_manager.cpu_block_pool
    except AttributeError as exc:
        raise RuntimeError(
            "Failed to read scheduler.connector.scheduler_manager.cpu_block_pool"
        ) from exc


def _dram_logical_cache_tokens(scheduler, block_size: int) -> int:
    block_pool = _dram_block_pool(scheduler)
    if block_pool is None:
        return 0
    return _logical_cache_tokens_from_block_pool(block_pool, block_size)


def _hbm_logical_capacity_tokens(scheduler) -> int:
    """Logical GPU KV cache capacity in tokens.

    For hybrid models, vLLM's BlockPool contains blocks for every KV cache
    group. One logical prefix block may therefore consume multiple physical
    pool blocks. Match vLLM's own "GPU KV cache size" accounting instead of
    reporting raw BlockPool capacity.
    """
    try:
        cfg = scheduler.kv_cache_manager.kv_cache_config
        groups = cfg.kv_cache_groups
        if not groups:
            raise RuntimeError("vLLM kv_cache_config.kv_cache_groups is empty")
        min_block_size = min(
            int(group.kv_cache_spec.block_size) for group in groups
        )
        num_tokens = int(cfg.num_blocks) // len(groups) * min_block_size

        parallel_config = scheduler.vllm_config.parallel_config
        cp_size = (
            int(parallel_config.prefill_context_parallel_size)
            * int(parallel_config.decode_context_parallel_size)
        )
        if cp_size > 1:
            num_tokens *= cp_size
        return num_tokens
    except AttributeError as exc:
        raise RuntimeError("Failed to read logical HBM KV cache capacity") from exc


def _resolved_block_size(scheduler) -> int:
    """Read the block size actually used by vLLM."""
    try:
        cfg = scheduler.kv_cache_manager.kv_cache_config
        block_sizes = [
            int(g.kv_cache_spec.block_size) for g in cfg.kv_cache_groups
        ]
        if not block_sizes:
            raise RuntimeError("vLLM kv_cache_config.kv_cache_groups is empty")
        return min(block_sizes)
    except AttributeError as exc:
        raise RuntimeError("Failed to read resolved vLLM block size") from exc


def _kv_bytes_per_token_from_engine(llm) -> int:
    """Read per-token KV footprint from the engine config.

    Sums bytes-per-block across all KV cache groups (attention + mamba) as
    declared in `kv_cache_config.kv_cache_groups[i].kv_cache_spec`, then
    divides by the aligned block size. Used only to turn the
    ``external_kv_transfer`` token counter into a byte estimate.
    """
    try:
        ec = llm.llm_engine.engine_core.engine_core  # InprocClient → EngineCore
        kvcm = ec.scheduler.kv_cache_manager
        cfg = kvcm.kv_cache_config
        # All groups in a hybrid model share the same aligned block size; grab
        # it from the first spec.
        first_spec = cfg.kv_cache_groups[0].kv_cache_spec
        block_size = int(first_spec.block_size)
        if block_size <= 0:
            raise RuntimeError(f"Invalid vLLM KV cache block_size={block_size}")
        total_bytes_per_block = 0
        for g in cfg.kv_cache_groups:
            total_bytes_per_block += int(g.kv_cache_spec.page_size_bytes)
        if total_bytes_per_block <= 0:
            raise RuntimeError(
                f"Invalid total KV bytes per block={total_bytes_per_block}"
            )
        return total_bytes_per_block // block_size
    except (AttributeError, IndexError) as exc:
        raise RuntimeError("Failed to read per-token KV footprint from vLLM") from exc


def _available_kv_cache_memory_bytes_from_engine(llm) -> int:
    """Read vLLM's profiled KV cache memory budget in bytes.

    This is the same value reported by vLLM as "Available KV cache memory".
    It is a memory budget from profiling or kv_cache_memory_bytes, not the
    allocatable logical token capacity after block/layer grouping.
    """
    try:
        ec = llm.llm_engine.engine_core.engine_core  # InprocClient -> EngineCore
        available_bytes = int(ec.available_gpu_memory_for_kv_cache)
        if available_bytes < 0:
            raise RuntimeError(
                f"Invalid available KV cache memory={available_bytes}"
            )
        return available_bytes
    except AttributeError as exc:
        raise RuntimeError(
            "Failed to read available KV cache memory from vLLM"
        ) from exc


# ── Main driver ─────────────────────────────────────────────────────────────


def run_live(
    requests: list[TokenizedRequest],
    *,
    model: str,
    block_size: int,
    hbm_strategy: str,
    dram_strategy: str,
    hbm_capacity_gb: float | None = None,
    cpu_capacity_gb: float = 8.0,
    dtype: str = "auto",
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = True,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
    max_gen_tokens: int = 1,
    nvml_sample: bool = False,
    nvml_interval_s: float = 0.05,
    extra_llm_kwargs: dict | None = None,
    progress: bool = True,
) -> RunState:
    """Run a list of tokenized requests through a live vLLM engine, serially.

    Serial submission mirrors the simulator's loop: the request order
    (timestamp / random / min_distance / ...) set upstream is the actual
    cache-interaction order on the engine. Batching would let vLLM reorder
    inside the scheduler, breaking the comparison with the simulator.
    """
    LLM, SamplingParams = _lazy_import_vllm()

    llm_kwargs = _build_llm_kwargs(
        model=model,
        block_size=block_size,
        hbm_strategy=hbm_strategy,
        dram_strategy=dram_strategy,
        hbm_capacity_gb=hbm_capacity_gb,
        cpu_capacity_gb=cpu_capacity_gb,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
    )
    if extra_llm_kwargs:
        llm_kwargs.update(extra_llm_kwargs)

    llm = LLM(**llm_kwargs)
    sp = SamplingParams(max_tokens=max_gen_tokens, temperature=0.0)

    scheduler = _scheduler(llm)
    block_size = _resolved_block_size(scheduler)
    bpt_engine = _kv_bytes_per_token_from_engine(llm)
    hbm_capacity_tokens = _hbm_logical_capacity_tokens(scheduler)
    hbm_capacity_bytes = _available_kv_cache_memory_bytes_from_engine(llm)

    state = RunState(
        block_size=block_size,
        hbm_capacity_tokens=hbm_capacity_tokens,
        hbm_capacity_bytes=hbm_capacity_bytes,
        pcie_kv_bytes_per_token=bpt_engine,
        engine_hbm_strategy=hbm_strategy,
        engine_dram_strategy=dram_strategy,
    )

    sampler: PcieSampler | None = None
    if nvml_sample:
        sampler = PcieSampler(interval_s=nvml_interval_s)
        sampler.start()
        state.nvml_available = sampler.available
        if sampler.available:
            print(f"[nvml] sampling on {sampler.detected_label}", flush=True)
        else:
            print(f"[nvml] disabled: {sampler.init_error}", flush=True)

    iterator = requests
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(
                requests,
                desc=f"vLLM[hbm={hbm_strategy}|dram={dram_strategy}]",
                unit="req",
            )
        except ImportError:
            pass

    t_infer_start = time.perf_counter()
    try:
        for req in iterator:
            # New-API: LLM.generate takes `prompts` (PromptType). For token-id
            # input use a TokensPrompt dict.
            outputs = llm.generate(
                [{"prompt_token_ids": list(req.token_ids)}],
                sp,
                use_tqdm=False,
            )
            out = outputs[0]
            input_tokens = len(req.token_ids)

            # RequestOutput carries the first-schedule prefix-cache snapshot.
            # Scheduler prefix_cache_stats are consumed/reset by vLLM's logger,
            # so reading them here races with periodic engine logging.
            if out.num_cached_tokens is None:
                raise RuntimeError("RequestOutput.num_cached_tokens is None")
            if out.num_external_computed_tokens is None:
                raise RuntimeError(
                    "RequestOutput.num_external_computed_tokens is None"
                )
            cached_total = int(out.num_cached_tokens)
            d_connector = int(out.num_external_computed_tokens)
            cached_total = min(input_tokens, cached_total)
            d_connector = min(cached_total, d_connector)
            d_native = cached_total - d_connector
            miss = input_tokens - (d_native + d_connector)
            state.traces.append(
                PerRequestTrace(
                    input_tokens=input_tokens,
                    hit_tokens=d_native + d_connector,
                    miss_tokens=miss,
                    hbm_hit_tokens=d_native,
                    dram_hit_tokens=d_connector,
                )
            )
            state.pcie_external_tokens += d_connector
            state.hbm_cache_token_samples.append(
                _hbm_logical_cache_tokens(scheduler, block_size)
            )
            state.dram_cache_token_samples.append(
                _dram_logical_cache_tokens(scheduler, block_size)
            )
    finally:
        state.inference_time_s = time.perf_counter() - t_infer_start
        if sampler is not None:
            sampler.stop()
            s = sampler.snapshot()
            state.nvml_pcie_tx_bytes = int(s.tx_bytes)
            state.nvml_pcie_rx_bytes = int(s.rx_bytes)
            state.nvml_sampling_duration_s = float(s.sampling_duration_s)
            state.nvml_pcie_peak_bytes_per_s = float(s.peak_total_bytes_per_s)

    return state
