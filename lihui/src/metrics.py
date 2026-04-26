"""Aggregate real-vLLM per-request traces into RunMetrics for CSV output.

Simulator-only columns (FLOP counts, wall-clock breakdowns, branch stats,
per-request saved-time percentiles) are dropped. Additional columns for
the DRAM-tier picture are added:
  * external_kv_transfer_tokens        — tokens restored by the connector
  * pcie_bytes_restore_estimate        — external tokens × bytes_per_token
  * pcie_nvml_bytes_tx/rx/total        — NVML hardware PCIe throughput
                                          integrated over the run (if enabled)
  * pcie_avg_bandwidth_gb_per_s        — average PCIe bandwidth over the
                                          sampled/inference window
  * pcie_peak_bandwidth_gb_per_s       — peak NVML sampled total bandwidth
  * pcie_total_transfer_bytes          — best-available canonical value:
                                          NVML total if sampled, else estimate
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.vllm_runner import RunState


@dataclass
class RunMetrics:
    # ── Tier hit rates ──────────────────────────────────────────────────────
    hbm_token_hit_rate: float = 0.0
    dram_token_hit_rate: float = 0.0

    # ── Run size ────────────────────────────────────────────────────────────
    num_requests: int = 0
    total_input_tokens: int = 0
    inference_time_s: float = 0.0

    # ── Tokens by source ────────────────────────────────────────────────────
    load_tokens: int = 0
    compute_tokens: int = 0
    load_compute_ratio: float | None = None
    external_kv_transfer_tokens: int = 0

    # ── Prefix-cache residency and HBM KV capacity ─────────────────────────
    peak_cached_tokens: int = 0
    avg_cached_tokens: float = 0.0
    # Token capacity is logical; byte capacity is vLLM's profiled KV memory
    # budget, matching the "Available KV cache memory" log line.
    hbm_capacity_tokens: int = 0
    hbm_capacity_bytes: int = 0

    # DRAM-tier logical cache residency and token flow.
    dram_peak_cached_tokens: int = 0
    dram_avg_cached_tokens: float = 0.0
    avg_promoted_tokens_per_req: float = 0.0
    avg_restore_tokens_per_req: float = 0.0

    # ── PCIe transfer bytes (three independent signals) ─────────────────────
    # (1) Token-based estimate — only the DRAM→HBM restore half, exact in
    #     tokens but relies on a derived bytes-per-token.
    pcie_bytes_restore_estimate: int = 0
    pcie_kv_bytes_per_token: int = 0
    # (2) NVML PCIe throughput integration — bi-directional, bytes, but
    #     includes weight loads and other non-KV traffic.
    pcie_nvml_bytes_tx: int = 0
    pcie_nvml_bytes_rx: int = 0
    pcie_nvml_bytes_total: int = 0
    pcie_avg_bandwidth_gb_per_s: float = 0.0
    pcie_peak_bandwidth_gb_per_s: float = 0.0
    nvml_available: bool = False
    # (3) Canonical — NVML total if sampled, else the restore-only estimate.
    pcie_total_transfer_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_run_metrics(state: RunState) -> RunMetrics:
    traces = state.traces
    assert traces, "Expected per-request traces to compute metrics, but got none"

    n_req = len(traces)
    total_in = sum(t.input_tokens for t in traces)
    load_tokens = sum(t.hit_tokens for t in traces)
    compute_tokens = sum(t.miss_tokens for t in traces)
    hbm_hit = sum(t.hbm_hit_tokens for t in traces)
    dram_hit = sum(t.dram_hit_tokens for t in traces)
    restore_tokens = dram_hit
    if int(state.pcie_external_tokens) != restore_tokens:
        raise RuntimeError(
            "Inconsistent external restore token accounting: "
            f"state.pcie_external_tokens={state.pcie_external_tokens}, "
            f"sum(trace.dram_hit_tokens)={restore_tokens}"
        )

    lcr: float | None
    lcr = None if compute_tokens == 0 else float(load_tokens / compute_tokens)

    hbm_samples = state.hbm_cache_token_samples
    hbm_peak = max(hbm_samples) if hbm_samples else 0
    hbm_avg = sum(hbm_samples) / len(hbm_samples) if hbm_samples else 0.0

    dram_samples = state.dram_cache_token_samples
    dram_peak = max(dram_samples) if dram_samples else 0
    dram_avg = sum(dram_samples) / len(dram_samples) if dram_samples else 0.0

    bpt = int(state.pcie_kv_bytes_per_token)
    restore_est = restore_tokens * bpt
    nvml_tx = int(state.nvml_pcie_tx_bytes)
    nvml_rx = int(state.nvml_pcie_rx_bytes)
    nvml_total = nvml_tx + nvml_rx

    canonical = (
        nvml_total
        if state.nvml_available and nvml_total > 0
        else restore_est
    )

    bandwidth_window_s = (
        float(state.nvml_sampling_duration_s)
        if state.nvml_available and state.nvml_sampling_duration_s > 0
        else float(state.inference_time_s)
    )
    avg_bandwidth_gb_per_s = (
        (float(canonical) / bandwidth_window_s) / 1e9
        if bandwidth_window_s > 0
        else 0.0
    )
    peak_bandwidth_gb_per_s = float(state.nvml_pcie_peak_bytes_per_s) / 1e9

    return RunMetrics(
        hbm_token_hit_rate=(hbm_hit / total_in) if total_in else 0.0,
        dram_token_hit_rate=(dram_hit / total_in) if total_in else 0.0,
        num_requests=n_req,
        total_input_tokens=total_in,
        inference_time_s=float(state.inference_time_s),
        load_tokens=load_tokens,
        compute_tokens=compute_tokens,
        load_compute_ratio=lcr,
        external_kv_transfer_tokens=restore_tokens,
        peak_cached_tokens=hbm_peak,
        avg_cached_tokens=hbm_avg,
        hbm_capacity_tokens=state.hbm_capacity_tokens,
        hbm_capacity_bytes=state.hbm_capacity_bytes,
        dram_peak_cached_tokens=dram_peak,
        dram_avg_cached_tokens=dram_avg,
        avg_promoted_tokens_per_req=0.0,
        avg_restore_tokens_per_req=float(restore_tokens) / n_req,
        pcie_bytes_restore_estimate=restore_est,
        pcie_kv_bytes_per_token=bpt,
        pcie_nvml_bytes_tx=nvml_tx,
        pcie_nvml_bytes_rx=nvml_rx,
        pcie_nvml_bytes_total=nvml_total,
        pcie_avg_bandwidth_gb_per_s=avg_bandwidth_gb_per_s,
        pcie_peak_bandwidth_gb_per_s=peak_bandwidth_gb_per_s,
        nvml_available=bool(state.nvml_available),
        pcie_total_transfer_bytes=int(canonical),
    )
