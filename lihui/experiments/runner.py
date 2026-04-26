"""Persist-to-CSV helpers and the request-prep bridge for live vLLM runs.

This is the real-run counterpart to the simulator's ``experiments/runner.py``.
The CSV schema is trimmed: the simulator's FLOP counts, branch statistics,
and per-request saved-time distribution are removed because those were
artefacts of the simulator's own hardware model. We keep the columns that
are directly measurable from a live vLLM engine.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

from src.config import DEFAULT_TOKENIZER_NAME, PREPARED_CACHE_DIR, ensure_hf_cache_dirs
from src.datasets_loader import load_raw_requests
from src.request_generator import (
    OrderingName,
    TokenizedRequest,
    load_or_tokenize,
    order_requests,
)


# ── CSV schema ───────────────────────────────────────────────────────────────

# Note: vLLM's native HBM eviction is hard-coded LRU and not user-selectable,
# so we omit the simulator's 'strategy' column. The 'dram_strategy' column now
# carries the primary sweep axis (vllm-native-none/all/align, or lmcache).

RESULT_CSV_FIELDS: List[str] = [
    # ── Dataset / request-prep args ─────────────────────────────────────
    "dataset",
    "ordering",
    "sessions_per_second",
    "words_per_min",
    "tokenizer",
    "seed",
    "tokenize_workers",
    "max_requests",
    "requested_page_size",
    "page_size",
    # ── vLLM engine args ────────────────────────────────────────────────
    "model_name",
    "hbm_strategy",
    "dram_strategy",
    "hbm_capacity_spec",
    "cpu_capacity_spec",
    "dtype",
    "max_model_len",
    "gpu_memory_utilization",
    "enforce_eager",
    "tensor_parallel_size",
    "max_gen_tokens",
    # ── Measurement args ────────────────────────────────────────────────
    "nvml_sample",
    "nvml_interval_s",
    # ── Engine-side capacity snapshot ───────────────────────────────────
    "hbm_capacity_tokens",
    "hbm_capacity_bytes",
    # ── Run size ────────────────────────────────────────────────────────
    "num_requests",
    "total_input_tokens",
    "inference_time_s",
    # ── Tier hit rates ──────────────────────────────────────────────────
    "hbm_token_hit_rate",
    "dram_token_hit_rate",
    # ── Token / capacity summary ────────────────────────────────────────
    "load_tokens",
    "compute_tokens",
    "load_compute_ratio",
    "external_kv_transfer_tokens",
    "peak_cached_tokens",
    "avg_cached_tokens",
    "dram_peak_cached_tokens",
    "dram_avg_cached_tokens",
    "avg_promoted_tokens_per_req",
    "avg_restore_tokens_per_req",
    # ── PCIe transfer bytes ─────────────────────────────────────────────
    "pcie_kv_bytes_per_token",
    "pcie_bytes_restore_estimate",
    "pcie_nvml_bytes_tx",
    "pcie_nvml_bytes_rx",
    "pcie_nvml_bytes_total",
    "pcie_avg_bandwidth_gb_per_s",
    "pcie_peak_bandwidth_gb_per_s",
    "nvml_available",
    "pcie_total_transfer_bytes",
]


def persist_result_row(
    out_csv: Path,
    out_json_dir: Path,
    row: Dict[str, Any],
    *,
    overwrite_existing: bool = False,
) -> Literal["inserted", "overwritten", "skipped"]:
    """Merge-upsert a result row into CSV + write a per-run JSON sidecar."""
    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    hbm_strategy = row.get("hbm_strategy") or ""
    dram_strategy = row.get("dram_strategy") or ""
    slug_parts = [
        f"{row.get('dataset')}",
        f"ps{row.get('page_size')}",
        f"{row.get('ordering')}",
        f"hbm-{hbm_strategy}",
        f"dram-{dram_strategy}",
        f"cap{row.get('hbm_capacity_spec')}",
    ]
    slug = "_".join(slug_parts)
    jpath = out_json_dir / f"{slug}.json"

    metrics = row.get("metrics") or {}
    flat: Dict[str, Any] = {field: row.get(field, "") for field in RESULT_CSV_FIELDS}
    for field in RESULT_CSV_FIELDS:
        if flat.get(field, "") != "":
            continue
        flat[field] = metrics.get(field)

    KEY_FIELDS = (
        "dataset",
        "ordering",
        "sessions_per_second",
        "words_per_min",
        "tokenizer",
        "seed",
        "tokenize_workers",
        "max_requests",
        "requested_page_size",
        "hbm_strategy",
        "dram_strategy",
        "hbm_capacity_spec",
        "cpu_capacity_spec",
        "model_name",
        "dtype",
        "max_model_len",
        "gpu_memory_utilization",
        "enforce_eager",
        "tensor_parallel_size",
        "max_gen_tokens",
        "nvml_sample",
        "nvml_interval_s",
    )

    new_row = {k: flat.get(k, "") for k in RESULT_CSV_FIELDS}
    key = tuple(str(new_row.get(k, "")) for k in KEY_FIELDS)

    if out_csv.is_file():
        with out_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = list(reader.fieldnames or [])
            existing = list(reader)
        merged_fields = list(RESULT_CSV_FIELDS)
        for f in old_fields:
            if f not in merged_fields:
                merged_fields.append(f)
        replaced = False
        for i, r in enumerate(existing):
            if tuple(str(r.get(k, "")) for k in KEY_FIELDS) == key:
                if not overwrite_existing:
                    return "skipped"
                existing[i] = new_row
                replaced = True
                break
        if not replaced:
            existing.append(new_row)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=merged_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows({k: r.get(k, "") for k in merged_fields} for r in existing)
        jpath.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        return "overwritten" if replaced else "inserted"
    else:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerow(new_row)
        jpath.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        return "inserted"


# ── Request preparation (identical to simulator's prepare_requests) ─────────

def _prepared_cache_path(
    dataset: str,
    ordering: str,
    tokenizer_name: str,
    *,
    seed: int,
    max_requests: Optional[int],
    sessions_per_second: float,
    words_per_min: float,
    narrativeqa_docs: int,
    sharegpt_conversations: int,
) -> Path:
    """Path of the post-tokenize, post-order pickle cache for one prep call."""
    safe_tok = tokenizer_name.replace("/", "_")
    n_label = "all" if max_requests is None else str(int(max_requests))
    name = (
        f"{dataset}__{ordering}__{safe_tok}"
        f"__seed{seed}__n{n_label}"
        f"__sps{sessions_per_second}__wpm{words_per_min}"
        f"__nqa{narrativeqa_docs}__sgc{sharegpt_conversations}"
        f".pkl"
    )
    return PREPARED_CACHE_DIR / name


def prepare_requests(
    dataset: str,
    ordering: str,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    *,
    narrativeqa_docs: int = 50,
    sharegpt_conversations: int = 10_000,
    seed: int = 0,
    tokenize_workers: int = 0,
    force_retokenize: bool = False,
    max_requests: Optional[int] = None,
    sessions_per_second: float = 1.0,
    words_per_min: float = 90.0,
) -> List[TokenizedRequest]:
    """Load (or re-tokenize) a dataset + apply an ordering, same as simulator.

    The pickle cache is shared with the simulator's PREPARED_CACHE_DIR via
    symlink, so prepared request lists generated for simulator runs are
    immediately reusable here.
    """
    cache_path = _prepared_cache_path(
        dataset, ordering, tokenizer_name,
        seed=seed,
        max_requests=max_requests,
        sessions_per_second=sessions_per_second,
        words_per_min=words_per_min,
        narrativeqa_docs=narrativeqa_docs,
        sharegpt_conversations=sharegpt_conversations,
    )
    if not force_retokenize and cache_path.is_file():
        try:
            with cache_path.open("rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    # Superset fast path: if no exact-n pickle exists, reuse a larger pickle
    # with identical (dataset, ordering, tokenizer, seed, sps, wpm, nqa, sgc)
    # and slice its first ``max_requests`` entries. This mirrors the
    # simulator's load_or_tokenize superset logic and lets small probe runs
    # (e.g. --max-requests 2) work without re-downloading the dataset.
    if not force_retokenize and max_requests is not None:
        safe_tok = tokenizer_name.replace("/", "_")
        prefix = (
            f"{dataset}__{ordering}__{safe_tok}"
            f"__seed{seed}__n"
        )
        suffix = (
            f"__sps{sessions_per_second}__wpm{words_per_min}"
            f"__nqa{narrativeqa_docs}__sgc{sharegpt_conversations}"
            f".pkl"
        )
        best: Optional[tuple] = None   # (n, path)
        for p in PREPARED_CACHE_DIR.glob(f"{prefix}*{suffix}"):
            stem = p.name[len(prefix):-len(suffix)]
            if not stem.isdigit():
                continue
            n_cached = int(stem)
            if n_cached >= int(max_requests) and (best is None or n_cached < best[0]):
                best = (n_cached, p)
        if best is not None:
            try:
                with best[1].open("rb") as f:
                    full = pickle.load(f)
                return list(full[: int(max_requests)])
            except Exception:
                pass

    ensure_hf_cache_dirs()
    raw = load_raw_requests(
        dataset,
        narrativeqa_docs=narrativeqa_docs,
        sharegpt_conversations=sharegpt_conversations,
        seed=seed,
        max_requests=max_requests,
    )
    if not raw:
        return []
    if max_requests is not None:
        raw = raw[:max_requests]

    order_kw = dict(
        mode=cast(OrderingName, ordering),
        seed=seed,
        sessions_per_second=sessions_per_second,
        words_per_min=words_per_min,
    )

    tok = load_or_tokenize(
        dataset,
        raw,
        tokenizer_name=tokenizer_name,
        num_workers=tokenize_workers,
        force_recompute=force_retokenize,
    )
    result = order_requests(tok, **order_kw)

    try:
        PREPARED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
    except Exception:
        pass
    return result
