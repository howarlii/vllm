"""NVML PCIe throughput sampler — best-effort hardware ground-truth.

NVML exposes PCIe byte-rate counters (`nvmlDeviceGetPcieThroughput`) that
report recent throughput in KB/s for the TX and RX paths independently.
The counters are not cumulative byte totals — they're a sliding window
(~20ms) rate. To approximate total bytes transferred during a window of
wall-clock time, we sample at a fixed cadence in a background thread and
sum ``rate * dt`` over the run.

This is system-wide PCIe traffic, not KV-specific — weight loads, kernel
launches and unrelated traffic count. Treat it as an upper bound / ground
truth, not a pure KV metric.

The sampler auto-detects which physical GPU the current process is bound
to (through CUDA_VISIBLE_DEVICES remapping) by reading torch's reported
PCI bus ID and looking that up in NVML. No manual device index needed.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional


# pynvml constants (cf. vllm/third_party/pynvml.py)
_NVML_PCIE_UTIL_TX_BYTES = 0
_NVML_PCIE_UTIL_RX_BYTES = 1
# _NVML_PCIE_UTIL_COUNT = 2  # sum of both — NVML returns KB/s


def _auto_detect_handle(nv):
    """Return an NVML device handle for the GPU this process is actually using.

    CUDA_VISIBLE_DEVICES remaps CUDA device ids (the child process sees its
    bound GPU as device 0) but NVML is not affected — it still enumerates
    all physical GPUs. We match the two by PCI bus ID, which is stable
    regardless of CUDA/NVML enumeration order.

    Fallbacks, in order:
      1. torch's pci_{domain,bus,device}_id on CUDA device 0 → nvmlDeviceGetHandleByPciBusId
      2. parse CUDA_VISIBLE_DEVICES and use the first id as NVML index
         (correct only when CUDA_DEVICE_ORDER=PCI_BUS_ID — noted in a comment)
      3. NVML device 0

    Returns ``(handle, detected_label)`` where detected_label is a short
    human-readable string for logs.
    """
    # 1. Primary: torch PCI bus id → NVML handle by PCI id.
    try:
        import torch  # type: ignore
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            prop = torch.cuda.get_device_properties(0)
            dom = int(getattr(prop, "pci_domain_id", 0) or 0)
            bus = int(getattr(prop, "pci_bus_id", 0) or 0)
            dev = int(getattr(prop, "pci_device_id", 0) or 0)
            bus_id = f"{dom:04X}:{bus:02X}:{dev:02X}.0"
            try:
                h = nv.nvmlDeviceGetHandleByPciBusId(bus_id.encode("ascii"))
                return h, f"pci={bus_id} ({getattr(prop, 'name', '?')})"
            except Exception:
                pass
    except Exception:
        pass
    # 2. Fallback: parse CUDA_VISIBLE_DEVICES first entry.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        first = cvd.split(",")[0].strip()
        if first.isdigit():
            try:
                h = nv.nvmlDeviceGetHandleByIndex(int(first))
                return h, f"cvd={first} (assumes CUDA_DEVICE_ORDER=PCI_BUS_ID)"
            except Exception:
                pass
    # 3. Last resort.
    return nv.nvmlDeviceGetHandleByIndex(0), "fallback index=0"


@dataclass
class PcieSample:
    tx_bytes: int = 0
    rx_bytes: int = 0
    samples: int = 0
    sampling_duration_s: float = 0.0
    sampling_dt_mean: float = 0.0
    peak_total_bytes_per_s: float = 0.0

    @property
    def total_bytes(self) -> int:
        return self.tx_bytes + self.rx_bytes


class PcieSampler:
    """Background-thread PCIe throughput sampler for the active GPU.

    No explicit device index needed — the sampler auto-detects which
    physical GPU this process is currently using (see _auto_detect_handle).

    Usage:
        s = PcieSampler(interval_s=0.05)
        s.start()
        ... run vLLM ...
        s.stop()
        print(s.snapshot(), s.detected_label)
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._sample = PcieSample()
        self._pynvml = None
        self._handle = None
        self._available = False
        self._init_error: Optional[str] = None
        self._tick_total = 0.0
        self.detected_label: str = ""

    def _try_init(self) -> bool:
        try:
            # Prefer vLLM's bundled pynvml to avoid adding deps.
            from vllm.third_party import pynvml as nv  # type: ignore
        except Exception:
            try:
                import pynvml as nv  # type: ignore
            except Exception as e:
                self._init_error = f"pynvml not importable: {e!r}"
                return False
        try:
            nv.nvmlInit()
            handle, label = _auto_detect_handle(nv)
            # Probe once — some drivers / virtualisation environments don't
            # implement PCIe throughput.
            nv.nvmlDeviceGetPcieThroughput(handle, _NVML_PCIE_UTIL_TX_BYTES)
            self._pynvml = nv
            self._handle = handle
            self._available = True
            self.detected_label = label
            return True
        except Exception as e:
            self._init_error = f"nvml init failed: {e!r}"
            return False

    def start(self) -> None:
        if not self._try_init():
            return
        self._stop.clear()
        self._th = threading.Thread(
            target=self._run, name=f"pcie-sampler:{self.detected_label}", daemon=True
        )
        self._th.start()

    def stop(self) -> None:
        if self._th is None:
            return
        self._stop.set()
        self._th.join(timeout=self.interval_s * 10)
        self._th = None
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def _run(self) -> None:
        assert self._pynvml is not None and self._handle is not None
        nv = self._pynvml
        handle = self._handle
        last = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(self.interval_s)
            now = time.perf_counter()
            dt = now - last
            last = now
            try:
                # NVML returns KB/s; convert to bytes/s with *1024.
                tx_kBps = int(nv.nvmlDeviceGetPcieThroughput(handle, _NVML_PCIE_UTIL_TX_BYTES))
                rx_kBps = int(nv.nvmlDeviceGetPcieThroughput(handle, _NVML_PCIE_UTIL_RX_BYTES))
            except Exception:
                continue
            tx_b = int(tx_kBps * 1024 * dt)
            rx_b = int(rx_kBps * 1024 * dt)
            total_bytes_per_s = float((tx_kBps + rx_kBps) * 1024)
            with self._lock:
                self._sample.tx_bytes += tx_b
                self._sample.rx_bytes += rx_b
                self._sample.samples += 1
                self._sample.peak_total_bytes_per_s = max(
                    self._sample.peak_total_bytes_per_s,
                    total_bytes_per_s,
                )
                self._tick_total += dt

    def snapshot(self) -> PcieSample:
        with self._lock:
            s = PcieSample(
                tx_bytes=self._sample.tx_bytes,
                rx_bytes=self._sample.rx_bytes,
                samples=self._sample.samples,
                sampling_duration_s=self._tick_total,
                sampling_dt_mean=(
                    self._tick_total / self._sample.samples
                    if self._sample.samples
                    else 0.0
                ),
                peak_total_bytes_per_s=self._sample.peak_total_bytes_per_s,
            )
        return s

    @property
    def available(self) -> bool:
        return self._available

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error
