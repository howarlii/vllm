#!/usr/bin/env bash
# lihui/sweep.sh — run many run_one.py experiments, auto-dispatched to GPUs.
#
# How it works
# ------------
# 1. `EXPERIMENTS` below is an array of whole CLI-arg strings, each passed
#    verbatim to run_one.py.
# 2. `GPUS` is the pool of GPU ids to use. The script keeps exactly one
#    concurrent experiment per GPU — as soon as a GPU frees up the next
#    queued experiment is dispatched on it. (Semaphore via a FIFO.)
# 3. Per-experiment stdout/stderr goes to results/logs/<slug>.gpu<N>.log
#    so the console stays readable.
#
# Usage
# -----
#   bash lihui/sweep.sh                        # use GPUs defined below
#   SWEEP_GPUS="0 1"     bash lihui/sweep.sh    # restrict to GPUs 0 and 1
#   SWEEP_DRY_RUN=1      bash lihui/sweep.sh    # print the commands, don't run
#   SWEEP_WRITE_CSV=0    bash lihui/sweep.sh    # pass --no-write-csv
#
# Notes
# -----
# - The machine has 4 mixed GPUs (2×RTX 3090 + 2×RTX 2080 Ti).  Bamba-9B does
#   NOT fit on an 11 GB 2080 Ti — set `GPUS=(0 1)` (the two 3090s) for
#   hybrid-Mamba experiments.  CUDA_DEVICE_ORDER=PCI_BUS_ID is forced below
#   so GPU indices match what nvidia-smi prints.
# - On 24 GB 3090, Bamba-9B leaves only ~2 GB KV cache space after weights —
#   run_one.py defaults to --max-model-len=8192 for this reason. Override if
#   your dataset has longer prompts AND you have a bigger card.
set -u

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          USER CONFIG                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# GPU pool — override with SWEEP_GPUS="0 2" on the command line.
GPUS=( ${SWEEP_GPUS:-0 1} )

# Python interpreter in the vLLM venv.
VENV_PY="${SWEEP_VENV_PY:-/data3/howarli/dev/vllm/.venv/bin/python}"
WRITE_CSV="${SWEEP_WRITE_CSV:-1}"
OVERWRITE_CSV="${SWEEP_OVERWRITE_CSV:-0}"

# Each entry is forwarded to run_one.py verbatim. Edit freely.
EXPERIMENTS=(
  # --- baseline sweep: 3 HBM modes × 2 DRAM modes on swesmith -------------
  "--dataset swesmith --ordering timestamp --hbm-strategy none  --dram-strategy none        --max-requests 500 --nvml-sample"
  "--dataset swesmith --ordering timestamp --hbm-strategy all   --dram-strategy none        --max-requests 500 --nvml-sample"
  "--dataset swesmith --ordering timestamp --hbm-strategy align --dram-strategy none        --max-requests 500 --nvml-sample"
  "--dataset swesmith --ordering timestamp --hbm-strategy all   --dram-strategy vllm-native --cpu-capacity 4.0 --max-requests 500 --nvml-sample"
  "--dataset swesmith --ordering timestamp --hbm-strategy align --dram-strategy vllm-native --cpu-capacity 4.0 --max-requests 500 --nvml-sample"

  "--dataset oasst1 --ordering timestamp --hbm-strategy none  --dram-strategy none        --max-requests 500 --nvml-sample"
  "--dataset oasst1 --ordering timestamp --hbm-strategy all   --dram-strategy none        --max-requests 500 --nvml-sample"
  "--dataset oasst1 --ordering timestamp --hbm-strategy align --dram-strategy none        --max-requests 500 --nvml-sample"
  "--dataset oasst1 --ordering timestamp --hbm-strategy all   --dram-strategy vllm-native --cpu-capacity 4.0 --max-requests 500 --nvml-sample"
  "--dataset oasst1 --ordering timestamp --hbm-strategy align --dram-strategy vllm-native --cpu-capacity 4.0 --max-requests 500 --nvml-sample"

  # --- example: dataset × strategy matrix --------------------------------
  # "--dataset loogle       --hbm-strategy all   --dram-strategy none        --max-requests 500 --nvml-sample"
  # "--dataset loogle       --hbm-strategy all   --dram-strategy vllm-native --cpu-capacity 8.0 --max-requests 500 --nvml-sample"
  # "--dataset narrativeqa  --hbm-strategy all   --dram-strategy vllm-native --cpu-capacity 8.0 --max-requests 500 --nvml-sample"
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       END OF CONFIG                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="$SCRIPT_DIR/experiments/run_one.py"
LOG_DIR="$SCRIPT_DIR/results/logs"
mkdir -p "$LOG_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# vLLM spawns workers; spawn is safer than fork inside a bash-managed pool.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

n=${#EXPERIMENTS[@]}
if (( n == 0 )); then
  echo "No experiments defined in EXPERIMENTS — edit $0." >&2
  exit 2
fi
if (( ${#GPUS[@]} == 0 )); then
  echo "GPUS is empty — set SWEEP_GPUS or edit $0." >&2
  exit 2
fi

slugify() {
  # "--dataset swesmith  --hbm-strategy all --max-requests 500"
  # → "dataset-swesmith_hbm-strategy-all_max-requests-500"
  local s="$1"
  # normalise whitespace runs to single space
  s="$(printf '%s' "$s" | tr -s ' ')"
  s="${s// --/_}"
  s="${s//--/}"
  s="${s// /-}"
  # note: '_-' puts underscore first so tr doesn't parse it as an option flag
  printf '%s' "$s" | tr -c 'A-Za-z0-9._-' '_' | tr -s '_-' | cut -c1-180
}
ts() { date +%H:%M:%S; }

# ── Dry-run: print commands and exit. ───────────────────────────────────────
if [[ "${SWEEP_DRY_RUN:-0}" = "1" ]]; then
  echo "# dry-run: $n experiment(s) would dispatch onto GPUs [${GPUS[*]}]"
  for i in "${!EXPERIMENTS[@]}"; do
    args="${EXPERIMENTS[$i]}"
    if [[ "$WRITE_CSV" != "1" ]]; then args="$args --no-write-csv"; fi
    if [[ "$OVERWRITE_CSV" = "1" ]]; then args="$args --overwrite-csv"; fi
    slug="$(slugify "$args")"
    printf '[%02d] slug=%s\n     CUDA_VISIBLE_DEVICES=<free-gpu> %s %s %s\n' \
      "$((i+1))" "$slug" "$VENV_PY" "$RUN_ONE" "$args"
  done
  exit 0
fi

# ── Semaphore: FIFO with one token per free GPU. ────────────────────────────
FIFO="$(mktemp -u)"
mkfifo "$FIFO"
exec 3<>"$FIFO"
rm "$FIFO"
for g in "${GPUS[@]}"; do printf '%s\n' "$g" >&3; done

# Kill live children on Ctrl-C so we don't orphan vLLM workers.
cleanup() {
  trap - INT TERM
  echo
  echo "[$(ts)] interrupted — killing workers..." >&2
  jobs -p | xargs -r kill 2>/dev/null || true
  wait 2>/dev/null || true
  exit 130
}
trap cleanup INT TERM

dispatch() {
  local gpu="$1" args="$2" slug="$3" idx="$4" total="$5"
  local log="$LOG_DIR/${slug}.log"
  if [[ "$WRITE_CSV" != "1" ]]; then args="$args --no-write-csv"; fi
  if [[ "$OVERWRITE_CSV" = "1" ]]; then args="$args --overwrite-csv"; fi
  # NVML's device index is NOT remapped by CUDA_VISIBLE_DEVICES, but the sampler
  # in src/nvml_util.py auto-detects the active GPU via torch's PCI bus id →
  # nvmlDeviceGetHandleByPciBusId, so we don't need to pass --nvml-device here.
  echo "[$(ts)] [$idx/$total] GPU $gpu START  $slug"
  # shellcheck disable=SC2086  # we intentionally split $args on whitespace
  CUDA_VISIBLE_DEVICES="$gpu" "$VENV_PY" "$RUN_ONE" $args >"$log" 2>&1
  local rc=$?
  if (( rc == 0 )); then
    echo "[$(ts)] [$idx/$total] GPU $gpu OK     $slug  (log=$log)"
  else
    echo "[$(ts)] [$idx/$total] GPU $gpu FAIL$rc $slug (log=$log)" >&2
  fi
  # Release GPU.
  printf '%s\n' "$gpu" >&3
  return $rc
}

echo "=== sweep: $n experiment(s), GPUs [${GPUS[*]}] ==="
t0=$SECONDS
pids=()
for i in "${!EXPERIMENTS[@]}"; do
  args="${EXPERIMENTS[$i]}"
  slug="$(slugify "$args")"
  # Block until a GPU frees up.
  read -r -u 3 gpu
  dispatch "$gpu" "$args" "$slug" "$((i+1))" "$n" &
  pids+=( "$!" )
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || ((fail++))
done
dt=$((SECONDS - t0))

if (( fail > 0 )); then
  echo "$fail / $n experiments FAILED (see $LOG_DIR). Total ${dt}s." >&2
  exit 1
fi
echo "All $n experiments OK in ${dt}s. Logs in $LOG_DIR"
