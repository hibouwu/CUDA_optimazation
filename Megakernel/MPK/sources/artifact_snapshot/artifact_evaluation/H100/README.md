# H100 — single-GPU + multi-GPU sweeps (Fig. 9 row 2 + Fig. 11 + Fig. 13)

Reproduces three H100 experiments from the paper:

- **Fig. 9** (row 2) — per-token decode latency, prompt = 64, generate
  = 1024, batch sizes 1/2/4/8/16, 5 models × 4 systems. Driven by
  `run_tgx.sh`, `run_pytorch.sh`, `run_vllm.sh`, `run_sglang.sh`.
- **Fig. 11** — Qwen3-1.7B multi-GPU comparison (PyTorch / vLLM /
  SGLang / MPK at TP=2,4,8). Driven by `run_fig11_multigpu.sh` +
  `plot_fig11.py`.
- **Fig. 13** — Qwen3-1.7B compute–communication overlap ablation on
  4 GPUs: MPK with vs without overlap of allgather-style allreduce.
  Driven by `run_fig13_overlap.sh` + `plot_fig13.py`.

The Fig. 11 and Fig. 13 scripts are paper-matching for H100 but also
runnable on other architectures — set `GPU=<tag>` to retag the JSONs
(e.g., `GPU=B200` when reproducing on a Blackwell host).

## TL;DR — running Fig. 11 + Fig. 13

Reviewer one-shot (assumes a fresh H100x8 host, e.g., Lambda / Modal):

```bash
# 1. Bootstrap repo + MPI + vLLM + SGLang. Takes ~10 min the first time.
curl -sSL https://raw.githubusercontent.com/mirage-project/mirage/tgx-osdi26-ae/artifact_evaluation/setup.sh \
    | bash -s -- --with-baselines

# 2. Activate the project conda env, install NVSHMEM, and export the env vars
#    MPK reads at megakernel-compile time. See the top-level
#    `README.md` §"Multi-GPU prerequisites — NVSHMEM + MPI" for the exact
#    NVSHMEM_HOME / NVSHMEM_INC_PATH / NVSHMEM_LIB_PATH / MPI_* exports.
conda activate mirage_2
export NVSHMEM_HOME=$HOME/lib/libnvshmem-linux-x86_64-3.6.5_cuda12-archive   # adjust
export NVSHMEM_INC_PATH=$NVSHMEM_HOME/include
export NVSHMEM_LIB_PATH=$NVSHMEM_HOME/lib
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export MPI_HOME=/usr  MPI_INC_PATH=/usr/include/x86_64-linux-gnu/openmpi  MPI_LIB_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib
export BASELINES_VENV=/opt/baselines-venv   # setup.sh --with-baselines drops both vllm + sglang here
export NCCL_NVLS_ENABLE=0                   # virtualized H100s (Modal): avoids NVLS OOM at TP=4

# 3. Run. Each script auto-plots at the end.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash artifact_evaluation/H100/run_fig13_overlap.sh    # ~30 min: TP=4, BS=1..16, 2 modes
bash artifact_evaluation/H100/run_fig11_multigpu.sh   # ~90 min: TP=2/4/8, BS=1..16, 4 systems
```

Outputs:

```
results/H100/fig13/fig13.png  # overlap (NvshmemTile) vs no-overlap, BS=1..16, ~1.07–1.17x
results/H100/fig11/fig11.png  # pytorch/vllm/sglang/mpk relative perf, TP=2,4,8 subplots
```

Override TP if fewer GPUs are free:

```bash
WORLD_SIZE=2 bash artifact_evaluation/H100/run_fig13_overlap.sh
TP_SIZES=2  bash artifact_evaluation/H100/run_fig11_multigpu.sh
```

Reproduce on a different GPU type by retagging the JSONs:

```bash
GPU=B200 bash artifact_evaluation/H100/run_fig13_overlap.sh
GPU=B200 bash artifact_evaluation/H100/run_fig11_multigpu.sh
```

## Demo entry points

H100 uses the **Hopper-tuned** demo files (WGMMA-aware grid sizing,
lm_head split):

| Model                 | Demo file                              |
|-----------------------|----------------------------------------|
| Qwen3-0.6B / 1.7B / 8B| `demo/qwen3/demo_hopper.py`            |
| Qwen3-30B-A3B (MoE)   | `demo/qwen3/demo_30B_A3B_hopper.py`    |
| Llama-3.2-1B-Instruct | `demo/llama3/demo.py`                  |

The Hopper demos do not support `--save-tokens`, so latency is parsed
from the `per-token latency ... ms` stdout line. Decode length is set
via `--max-seq-length 1088` (= prompt 64 + generate 1024).

## How to run (any H100 host with CUDA 12.4)

These instructions work on **any** Linux + H100 host: bare metal,
Lambda, Crusoe, on-prem cluster. They do not depend on Modal.

### Prerequisites

- H100 80 GB with NVIDIA driver supporting CUDA 12.4
- Ubuntu 22.04+ with Python 3.10+
- HuggingFace token (`HF_TOKEN`) for Llama-3.2 (gated)

### One-shot setup

```bash
curl -sSL https://raw.githubusercontent.com/mirage-project/mirage/tgx-osdi26-ae/artifact_evaluation/setup.sh | bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export HF_TOKEN=hf_xxx
```

**Non-root hosts** (shared clusters where `/mirage` and `/opt` aren't
writable): set `MIRAGE_HOME` and `SKIP_APT` first:

```bash
export MIRAGE_HOME=$HOME/mirage-ae
export SKIP_APT=1
curl -sSL https://raw.githubusercontent.com/mirage-project/mirage/tgx-osdi26-ae/artifact_evaluation/setup.sh | bash
```

### TGX + PyTorch sweeps (~75 min total)

```bash
bash artifact_evaluation/H100/run_tgx.sh
bash artifact_evaluation/H100/run_pytorch.sh
```

### vLLM + SGLang baselines

vLLM and SGLang have fundamentally incompatible dependency cones
(different torch / transformers versions). Install each in its **own
venv** — `run_vllm.sh` auto-activates `/opt/vllm-venv` and
`run_sglang.sh` auto-activates `/opt/sglang-venv`.

**vLLM venv** (pinned, since newer vLLM ships torch built for CUDA 13):

```bash
rm -rf /opt/vllm-venv
python3 -m venv /opt/vllm-venv
source /opt/vllm-venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
pip install vllm==0.8.5
pip install transformers==4.51.3     # vllm 0.8.5 needs transformers 4.x
deactivate

bash artifact_evaluation/H100/run_vllm.sh
```

**SGLang venv** (let pip resolve internally consistent set):

```bash
rm -rf /opt/sglang-venv
python3 -m venv /opt/sglang-venv
source /opt/sglang-venv/bin/activate
pip install --upgrade pip
pip install 'sglang[all]'
deactivate

bash artifact_evaluation/H100/run_sglang.sh
```

**Non-root hosts** (e.g., shared clusters where `/opt` is not writable):
substitute `$HOME/vllm-venv` / `$HOME/sglang-venv` for `/opt/...` and
override the env vars:

```bash
BASELINES_VENV=$HOME/vllm-venv MIRAGE_HOME=$HOME/mirage-ae \
    bash $HOME/mirage-ae/artifact_evaluation/H100/run_vllm.sh
BASELINES_VENV=$HOME/sglang-venv MIRAGE_HOME=$HOME/mirage-ae \
    bash $HOME/mirage-ae/artifact_evaluation/H100/run_sglang.sh
```

### Multi-GPU prerequisites — NVSHMEM + MPI (Fig. 11 + Fig. 13)

Fig. 11 and Fig. 13 require NVSHMEM and an MPI implementation, which
the standalone `pip install -e .` workflow does **not** pull in. See
the top-level [`README.md`](../../README.md) §"Multi-GPU prerequisites
— NVSHMEM + MPI" for the install steps and the four env vars
(`NVSHMEM_INC_PATH`, `NVSHMEM_LIB_PATH`, `MPI_INC_PATH`, `MPI_LIB_PATH`)
that MPK looks up at compile time.

### Fig. 11 — Qwen3-1.7B multi-GPU comparison (TP=2,4,8)

Reproduces the cross-system bar chart: relative performance of
PyTorch, vLLM, SGLang, and MPK on Qwen3-1.7B under tensor parallelism,
normalized to MPK at each (TP, batch_size).

Per-system invocation:

- **pytorch**: `mpirun -np $TP python demo/qwen3/demo.py --model
  Qwen/Qwen3-1.7B …` (no `--use-mirage`). Falls into the vanilla HF
  decode loop in `demo.py`; TP collectives come from `Qwen3ShardLoader`.
  Note: this path forces `total_num_requests=1` in `demo.py`, so the
  PyTorch curve is effectively single-request across the BS sweep
  (matches Fig 9 PyTorch behavior).
- **vllm**: `vllm bench latency --tensor-parallel-size $TP …` from
  `$VLLM_VENV` (default `/opt/vllm-venv`, or `$BASELINES_VENV` if set).
  vLLM manages its own worker procs; no mpirun.
- **sglang**: `python -m sglang.bench_one_batch --tensor-parallel-size
  $TP …` from `$SGLANG_VENV` (default `/opt/sglang-venv`, or
  `$BASELINES_VENV` if set).
- **mpk**: `mpirun -np $TP python demo/qwen3/demo.py --use-mirage …`
  from the mirage_2 conda env.

**One-time baseline venv setup.** Run `setup.sh --with-baselines` to
install both vLLM and SGLang into `/opt/baselines-venv`, then point
the Fig 11 script at that shared venv via `BASELINES_VENV`:

```bash
bash artifact_evaluation/setup.sh --with-baselines
export BASELINES_VENV=/opt/baselines-venv
```

The shared venv works for vLLM versions that don't require the older
`flash_attn.ops` API (latest releases). If you hit
`ModuleNotFoundError: No module named 'flash_attn.ops'` from vLLM,
install vLLM into its own venv at `/opt/vllm-venv` (or `$HOME/vllm-venv`)
to keep its `flash_attn` lineage clean.

**MPK batched-token policy.** The run scripts auto-set
`--max-num-batched-tokens = max(8, bs)` per cell (8 for bs ≤ 8, bs
itself when bs > 8) to match the wider AE sweep convention. This is
baked into both `run_fig11_multigpu.sh` and `run_fig13_overlap.sh`;
no manual flag needed.

**Cloud / virtualized hosts (Modal, etc.).** NCCL's NVLink-SHARP (NVLS)
transport can OOM during NVSHMEM team allocation on virtualized H100s
(seen at TP=4 with ~32 NVSHMEM teams). Set
`export NCCL_NVLS_ENABLE=0` before running the scripts; both already
forward this env var to all MPI ranks via `mpirun -x`.

Run the sweep:

```bash
# Ensure mirage_2 is the active env (used for pytorch + mpk runs).
conda activate mirage_2

# Pick free GPUs. For TP=4 you need 4 idle; for TP=8, eight.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Default: TP_SIZES="2 4 8" (override to "2 4" or "2" for fewer GPUs).
# The script auto-invokes plot_fig11.py at the end and writes fig11.png.
bash artifact_evaluation/H100/run_fig11_multigpu.sh
```

`run_fig11_multigpu.sh` writes
`results/$GPU/fig11/<system>__tp<TP>__bs<BS>.json` (default
`results/H100/fig11/…`) with `latency_ms_per_token`. `plot_fig11.py`
normalizes per-(TP, BS) by MPK and emits `fig11.png` with one subplot
per TP value and a "Nx" speedup annotation above each batch-size
group.

### Fig. 13 — compute–communication overlap ablation (TP=4)

Measures the speedup from overlapping the allgather phase of MPK's
split-phase allreduce with downstream computation, on Qwen3-1.7B with
tensor parallelism across 4 GPUs.

Two compiler env vars gate the experiment:

| Env var                        | Layer  | What it does                                                              |
|--------------------------------|--------|---------------------------------------------------------------------------|
| `MPK_FORCE_ALLGATHER_REDUCE=1` | Python | Skip the NvshmemTile path on SM≥90; force allgather + local-reduce.       |
| `MPK_DISABLE_AG_OVERLAP=1`     | C++    | Override `event_dim=1` on edges touching the allgather task (serializes). |

`MPK_DISABLE_AG_OVERLAP` is read at compile time inside
`build_annotated_graph()`; changes take effect on the next compile.
Both env vars default off — leaving them unset preserves stock
behavior.

```bash
# Pick 4 idle GPUs on the host.
export CUDA_VISIBLE_DEVICES=0,1,2,3

# The script auto-invokes plot_fig13.py at the end and writes fig13.png.
bash artifact_evaluation/H100/run_fig13_overlap.sh
```

The run script sweeps `{overlap, no-overlap} × {bs=1,2,4,8,16}` and
writes `results/$GPU/fig13/mpk-{overlap,no-overlap}__bs<bs>.json`
(default `results/H100/fig13/…`). `plot_fig13.py` consumes those
JSONs and produces `fig13.png` in the same directory, with the
per-pair speedup annotated above each bar pair (paper reports ~1.1×).

The shared HF cache at `/raid/catalyst/models` already contains
`Qwen/Qwen3-1.7B` on the cluster used for development; the script
defaults `HF_HOME` there. Override `HF_HOME` / `HF_ID` on other hosts.

If your shell exports `NVSHMEM_LIB_PATH` (the standard NVSHMEM install
layout — `~/.bashrc` usually sets it), the script auto-sets
`LD_PRELOAD` to the matching `libnvshmem_host.so.3`. Without this the
compiled megakernel can pick up an older `/usr/lib` libnvshmem and
fail at import with `undefined symbol: nvshmem_selected_device_transport`.
On hosts without a curated `NVSHMEM_LIB_PATH`, export it manually
before invoking the script.

### Output schema

Same as A100 (`results/H100/<system>/<model_tag>__bs<bs>.json` with
`latency_ms_per_token`). Fig. 13 additionally records `mode`,
`world_size`, and the two MPK env-var states per JSON.

If you don't have an H100 host, see the **Optional: cloud hosting on
Modal** section in the top-level [`README.md`](../../README.md).
