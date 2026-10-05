# Artifact Evaluation — TGX (OSDI '26 paper #74)

*TGX: A Compiler and Runtime for Mega-Kernelizing Tensor Programs*

> **Branch `tgx-osdi26-ae`** — the frozen artifact for OSDI '26 AE.

This README documents (1) the paper and how to cite it, (2) what's
reproduced, (3) how to reproduce each figure, (4) prerequisites, and
(5) optional cloud hosting.

---

## Paper

**TGX: A Compiler and Runtime for Mega-Kernelizing Tensor Programs**
Xinhao Cheng, Zhihao Zhang, Yu Zhou, Jianan Ji, Jinchen Jiang, Zepeng
Zhao, Ziruo Xiao, Zihao Ye, Yingyi Huang, Ruihang Lai, Hongyi Jin,
Bohan Hou, Mengdi Wu, Yixin Dong, Anthony Yip, Zihao Ye, Songting
Wang, Wenqin Yang, Xupeng Miao, Tianqi Chen, Zhihao Jia.
*Proceedings of the 20th USENIX Symposium on Operating Systems Design
and Implementation (OSDI '26)*. Paper #74.

Affiliations: Carnegie Mellon University, Tsinghua University, NVIDIA,
University of Michigan, Purdue University.

Code & artifact: <https://github.com/mirage-project/mirage> (this
branch: `tgx-osdi26-ae`).

### Cite

```bibtex
@inproceedings{tgx-osdi26,
  title     = {{TGX}: A Compiler and Runtime for Mega-Kernelizing
               Tensor Programs},
  author    = {Cheng, Xinhao and Zhang, Zhihao and Zhou, Yu and Ji, Jianan
               and Jiang, Jinchen and Zhao, Zepeng and Xiao, Ziruo and
               Ye, Zihao and Huang, Yingyi and Lai, Ruihang and Jin, Hongyi
               and Hou, Bohan and Wu, Mengdi and Dong, Yixin and Yip, Anthony
               and Ye, Zihao and Wang, Songting and Yang, Wenqin and
               Miao, Xupeng and Chen, Tianqi and Jia, Zhihao},
  booktitle = {Proceedings of the 20th USENIX Symposium on Operating
               Systems Design and Implementation (OSDI '26)},
  year      = {2026},
}
```

---

## What's reproduced

The artifact reproduces five figures from the paper.

| Figure | Experiment | GPU |
|--------|------------|-----|
| Fig. 9 | Per-token decode latency, 5 models × 5 batch sizes × 4 systems | A100 / H100 / B200 (single) |
| Fig. 10 | Qwen3-30B-A3B MoE microbench: TGX-Hybrid-MoE vs SGLang-MoE | B200 (single) |
| Fig. 11 | Qwen3-1.7B multi-GPU tensor-parallel comparison, TP = 2 / 4 / 8 | H100 × N |
| Fig. 12 | Qwen3-8B cross-task pipelining ablation (TGX-Pipe vs TGX-No-Pipe) | B200 (single) |
| Fig. 13 | Qwen3-1.7B compute–communication overlap ablation, TP = 4 | H100 × 4 |

### Models, batch sizes

| Paper name             | HuggingFace ID                       | Demo entry point                        |
|------------------------|--------------------------------------|-----------------------------------------|
| Qwen3-0.6B             | `Qwen/Qwen3-0.6B`                    | `demo/qwen3/demo.py` (A100/B200) / `demo_hopper.py` (H100) |
| Llama-3.2-1B-Instruct  | `meta-llama/Llama-3.2-1B-Instruct`   | `demo/llama3/demo.py`                   |
| Qwen3-1.7B             | `Qwen/Qwen3-1.7B`                    | `demo/qwen3/demo.py` / `demo_hopper.py` |
| Qwen3-8B               | `Qwen/Qwen3-8B`                      | `demo/qwen3/demo.py` / `demo_hopper.py` |
| Qwen3-30B-A3B (MoE)    | `Qwen/Qwen3-30B-A3B`                 | `demo/qwen3/demo_30B_A3B.py` / `demo_30B_A3B_hopper.py` |

**Batch sizes:** 1, 2, 4, 8, 16 (per the paper's offline batched setup).

**Workload.** Offline batched inference, prompt length **64**, decode
**1024** tokens, batch sizes **{1, 2, 4, 8, 16}**, greedy
(`--temperature 0`). Numbers in the paper are the median of 5 runs
after a 4-iteration warmup.

**Metric.** Per-token decoding latency in ms, parsed from each demo's
stdout line:

```
Prompt length 64, generate length 1024, per-token latency (both prefill and decode): X.XXX ms
```

All baselines are converted to the same metric. Each cell produces a
JSON of the shape `{system, gpu, model, batch_size, latency_ms_per_token, ...}`.

---

## Reproduction by figure

Each figure has dedicated scripts under `artifact_evaluation/<gpu>/`.
Per-GPU READMEs (linked below) contain the exact commands; this
section is the index.

### Fig. 9 — single-GPU end-to-end (A100 / H100 / B200)

Per-GPU sweep driving all four systems (TGX, PyTorch, vLLM, SGLang)
across the 5 models × 5 batch sizes:

```bash
bash artifact_evaluation/<gpu>/run_tgx.sh
bash artifact_evaluation/<gpu>/run_pytorch.sh
bash artifact_evaluation/<gpu>/run_vllm.sh
bash artifact_evaluation/<gpu>/run_sglang.sh
```

Outputs: `results/<gpu>/<system>/<model_tag>__bs<bs>.json`.

Per-GPU READMEs:
[A100](artifact_evaluation/A100/README.md) ·
[H100](artifact_evaluation/H100/README.md) ·
[B200](artifact_evaluation/B200/README.md).

### Fig. 10 — MoE microbenchmark (Qwen3-30B-A3B, B200)

Compares TGX's hybrid workload balancer + fused gather-GEMM to SGLang's
MoE implementation:

```bash
bash artifact_evaluation/B200/run_fig10_moe.sh                  # TGX-Hybrid-MoE
bash artifact_evaluation/B200/run_sglang.sh \
    MODELS=qwen3-30b-a3b OUTPUT_ROOT=results/B200/fig10/sglang   # SGLang-MoE
```

Outputs land in `results/B200/fig10/<system>__bs<bs>.json` with
`latency_ms_per_token`. Qwen3-30B-A3B is MoE-dominated, so per-token
latency tracks MoE-block runtime closely.

The paper's third bar, `TGX-Static-MoE`, is an internal ablation of
the hybrid workload balancer and is not exposed as a separate runtime
flag in this artifact. For the per-MoE-block μs numbers, profile the
`moe_w13_linear` / `moe_w2_linear` kernels with NCU:

```bash
ncu --kernel-name regex:'moe_w[12]3?_linear' --launch-count 4 \
    python demo/qwen3/demo_30B_A3B.py --use-mirage \
    --model Qwen/Qwen3-30B-A3B --max-num-batched-requests 1 \
    --max-seq-length 128 --ignore-eos
```

The historical `MAX_TOKENS = 1` attention-header edit that earlier
drafts of `demo_30B_A3B.py` required is now handled automatically at
compile time by `src/kernel/task_register.cc` (commit `688632e`): when
`NUM_QO_PER_KV >= 8` the task registry instantiates the attention
kernel template with `MAX_TOKENS=4`. No manual rebuild needed.

### Fig. 11 — multi-GPU TP comparison (H100 × 2 / 4 / 8)

Cross-system tensor-parallel scaling for Qwen3-1.7B. Requires NVSHMEM +
MPI — see §"Prerequisites — Multi-GPU additions" below.

```bash
bash artifact_evaluation/H100/run_fig11_multigpu.sh   # ~90 min: TP=2/4/8, BS=1..16, 4 systems
```

Auto-plots `results/H100/fig11/fig11.png`. Subplots per TP show
relative throughput vs TGX.

Override TP if fewer GPUs are free:

```bash
TP_SIZES=2 bash artifact_evaluation/H100/run_fig11_multigpu.sh
```

Reproduce on a different GPU type by retagging the JSONs:

```bash
GPU=B200 bash artifact_evaluation/H100/run_fig11_multigpu.sh
```

See [H100 README §"Fig. 11"](artifact_evaluation/H100/README.md) for
the full reviewer one-shot recipe (Modal-friendly).

### Fig. 12 — cross-task pipelining ablation (Qwen3-8B, B200)

Compares two TGX configurations differing only in the grid size of the
final `lm_head` linear layer:

- `TGX-Pipe` (`grid_dim[0] = 128`): few enough tasks that the
  cross-task pipeliner engages — the next layer's pre-load overlaps
  the current layer's compute.
- `TGX-No-Pipe` (`grid_dim[0] = vocab_size // 256 = 600`): too many
  tasks for pipelining to fit.

```bash
bash artifact_evaluation/B200/run_fig12_pipe_ablation.sh
```

Outputs `results/B200/fig12/tgx-pipe__bs<bs>.json` and
`results/B200/fig12/tgx-no-pipe__bs<bs>.json`. The pipe-minus-no-pipe
per-token-latency delta approximates the lm-head pipelining benefit.
For the absolute per-layer μs numbers, profile the final
`linear_layer` kernel with NCU:

```bash
ncu --kernel-name regex:'.*linear.*' --launch-count 8 \
    python demo/qwen3/demo.py --use-mirage --model Qwen/Qwen3-8B \
    --max-num-batched-requests 1 --max-seq-length 128 \
    --lm-head-grid pipe --ignore-eos
# repeat with --lm-head-grid no-pipe
```

The `--lm-head-grid` flag is opt-in (defaults to `default` =
`mpk.num_workers`), so existing Fig. 9 sweeps are unaffected.

The cuBLAS curve in Fig. 12 is the same Qwen3-8B Fig. 9 cell —
re-run with `bash artifact_evaluation/B200/run_pytorch.sh MODELS=qwen3-8b`
and read `latency_ms_per_token` from `results/B200/pytorch/qwen3-8b__bs<bs>.json`.

### Fig. 13 — compute–communication overlap ablation (H100 × 4)

Qwen3-1.7B at TP=4, comparing TGX with and without overlap of
allgather-style allreduce. Requires NVSHMEM + MPI.

```bash
bash artifact_evaluation/H100/run_fig13_overlap.sh    # ~30 min: TP=4, BS=1..16, 2 modes
```

Auto-plots `results/H100/fig13/fig13.png` (overlap vs no-overlap,
expected ~1.07–1.17× speedup with overlap).

Override world size if fewer GPUs are free:

```bash
WORLD_SIZE=2 bash artifact_evaluation/H100/run_fig13_overlap.sh
```

See [H100 README §"Fig. 13"](artifact_evaluation/H100/README.md) for
the full reviewer one-shot recipe.

---

## Prerequisites

### 1. Single-GPU setup (`setup.sh`)

```bash
curl -sSL https://raw.githubusercontent.com/mirage-project/mirage/tgx-osdi26-ae/artifact_evaluation/setup.sh | bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export HF_TOKEN=hf_xxx                                    # for Llama-3.2 (gated)
```

`setup.sh` clones the branch into `/mirage`, installs apt + pip deps,
auto-detects the GPU compute capability, and builds TGX. Takes
~10–15 min on first run. Sufficient for Fig. 9, Fig. 10, Fig. 12.

### 2. Multi-GPU additions — NVSHMEM + MPI (Fig. 11 + Fig. 13)

The multi-GPU experiments require NVSHMEM and an MPI implementation.
The single-GPU `setup.sh` does **not** pull these in — install them
once before running the multi-GPU sweeps.

**What TGX looks up.** At compile time
`mpk.PersistentKernel.compile()` reads four env vars and validates the
corresponding files exist (see
[`python/mirage/mpk/persistent_kernel.py:2376-2442`](python/mirage/mpk/persistent_kernel.py#L2376-L2442)):

| Env var             | Validates                                      | Falls back to                  |
|---------------------|------------------------------------------------|--------------------------------|
| `NVSHMEM_INC_PATH`  | `$NVSHMEM_INC_PATH/nvshmem.h`                  | `/usr/include/nvshmem_12/`     |
| `NVSHMEM_LIB_PATH`  | `$NVSHMEM_LIB_PATH/libnvshmem_device.a`        | `/usr/lib/x86_64-linux-gnu/`   |
| `MPI_INC_PATH`      | `$MPI_INC_PATH/mpi.h`                          | `/usr/include/`                |
| `MPI_LIB_PATH`      | `$MPI_LIB_PATH/libmpi.so`                      | `/usr/lib/`                    |

The generated `nvcc` command links with `-ccbin=mpic++ -lnvshmem_host
-lnvshmem_device -lmpi` and bakes `$NVSHMEM_LIB_PATH` and `$MPI_LIB_PATH`
into `DT_RPATH` (with `--disable-new-dtags`).

**Install NVSHMEM.** The compiler needs **both** the host library
(`libnvshmem_host.so`) and the device library
(`libnvshmem_device.a`). The redistributable archive from NVIDIA
Developer (`libnvshmem-linux-x86_64-<ver>_cuda<cuda>-archive`) contains
both. Pick the CUDA major matching your toolkit (e.g.  `cuda13` for
CUDA 12.x/13.x, `cuda12` for CUDA 12.x).

```bash
# 1. Download the prebuilt archive (no sudo needed).
NVSHMEM_VER=3.6.5
NVSHMEM_CUDA=cuda13
mkdir -p $HOME/lib && cd $HOME/lib
wget -q https://developer.download.nvidia.com/compute/redist/nvshmem/${NVSHMEM_VER}/local_installers/libnvshmem-linux-x86_64-${NVSHMEM_VER}_${NVSHMEM_CUDA}-archive.tar.xz
tar xJf libnvshmem-linux-x86_64-${NVSHMEM_VER}_${NVSHMEM_CUDA}-archive.tar.xz
export NVSHMEM_HOME=$HOME/lib/libnvshmem-linux-x86_64-${NVSHMEM_VER}_${NVSHMEM_CUDA}-archive

# 2. Export the env vars TGX reads.
export NVSHMEM_INC_PATH=$NVSHMEM_HOME/include
export NVSHMEM_LIB_PATH=$NVSHMEM_HOME/lib
export NVSHMEM_PREFIX=$NVSHMEM_HOME
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:${LD_LIBRARY_PATH:-}

# 3. Verify.
ls $NVSHMEM_INC_PATH/nvshmem.h $NVSHMEM_LIB_PATH/libnvshmem_device.a
```

**Install MPI.** Any MPI ≥ 3.1 works. The simplest path is
conda-installed Open MPI (no root needed); on hosts where MPI is
already system-installed (`apt install libopenmpi-dev`), point the env
vars at the system paths instead.

```bash
# Option A — conda (recommended for non-root hosts; assumes the
# project conda env, e.g. mirage_2, is active):
conda install -y -c conda-forge openmpi mpi4py
export MPI_HOME=$CONDA_PREFIX
export MPI_INC_PATH=$MPI_HOME/include
export MPI_LIB_PATH=$MPI_HOME/lib

# Option B — Ubuntu system Open MPI:
sudo apt install -y libopenmpi-dev openmpi-bin
export MPI_HOME=/usr
export MPI_INC_PATH=/usr/include/openmpi
export MPI_LIB_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib

# Verify mpic++ is on PATH and resolves to the same install.
which mpic++ mpirun
ls $MPI_INC_PATH/mpi.h $MPI_LIB_PATH/libmpi.so
```

The Python multi-GPU launch path uses `mpi4py`; the conda command
above pulls it in. With Option B, also run `pip install mpi4py`.

**Persist the env vars.** Add the exports to `~/.bashrc` (or a
project setup script) so every shell that runs the AE scripts inherits
them:

```bash
cat >> ~/.bashrc <<'EOF'
# TGX multi-GPU build deps
export NVSHMEM_HOME=$HOME/lib/libnvshmem-linux-x86_64-3.6.5_cuda13-archive
export NVSHMEM_INC_PATH=$NVSHMEM_HOME/include
export NVSHMEM_LIB_PATH=$NVSHMEM_HOME/lib
export NVSHMEM_PREFIX=$NVSHMEM_HOME
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export MPI_HOME=$CONDA_PREFIX                  # adjust if not using conda
export MPI_INC_PATH=$MPI_HOME/include
export MPI_LIB_PATH=$MPI_HOME/lib
EOF
```

Once `NVSHMEM_LIB_PATH` is set, `run_fig13_overlap.sh` and
`run_fig11_multigpu.sh` automatically prepend the matching
`libnvshmem_host.so.3` to `LD_PRELOAD` — avoiding the well-known
`undefined symbol: nvshmem_selected_device_transport` import error
caused by picking up an older `/usr/lib` NVSHMEM at runtime.

**Rebuild requirements.** You do not need to rebuild the C++ runtime
library after installing or upgrading NVSHMEM / MPI — each kernel
cell re-emits its `nvcc` command with the currently-set paths. Only
edits under `src/` require `pip install -e . --no-deps -v`.

### 3. Baseline systems — vLLM + SGLang venvs

vLLM and SGLang have incompatible torch / transformers pins and must
be installed in separate venvs. The `setup.sh --with-baselines` flag
creates both at `/opt/vllm-venv` and `/opt/sglang-venv`. Each per-GPU
`run_vllm.sh` / `run_sglang.sh` auto-activates the corresponding venv.

See the per-GPU README for the exact install commands when bootstrapping
manually:
[A100](artifact_evaluation/A100/README.md) ·
[H100](artifact_evaluation/H100/README.md) ·
[B200](artifact_evaluation/B200/README.md).

---

## Optional: cloud hosting on Modal

If you don't have a GPU host, contact the authors — we can provide
Modal access.

### Local prereqs (one-time)

```bash
pip install modal
modal setup    # browser-based auth
```

Make sure `~/.ssh/id_rsa.pub` exists (`ssh-keygen -t rsa` if not).

### Start a GPU box

```bash
# pick one
modal run scripts/ae/ae_ssh.py --gpu a100-40gb
modal run scripts/ae/ae_ssh.py --gpu a100-80gb
modal run scripts/ae/ae_ssh.py --gpu h100
modal run scripts/ae/ae_ssh.py --gpu h100x4
modal run scripts/ae/ae_ssh.py --gpu h100x8
modal run scripts/ae/ae_ssh.py --gpu b200
```

The launcher prints:

```
SSH ready:  ssh root@<host>.modal.host -p <port>
```

In another terminal, paste that ssh line; you're now on a fresh
Modal-hosted GPU box. Run the per-GPU `setup.sh` + sweep scripts as in
the previous section.

### Persistent volumes

The Modal container mounts:

- `/root/.cache/huggingface` → `tgx-ae-hf-cache` Volume (HF weights
  reused across runs)
- `/mirage/results` → `tgx-ae-results` Volume (all benchmark JSONs
  persist across container restarts)

The launcher commits the results volume every 2 minutes to survive
ungraceful container exits (OOM, force-stop, etc.).

---

## Reproduction notes

- **First TGX run is slow.** Triggers a one-time NVCC compile of the
  megakernel (~1–5 min depending on model). Cached under
  `~/.cache/mirage/`. To pre-warm, run with a small `--max-seq-length`
  once.
- **Run-to-run variance.** Sweep drivers run a built-in warmup before
  timing.
- **HuggingFace gating.** Llama-3.2 requires accepting Meta's license.
  Set `HF_TOKEN=<your_token>` before the sweep.
- **Qwen3-30B-A3B on A100** is omitted in Fig. 9 (paper §6.2) due to
  OOM on a 40 GB A100.
- **Qwen3-30B-A3B on H100** at batch sizes >1 currently scales linearly
  with batch (kernel limitation, see
  [`artifact_evaluation/H100/README.md`](artifact_evaluation/H100/README.md)).
- **Multi-GPU NCCL on virtualized hosts.** On Modal-hosted H100s set
  `NCCL_NVLS_ENABLE=0` to avoid NVLS OOM at TP=4.

---

## Layout

```
artifact_evaluation/
├── setup.sh                   # bootstrap TGX on a fresh GPU host
├── A100/                      # Fig. 9 row 3
├── H100/                      # Fig. 9 row 2 + Fig. 11 + Fig. 13
│   ├── run_tgx.sh             # Fig. 9
│   ├── run_pytorch.sh         # Fig. 9
│   ├── run_vllm.sh            # Fig. 9
│   ├── run_sglang.sh          # Fig. 9
│   ├── run_fig11_multigpu.sh  # Fig. 11
│   ├── run_fig13_overlap.sh   # Fig. 13
│   ├── plot_fig11.py
│   └── plot_fig13.py
└── B200/                      # Fig. 9 row 1 + Fig. 10 + Fig. 12
    ├── run_tgx.sh             # Fig. 9
    ├── run_pytorch.sh         # Fig. 9
    ├── run_vllm.sh            # Fig. 9
    ├── run_sglang.sh          # Fig. 9
    ├── run_fig10_moe.sh       # Fig. 10
    └── run_fig12_pipe_ablation.sh  # Fig. 12
scripts/ae/
├── ae_ssh.py                  # Optional: Modal SSH launcher
└── ae_modal.py                # Optional: Modal one-shot launcher
```
