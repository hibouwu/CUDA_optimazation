#!/usr/bin/env bash
# Bootstrap a bare GPU host into a state where artifact_evaluation/<gpu>/*.sh
# can run. Idempotent: re-running is safe.
#
# Usage:
#   bash artifact_evaluation/setup.sh                  # TGX/MPK install
#   bash artifact_evaluation/setup.sh --with-baselines # also install vLLM + SGLang
#   bash artifact_evaluation/setup.sh --baselines-only # only vLLM + SGLang (no MPK)
#
# Honors:
#   MIRAGE_HOME (default: /mirage)
#   BRANCH      (default: tgx-osdi26-ae)
#   HF_TOKEN    (forwarded to huggingface-cli login if set)
#
# Tested on: Ubuntu 22.04 + CUDA 12.4 (the Modal ae_ssh image, and a fresh
# Lambda H100 instance).

set -euo pipefail

MIRAGE_HOME="${MIRAGE_HOME:-/mirage}"
BRANCH="${BRANCH:-tgx-osdi26-ae}"
WITH_BASELINES=0
BASELINES_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --with-baselines)  WITH_BASELINES=1 ;;
        --baselines-only)  BASELINES_ONLY=1; WITH_BASELINES=1 ;;
        -h|--help)
            sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# ---------- 1. system packages ----------
SKIP_APT="${SKIP_APT:-0}"
need_apt=()
for pkg in git curl build-essential pkg-config wget python3 python3-pip; do
    dpkg -s "$pkg" >/dev/null 2>&1 || need_apt+=("$pkg")
done
# Multi-GPU runs need MPI; harmless to install even on single-GPU hosts.
for pkg in libmpich-dev libopenmpi-dev openmpi-bin; do
    dpkg -s "$pkg" >/dev/null 2>&1 || need_apt+=("$pkg")
done
if (( ${#need_apt[@]} && ! SKIP_APT )); then
    echo "[setup] apt installing: ${need_apt[*]}"
    if [[ "$EUID" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
        APT_PREFIX="sudo"
    elif [[ "$EUID" -ne 0 ]]; then
        echo "[setup] WARNING: not root and no sudo. Skipping apt step." >&2
        echo "[setup] Set SKIP_APT=1 to silence, or ensure these are installed:" >&2
        echo "[setup]   ${need_apt[*]}" >&2
        APT_PREFIX="false"   # makes apt commands no-op
    else
        APT_PREFIX=""
    fi
    if [[ "$APT_PREFIX" != "false" ]]; then
        $APT_PREFIX env DEBIAN_FRONTEND=noninteractive apt-get update -qq || true
        $APT_PREFIX env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${need_apt[@]}" || \
            echo "[setup] WARNING: apt install failed; assuming packages are present" >&2
    fi
fi

# ---------- 2. mirage repo ----------
if (( ! BASELINES_ONLY )); then
    # If $MIRAGE_HOME exists but isn't a git repo (e.g. fresh Modal SSH
    # container with the results-volume symlink already in place), clear
    # it before cloning. Preserve any volume-mount symlinks (they'll be
    # restored after the clone).
    declare -A _saved_links
    if [[ -d "$MIRAGE_HOME" && ! -d "$MIRAGE_HOME/.git" ]]; then
        echo "[setup] $MIRAGE_HOME exists but is not a git repo; clearing"
        for entry in "$MIRAGE_HOME"/*; do
            [[ -L "$entry" ]] || continue
            target=$(readlink "$entry")
            _saved_links["$(basename "$entry")"]="$target"
        done
        rm -rf "$MIRAGE_HOME"
    fi
    if [[ ! -d "$MIRAGE_HOME/.git" ]]; then
        echo "[setup] cloning mirage@${BRANCH} into $MIRAGE_HOME"
        git clone --recursive --branch "$BRANCH" \
            https://github.com/mirage-project/mirage.git "$MIRAGE_HOME"
        # restore preserved symlinks (e.g. results -> volume)
        for name in "${!_saved_links[@]}"; do
            target="${_saved_links[$name]}"
            link_path="$MIRAGE_HOME/$name"
            [[ -e "$link_path" ]] && rm -rf "$link_path"
            ln -s "$target" "$link_path"
            echo "[setup] restored symlink $link_path -> $target"
        done
    else
        echo "[setup] updating $MIRAGE_HOME (branch $BRANCH)"
        git -C "$MIRAGE_HOME" fetch --quiet origin "$BRANCH"
        git -C "$MIRAGE_HOME" checkout --quiet "$BRANCH"
        git -C "$MIRAGE_HOME" reset --hard --quiet "origin/$BRANCH"
        git -C "$MIRAGE_HOME" submodule update --init --recursive --quiet
    fi
    export MIRAGE_HOME
    BASHRC="${HOME:-/root}/.bashrc"
    if [[ -w "$BASHRC" || ! -e "$BASHRC" ]]; then
        grep -q '^export MIRAGE_HOME=' "$BASHRC" 2>/dev/null \
            || echo "export MIRAGE_HOME=$MIRAGE_HOME" >> "$BASHRC"
    fi

    # CUDA toolchain on PATH (nvcc lives in /usr/local/cuda/bin on the
    # nvidia/cuda:*-devel base image, but PATH isn't set by default).
    if [[ -d /usr/local/cuda/bin ]]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        export CUDA_HOME="/usr/local/cuda"
        if [[ -w "$BASHRC" || ! -e "$BASHRC" ]]; then
            grep -q '^export PATH=/usr/local/cuda/bin' "$BASHRC" 2>/dev/null \
                || cat >> "$BASHRC" <<'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
EOF
        fi
    fi

    # ---------- 3. rust (only needed for abstract_subexpr / formal_verifier) ----------
    if ! command -v rustc >/dev/null 2>&1; then
        echo "[setup] installing rust toolchain"
        curl -sSf https://sh.rustup.rs | bash -s -- -y --default-toolchain stable
    fi
    export PATH="$HOME/.cargo/bin:$PATH"

    # ---------- 4. python deps + MPK build ----------
    # Detect PEP 668 "externally-managed-environment". On modern Debian/Ubuntu
    # the system python blocks pip; auto-create a venv if so.
    if pip install --dry-run --quiet pip 2>&1 | grep -q "externally-managed"; then
        MPK_VENV="${MPK_VENV:-$HOME/mirage-venv}"
        echo "[setup] system python is PEP 668 protected; using venv at $MPK_VENV"
        if [[ ! -d "$MPK_VENV" ]]; then
            python3 -m venv "$MPK_VENV"
        fi
        # shellcheck disable=SC1091
        source "$MPK_VENV/bin/activate"
        if [[ -w "$BASHRC" || ! -e "$BASHRC" ]]; then
            grep -q "^source $MPK_VENV/bin/activate" "$BASHRC" 2>/dev/null \
                || echo "source $MPK_VENV/bin/activate" >> "$BASHRC"
        fi
    fi

    # Detect GPU compute capability so we can pick the right torch wheel.
    # B200 = sm_100, requires torch 2.7+cu128. Hopper/Ampere use torch 2.6+cu124.
    CC=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.')
    fi
    case "$CC" in
        100|110|120)
            TORCH_INDEX="https://download.pytorch.org/whl/cu128"
            TORCH_VER="2.7.0"
            FLASHINFER_INDEX="https://flashinfer.ai/whl/cu128/torch2.7"
            echo "[setup] detected sm_${CC} (Blackwell+); using torch 2.7+cu128"
            ;;
        *)
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            TORCH_VER="2.6.0"
            FLASHINFER_INDEX="https://flashinfer.ai/whl/cu124/torch2.6"
            echo "[setup] detected sm_${CC:-unknown}; using torch 2.6+cu124 (Ampere/Hopper)"
            ;;
    esac

    echo "[setup] pip install MPK + deps (this builds the C++/CUDA extension)"
    pip install --upgrade pip
    pip install --index-url "$TORCH_INDEX" "torch==$TORCH_VER"
    pip install transformers mpi4py
    (cd "$MIRAGE_HOME" && pip install -e . -v)
    pip install flashinfer-python -i "$FLASHINFER_INDEX"
fi

# ---------- 5. baselines (optional) ----------
if (( WITH_BASELINES )); then
    echo "[setup] pip install vLLM + SGLang (baselines)"
    # vllm and sglang each manage their own torch pin; install in a fresh
    # virtualenv to avoid clobbering the MPK environment.
    if (( BASELINES_ONLY )); then
        pip install vllm
        pip install 'sglang[all]'
    else
        BASELINES_VENV="${BASELINES_VENV:-/opt/baselines-venv}"
        if [[ ! -d "$BASELINES_VENV" ]]; then
            python3 -m venv "$BASELINES_VENV"
        fi
        # shellcheck disable=SC1091
        source "$BASELINES_VENV/bin/activate"
        pip install --upgrade pip
        pip install vllm
        pip install 'sglang[all]'
        deactivate
        echo "[setup] baselines installed in $BASELINES_VENV"
        echo "        activate before running run_vllm.sh / run_sglang.sh:"
        echo "        source $BASELINES_VENV/bin/activate"
    fi
fi

# ---------- 6. HF auth ----------
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[setup] logging in to HuggingFace"
    pip install --quiet 'huggingface_hub[cli]' || true
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null
fi

# ---------- summary ----------
echo
echo "[setup] done."
if (( ! BASELINES_ONLY )); then
    echo "  MIRAGE_HOME=$MIRAGE_HOME"
    echo "  python -c 'import mirage; print(mirage.__file__)'"
fi
echo
echo "Next:"
echo "  bash $MIRAGE_HOME/artifact_evaluation/H100/run_tgx.sh"
