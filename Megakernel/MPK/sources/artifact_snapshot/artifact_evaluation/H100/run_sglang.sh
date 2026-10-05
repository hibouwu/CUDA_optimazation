#!/usr/bin/env bash
# E1 single-H100 sweep, SGLang baseline. See artifact_evaluation/H100/README.md.
# Output: results/H100/sglang/<model_tag>__bs<bs>.{log,json}

set -euo pipefail

export PATH="${CUDA_BIN:-/usr/local/cuda/bin}:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

BASELINES_VENV="${BASELINES_VENV:-/opt/sglang-venv}"
if [[ "$BASELINES_VENV" != "skip" && -f "$BASELINES_VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$BASELINES_VENV/bin/activate"
fi

MIRAGE_HOME="${MIRAGE_HOME:-/mirage}"
cd "$MIRAGE_HOME"

OUTPUT_ROOT="${OUTPUT_ROOT:-$MIRAGE_HOME/results/H100/sglang}"
mkdir -p "$OUTPUT_ROOT"

MODELS="${MODELS:-qwen3-0.6b llama-3.2-1b qwen3-1.7b qwen3-8b qwen3-30b-a3b}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16}"
INPUT_LEN="${INPUT_LEN:-64}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"

declare -A HF_ID
HF_ID[qwen3-0.6b]="Qwen/Qwen3-0.6B"
HF_ID[llama-3.2-1b]="meta-llama/Llama-3.2-1B-Instruct"
HF_ID[qwen3-1.7b]="Qwen/Qwen3-1.7B"
HF_ID[qwen3-8b]="Qwen/Qwen3-8B"
HF_ID[qwen3-30b-a3b]="Qwen/Qwen3-30B-A3B"

run_cell() {
    local model_tag="$1"
    local bs="$2"
    local hfid="${HF_ID[$model_tag]}"
    local log="$OUTPUT_ROOT/${model_tag}__bs${bs}.log"
    local json="$OUTPUT_ROOT/${model_tag}__bs${bs}.json"

    echo "===== SGLANG  ${model_tag}  bs=${bs}  =====" | tee "$log"

    # bench_one_batch is SGLang's latency benchmark: runs one batch prefill +
    # output decoding, prints "Total. latency: X.XXX s" (parallel to
    # `vllm bench latency` which prints "Avg latency: X.XX seconds").
    if ! python -m sglang.bench_one_batch \
            --model-path "$hfid" \
            --batch-size "$bs" \
            --input "$INPUT_LEN" \
            --output "$OUTPUT_LEN" \
            --dtype bfloat16 \
            >>"$log" 2>&1; then
        echo "FAILED: sglang ${model_tag} bs=${bs}" >&2
        return 1
    fi

    # Parse "Total. latency: <s> s" (or "total latency" depending on version).
    local total_s
    total_s=$(grep -iE 'total\.? latency: *[0-9]+\.[0-9]+' "$log" | tail -1 \
              | grep -oE '[0-9]+\.[0-9]+' | head -1)
    if [[ -z "${total_s:-}" ]]; then
        # Fallback: sum prefill + decode latency lines if present.
        local prefill decode
        prefill=$(grep -iE 'prefill\.? latency: *[0-9]+\.[0-9]+' "$log" | tail -1 \
                  | grep -oE '[0-9]+\.[0-9]+' | head -1)
        decode=$(grep -iE 'decode\.? latency: *[0-9]+\.[0-9]+' "$log" | tail -1 \
                 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        if [[ -n "${prefill:-}" && -n "${decode:-}" ]]; then
            total_s=$(python3 -c "print($prefill + $decode)")
        fi
    fi
    if [[ -z "${total_s:-}" ]]; then
        echo "Could not parse latency from $log" >&2
        return 1
    fi

    python3 - <<EOF >"$json"
import json
total_s = $total_s
input_len = $INPUT_LEN
output_len = $OUTPUT_LEN
total_tokens = (input_len + output_len) * $bs
ms_per_token = (total_s * 1000.0) / total_tokens
print(json.dumps({
    "system": "sglang",
    "gpu": "H100",
    "model": "$hfid",
    "batch_size": $bs,
    "input_len": input_len,
    "output_len": output_len,
    "total_iter_seconds": total_s,
    "latency_ms_per_token": ms_per_token,
}, indent=2))
EOF
}

for model_tag in $MODELS; do
    [[ -z "${HF_ID[$model_tag]:-}" ]] && { echo "Unknown model tag: $model_tag" >&2; exit 1; }
    for bs in $BATCH_SIZES; do
        run_cell "$model_tag" "$bs" || true
    done
done

echo "SGLang sweep done. Per-cell JSONs in $OUTPUT_ROOT"
