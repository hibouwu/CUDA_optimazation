#!/usr/bin/env bash
# 在已部署源码包上一次提交剩余实验；现有采样器及审查器保持原含义。
set -euo pipefail

usage() {
    cat <<'HELP'
用法：
  bash run_s15_once.sh --submit /ABS/DEPLOYMENT/repo [remaining|all]
  bash run_s15_once.sh --gpu /ABS/DEPLOYMENT/repo [remaining|all]
  bash run_s15_once.sh --analyse /ABS/DEPLOYMENT/repo [remaining|all]

--submit 在 ROMEO 一次提交一个 short/3h/单 GPU 作业及依赖它的 CPU 审查作业。
--gpu 仅在有效单 GPU 分配中执行；默认 remaining，连续覆盖原剩余12配置。
--analyse 使用包自身普通审查器，输出逐组表、真实样本及参数候选，不授 C。
all 用于新实验重现原68配置，不用于当前56点的接续。所有正式重复、预热、
数值与配额检查由原采样器执行；完整容量不足或未知状态将保留检查点并停止。
提交前要求扣除原预留后的持久余量至少6GiB；这是运营预算，压缩量仍逐点核验。
图由取回归档后的 report_tma_tensor_2d.py 生成；它不需要再次执行 GPU。
HELP
}

if [[ ${1:-} == --help || $# == 0 ]]; then usage; exit 0; fi
action=$1
package_arg=${2:?需要完整已签 formal-sampling 源码包的绝对路径}
scope=${3:-remaining}
[[ $# -le 3 && ( $scope == remaining || $scope == all ) ]] || { usage >&2; exit 2; }
[[ $action == --submit || $action == --gpu || $action == --analyse ]] || { usage >&2; exit 2; }
[[ $package_arg == /* ]] || { echo '源码包路径必须是绝对路径' >&2; exit 2; }
[[ -d $package_arg && ! -L $package_arg ]] || { echo '源码包目录不存在或是符号链接' >&2; exit 2; }
package=$(realpath -e -- "$package_arg")
script_path=$(realpath -e -- "${BASH_SOURCE[0]}")
campaign_dir=$(dirname -- "$(dirname -- "$script_path")")
report_script="$campaign_dir/report_tma_tensor_2d.py"
for name in run_formal.py audit_formal.py formal-source-manifest.json reviews/formal-source-review.json; do
    [[ -f $package/$name && ! -L $package/$name ]] || { echo "缺少完整源码包文件：$name" >&2; exit 2; }
done
[[ -f $report_script ]] || { echo '缺少 report_tma_tensor_2d.py' >&2; exit 2; }

cohorts=(padding_through16kib padding32kib padding64kib_gmem_to_smem padding64kib_smem_to_gmem)
if [[ $scope == all ]]; then cohorts=(oneCTA_and_nonpadding "${cohorts[@]}"); fi
deployment=$(dirname -- "$package")
export PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE=0

check_owner_gpu_queue() {
    python3 -B - "$1" <<'PY'
import json, os, pwd, subprocess, sys
user = pwd.getpwuid(os.geteuid()).pw_name
r = subprocess.run(['squeue', '-u', user, '-h', '-o', '%i|%T|%b|%j|%N'], capture_output=True, text=True, timeout=30)
assert r.returncode == 0, '无法核owner queue；不提交或执行GPU'
gpu_jobs = [line for line in r.stdout.splitlines() if 'gpu' in line.split('|')[2].lower()]
allowed = sys.argv[1]
assert all(line.split('|')[0] == allowed for line in gpu_jobs), 'owner已有其它GPU作业；保持串行，不自动重提'
if allowed:
    assert len(gpu_jobs) == 1 and gpu_jobs[0].split('|')[0] == allowed, '实际GPU分配身份不在owner queue'
print(json.dumps({'owner': user, 'allowed_job': allowed or None, 'queue': r.stdout, 'GPU_serial_guard': 'pass'}))
PY
}

if [[ $action == --submit ]]; then
    command -v sbatch >/dev/null
    # 只读检查放在申请 GPU 之前；规则直接来自已签包，不复制另一套门禁。
    python3 -B - "$package" <<'PY'
from pathlib import Path
import sys
package = Path(sys.argv[1])
sys.dont_write_bytecode = True
sys.path.insert(0, str(package / 'microbench/gh200_resource_campaign'))
sys.path.insert(0, str(package))
from common.suite_io import read_json, validate_gate, verify_files
from project_quota import current_project_quota
gate = validate_gate(package / 'reviews/formal-source-review.json', package, 'S15', 'formal-cohort-source-B')
assert gate['authorization']['formal_after_own_pilot_and_independent_full_values'] is True
assert gate['authorization']['formal_parameters_without_C'] is False
verify_files(package, read_json(package / 'formal-source-manifest.json'))
quota = current_project_quota(package.parent)
assert quota['headroom_after_reserve_bytes'] >= 6 * 1024**3, '完整实验运营预算不足6GiB；先准备归档容量，不提交GPU'
PY
    check_owner_gpu_queue ''
    workflow="$deployment/once-$scope"
    mkdir -- "$workflow" # 已有提交意图不能覆盖或盲目重提。
    mkdir -- "$workflow/runners"
    cp -- "$script_path" "$workflow/runners/run_s15_once.sh"
    cp -- "$report_script" "$workflow/report_tma_tensor_2d.py"
    sha256sum -- "$workflow/runners/run_s15_once.sh" "$workflow/report_tma_tensor_2d.py" > "$workflow/source.sha256"
    chmod 0444 -- "$workflow/runners/run_s15_once.sh" "$workflow/report_tma_tensor_2d.py"
    script_path="$workflow/runners/run_s15_once.sh"
    printf '%s\n' "$script_path" "$package" "$scope" > "$workflow/submission-intent.txt"
    sbatch --parsable --account=r260073 --partition=short --time=03:00:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --gpus-per-node=1 --constraint=armgpu --mem=32G \
        --chdir="$deployment" --output="$workflow/gpu-%j.out" --error="$workflow/gpu-%j.err" \
        "$script_path" --gpu "$package" "$scope" > "$workflow/gpu-submission.stdout" 2> "$workflow/gpu-submission.stderr"
    job_reply=$(cat -- "$workflow/gpu-submission.stdout")
    gpu_job=${job_reply%%;*}
    [[ $gpu_job =~ ^[0-9]+$ ]] || { echo 'GPU 提交结果未知，检查原 submission 文件后再处理' >&2; exit 2; }
    sbatch --parsable --account=r260073 --partition=short --time=03:00:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --constraint=armgpu --mem=8G \
        --dependency="afterok:$gpu_job" --kill-on-invalid-dep=yes --chdir="$deployment" \
        --output="$workflow/cpu-%j.out" --error="$workflow/cpu-%j.err" \
        "$script_path" --analyse "$package" "$scope" > "$workflow/cpu-submission.stdout" 2> "$workflow/cpu-submission.stderr"
    cpu_reply=$(cat -- "$workflow/cpu-submission.stdout")
    cpu_job=${cpu_reply%%;*}
    [[ $cpu_job =~ ^[0-9]+$ ]] || { echo 'CPU 提交结果未知；已提交 GPU 不得重新提交' >&2; exit 2; }
    printf 'GPU 作业：%s；结束后自动运行 CPU 审查作业：%s\n' "$gpu_job" "$cpu_job"
    exit 0
fi

: "${SLURM_JOB_ID:?必须在实际 Slurm 分配内执行}"
romeo_load_armgpu_env
spack load cuda@12.9.0
if [[ $action == --gpu ]]; then check_owner_gpu_queue "$SLURM_JOB_ID"; fi
cd -- "$package"
for cohort in "${cohorts[@]}"; do
    run_dir="$deployment/formal-$cohort"
    if [[ $action == --gpu ]]; then
        # 不在同一入口重做已封存组；原审查器检查数据完整性。
        if [[ -f $run_dir/summary.json && -f $run_dir/artifact-manifest.json ]]; then
            printf '已有采样组，留到 CPU 重算：%s\n' "$cohort"
        elif [[ -d $run_dir ]]; then
            python3 -B "$package/run_formal.py" --cohort "$cohort" --resume
        else
            python3 -B "$package/run_formal.py" --cohort "$cohort"
        fi
    else
        report_dir="$deployment/report-$cohort"
        [[ ! -e $report_dir ]] || { echo "已有报告，保留并检查：$report_dir" >&2; exit 2; }
        mkdir -- "$report_dir"
        python3 -B - "$report_dir" <<'PY'
from pathlib import Path
import hashlib, json, os, platform, sys
executable = Path(sys.executable).resolve()
record = {'job': os.environ['SLURM_JOB_ID'], 'hostname': platform.node(),
          'machine': platform.machine(), 'uid': os.geteuid(), 'python_version': sys.version,
          'python_executable': str(executable), 'python_executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
          'GPU_target_requested': False, 'ordinary_audit': True}
(Path(sys.argv[1]) / 'analysis-environment.json').write_text(json.dumps(record, indent=2) + '\n')
PY
        python3 -B "$package/audit_formal.py" --run "$run_dir" > "$report_dir/replay.json" 2> "$report_dir/replay.stderr"
        python3 -B "$report_script" --run "$run_dir" --replay "$report_dir/replay.json" \
            --output "$report_dir/tables" --tables-only
    fi
done
printf '%s 完成。正式参数仍须独立 C；已有有效样本保留。\n' "$action"
