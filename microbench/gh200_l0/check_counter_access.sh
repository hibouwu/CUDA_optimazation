#!/usr/bin/env bash
# Read-only permission diagnosis within a Slurm GPU allocation.
set -u
printf '\n[identity]\n'
hostname
date -u
id
printf 'job=%s partition=%s CUDA_VISIBLE_DEVICES=%s\n' "${SLURM_JOB_ID:-none}" "${SLURM_JOB_PARTITION:-unknown}" "${CUDA_VISIBLE_DEVICES:-unset}"
printf '\n[driver]\n'
cat /proc/driver/nvidia/version
grep -Ei 'profil|restrict' /proc/driver/nvidia/params
printf '\n[process capabilities]\n'
grep -E '^Cap|^NoNewPrivs' /proc/self/status
printf '\n[existing sudo authorization]\n'
sudo -n -l 2>&1
printf '\n[device access]\n'
ls -ld /dev/nvidia0 /dev/nvidiactl /dev/nvidia-caps 2>&1
printf '\n[software]\n'
romeo_load_armgpu_env
spack load cuda@12.9.0
type -a ncu
ncu_path=$(command -v ncu)
readlink -f "$ncu_path"
getcap "$ncu_path" "$(readlink -f "$ncu_path")" 2>&1
ncu --version
printf '\n[gpu]\n'
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
printf '\n[existing probe via documented environment]\n'
timeout 20 ncu --clock-control none --cache-control none --metrics sm__inst_executed_pipe_tensor.sum "$HOME/gh200_l0/20260930-audit/audit" 6
result=$?
printf 'ncu_exit=%d\n' "$result"
exit 0
