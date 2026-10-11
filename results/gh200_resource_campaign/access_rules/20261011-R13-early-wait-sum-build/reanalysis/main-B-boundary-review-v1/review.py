#!/usr/bin/env python3
"""Read-only source and index audit of the prepared R13 wait-sum observer."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/jianyeshi/Note/CUDA/CUDA_optimazation')
RESULTS = ROOT / 'results/gh200_resource_campaign/access_rules'
BASE = RESULTS / '20261010-R13-stage-progress'
CANDIDATE = RESULTS / '20261011-R13-early-wait-sum-build/attempts/prepare3'
OUTPUT = Path(__file__).resolve().parent
COLLECTIVE = 'source/overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp'
KERNEL = 'source/overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp'
PIPELINE = 'source/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp'
TRACE = 'source/probes/v06_trace.hpp'
PROBE = 'source/probes/r18.cu'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def site(path, text):
    lines = path.read_text().splitlines()
    matches = [index + 1 for index, line in enumerate(lines) if text in line]
    assert matches, (path, text)
    return dict(path=str(path), lines=matches, text=text)


def verify_manifest(directory):
    path = directory / 'source_hashes.json'
    hashes = json.loads(path.read_text())
    for name, expected in hashes.items():
        assert sha(directory / name) == expected, name
    return dict(path=str(path), sha256=sha(path), verified_source_files=len(hashes))


def main():
    manifests = [verify_manifest(BASE), verify_manifest(CANDIDATE)]
    assert manifests[0]['sha256'] == '2bc91a493dcbbbbfa7f08504d6864f4038025bc20bac3254624b6f2cf57fe6e2'
    original = (BASE / COLLECTIVE).read_text()
    candidate = (CANDIDATE / COLLECTIVE).read_text()
    inserted_flag = '    const bool r13_sum_enabled = r13_target && r13_wait_sum_observer;\n'
    inserted_before = '''      const bool r13_sum_selected = r13_sum_enabled && r13_k>=2 && r13_k<=16;
      uint64_t r13_wait_begin = 0;
      if (r13_sum_selected) r13_wait_begin = v06_clock();
'''
    inserted_after = '''      if (r13_sum_selected) {
        uint64_t r13_wait_end = v06_clock();
        r13_s->wait_sum.record(r13_k, r13_wait_begin, r13_wait_end);
      }
'''
    restored = candidate
    for insertion in [inserted_flag, inserted_before, inserted_after]:
        assert restored.count(insertion) == 1
        restored = restored.replace(insertion, '', 1)
    assert restored == original
    assert inserted_before + '''      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
''' + inserted_after in candidate
    assert 'static constexpr int K_PIPE_MMAS = 1;' in candidate
    assert candidate.count('r13_s->wait_sum.record(') == 1
    assert (CANDIDATE / KERNEL).read_bytes() == (BASE / KERNEL).read_bytes()
    assert (CANDIDATE / PIPELINE).read_bytes() == (BASE / PIPELINE).read_bytes()
    kernel = (CANDIDATE / KERNEL).read_text()
    lifetime = kernel.index('      R13SparseConsumer r13_s;')
    loop = kernel.index('      while (work_tile_info.is_valid()) {', lifetime)
    close = kernel.index('      } // Scheduler work fetch loop', loop)
    flush = kernel.index('r13_sparse_consumer_flush(r13_s)', close)
    assert lifetime < loop < close < flush
    assert 'mainloop_pipe_consumer_state.advance(k_tile_count);' in kernel
    trace = (CANDIDATE / TRACE).read_text()
    assert trace == (CANDIDATE / 'source/probes/r18_trace.hpp').read_text()
    assert (CANDIDATE / 'source/overlay/v06_trace.hpp').read_text() == '#pragma once\n#include "../probes/v06_trace.hpp"\n'
    assert 'b[31]=0x52313309;' in trace
    assert '#if !defined(R13_STAGE_PROGRESS) || !defined(R13_WAIT_SUM)' in trace
    assert 'asm volatile("mov.u64 %0, %%clock64;" : "=l"(c) :: "memory")' in trace
    probe = (CANDIDATE / PROBE).read_text()
    assert probe.index('cudaMemcpyToSymbol(r13_wait_sum_observer,&wait_sum_off') < probe.index('auto warm_sequence')
    prefix = probe.index('for (int i = 0; i < 16; ++i) warm.push_back(timed(warmup_launch_params));')
    enable = probe.index('cudaMemcpyToSymbol(r13_wait_sum_observer,&o.wait_sum_observer', prefix)
    target = probe.index('float ms = timed(launch_params);', enable)
    assert prefix < enable < target
    assert probe.count('cudaMemcpyToSymbol(r13_wait_sum_observer') == 2
    assert 'o.wait_sum_observer>1' in probe and 'o.warmup_policy!="common-sw1-16"' in probe

    rows = []
    summaries = []
    for stages in [4, 6]:
        # Consumer1 starts after Consumer0's 64 groups. Reconstruct by ++,
        # then check against the absolute group formula independently.
        slot, phase = 0, 0
        for _ in range(64):
            slot += 1
            if slot == stages:
                slot = 0
                phase ^= 1
        local_groups = [0]
        remaining = 63
        while remaining > 0:
            local_groups.append(64 - remaining)
            remaining -= 1
        assert local_groups == list(range(64))
        selected = []
        for local in local_groups:
            global_group = 64 + local
            assert slot == global_group % stages
            assert phase == (global_group // stages) % 2
            sampled = 2 <= local <= 16
            if sampled:
                selected.append(local)
            rows.append(dict(stages=stages, cta=0, thread=256, CTA_j=1, WG_local_j=0,
                local_group=local, global_group=global_group, read_slot=slot, read_phase=phase,
                prologue=int(local == 0), loop_k_tile_count='' if local == 0 else 64-local,
                sampled=int(sampled)))
            slot += 1
            if slot == stages:
                slot = 0
                phase ^= 1
        assert selected == list(range(2, 17))
        summaries.append(dict(stages=stages, total_groups=64, prologue_groups=[0],
            loop_groups=[1, 63], selected_groups=selected, count=len(selected),
            mask=sum(1 << group for group in selected), mask_hex=hex(sum(1 << group for group in selected)),
            selected_global_groups=[66, 80], C1_global_group=65, C16_global_group=80))
    with (OUTPUT / 'group-coverage.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sites = []
    for relative, fragments in [
        (COLLECTIVE, ['K_PIPE_MMAS = 1', 'const bool r13_target', 'const bool r13_sum_enabled',
            'prologue_mma_count = min', 'k_tile_count -= prologue_mma_count',
            'for ( ; k_tile_count > 0', 'r13_initial_k-k_tile_count',
            'if (r13_sum_selected) r13_wait_begin', 'consumer_try_wait(smem_pipe_read)',
            'consumer_wait(smem_pipe_read, barrier_token)', 'uint64_t r13_wait_end',
            'r13_s->wait_sum.record', 'r13_k==1)', 'r13_k==16) r13_s->commit16',
            '++smem_pipe_read;', '++smem_pipe_release;']),
        (KERNEL, ['mainloop_pipe_consumer_state.advance(k_tile_count);', 'R13SparseConsumer r13_s;',
            '++v06_tile_seq;', 'params.mainloop, v06_tile_seq, &r13_s',
            'mainloop_pipe_consumer_state.advance(k_tile_count * NumMmaWarpGroups)',
            '} // Scheduler work fetch loop', 'r13_sparse_consumer_flush(r13_s)']),
        (PIPELINE, ['PipelineState(): index_{}, phase_{}, count_{}',
            'ConsumerToken consumer_try_wait(uint32_t stage',
            'full_barrier_ptr_[stage].try_wait(phase)',
            'if (barrier_token == BarrierStatus::WaitAgain)']),
        (TRACE, ['asm volatile("mov.u64 %0, %%clock64;', 'R13WaitSum wait_sum;',
            'b[34]=s.wait_sum.cycles;', 'b[39]=s.wait_sum.invalid;']),
        (PROBE, ['unsigned wait_sum_off=0;', 'for (int i = 0; i < 16;',
            'cudaMemcpyToSymbol(r13_wait_sum_observer,&o.wait_sum_observer'])]:
        for fragment in fragments:
            sites.append(site(CANDIDATE / relative, fragment))
    reviewed = [COLLECTIVE, KERNEL, PIPELINE, TRACE, PROBE, 'source/overlay/v06_trace.hpp', 'source/probes/r13_wait_sum.cuh',
                'source/prepare_r13_wait_sum.py', 'contract.json', 'build/commands.json']
    decoder_path = ROOT / 'microbench/gh200_resource_campaign/access_rules/decode_r13_wait_sum.py'
    test_path = ROOT / 'microbench/gh200_resource_campaign/access_rules/test_r13_wait_sum.py'
    decoder = decoder_path.read_text()
    for fragment in ['b[35]!=15 or b[36]!=MASK or b[39]',
                     'b[20]<=b[37]<=b[38]<=b[17]', 'b[34]>b[38]-b[37]',
                     'outside=duration-wait', 'wait=outside=None',
                     'full_wait_ns=None,outside_wait_ns=None', 'if any(b[34:40])',
                     'b[40]!=flag', 'b[1]!=b[23]', 'b[13:15]!=[12,1]']:
        assert fragment in decoder, fragment
    report = dict(
        scope='Source placement, state/index coverage, lifetime, and prefix switch only; no GPU or SSH/Slurm by B.',
        candidate=str(CANDIDATE), manifests=manifests,
        input_sha256={**{str(CANDIDATE / name): sha(CANDIDATE / name) for name in reviewed},
                      str(decoder_path):sha(decoder_path), str(test_path):sha(test_path)},
        source_checks=dict(restored_collective_equals_base=True, unchanged_kernel_and_pipeline=True,
            H_before_same_group_try_and_wait=True, B_after_wait_before_record=True,
            no_next_group_prefetched_token_in_selected_source=True,
            collector_outlives_WG_scheduler_loop=True, deferred_flush_after_WG_scheduler_loop=True,
            selected_only_CTA0_thread256_WG_local_output0=True,
            prefix_16_calls_observer_off_then_one_target_flag=True,
            fixed_progress_protocol_excludes_other_trace_variants=True,
            canonical_trace_header_with_forwarding_overlay=True),
        decoder_interface_review=dict(path=str(decoder_path),reviewed=True,tests_repeated_by_B=False,
            count_mask_invalid_and_containment_enforced=True,
            Q_bound_is_stronger_than_Q_le_D=True,
            off_Q_and_residual_None=True,component_nanoseconds_None=True,
            raw_marker_same_SM_work_and_runtime_flag_checked=True,
            full_process_GPU_and_numerical_witness_checks_present=True,
            note='Source interface review only. Main owns test execution, acquisition binding, build and qualification.'),
        coverage=summaries, source_sites=sites,
        partition=dict(D='C16-C1', Q='sum(B_i-H_i), i=2..16',
            residual='sum((H_i-C_(i-1))+(C_i-B_i)), i=2..16',
            identity='D=Q+residual for ordered endpoints in one call/thread/output',
            first_after_C1='H2', last_before_C16='B16',
            interpretation='Q is the measured full try/wait path, including observer clocks and software. Residual includes collector work after B and other loop/issue/release work.'),
        decoder_requirements=[
            'Bind setup/call/raw marker to r13-early-wait-sum-v1 and the fixed shape/stages/output/thread identity.',
            'Observer on: count=15, mask=0x1fffc, invalid=0; same-SM entry/final identity; C1<=H2<=B16<=C16.',
            'Observer on: use integer cycle differences before aggregation; require 0<=Q<=C16-C1.',
            'Observer off: all new accumulated fields zero and flag zero; report Q and residual as unobserved/None, never Q=0.',
            'Compute per-call D, Q, D-Q before any median or stage comparison; components have no nanosecond allocation.',
            'Off/on controls share one binary per stage and 16 observer-off prefix calls with the same buffers; this does not equate stage4 and stage6 histories.'
        ],
        SASS_pending_checks=[
            'On every selected fast/slow path, the H clock precedes the actual full barrier try/test, and B follows successful wait/fast-path return.',
            'No token/barrier evaluation hoisted before H; no record arithmetic or MMA/fence/state update moved before B.',
            'No per-group global store; report register/spill/resource changes and added selected/unselected software paths.'
        ],
        minimal_fields=dict(cycles='uint64 sum', count='15', mask='0x1fffc', observer='explicit runtime 0/1 plus unique protocol'),
        candidate_extra_fields=['first_H2', 'last_B16', 'invalid_bits'],
        observer_scope_limits=[
            'The shared off/on binary measures the runtime flag increment. Its longer-lived collector/register allocation can differ from the old progress binary even with observer off.',
            'Post-B collector work is assigned to the recorded residual, but can alter readiness of subsequent groups and thereby their measured Q. This is an observed-path partition, not counterfactual uninstrumented wait time.'
        ],
        findings=[], source_boundary_review_passed=True,
        SASS_reviewed=False, GPU_executed_by_B=False, hardware_correctness=False,
        observation_qualified=False, service_identified=False, prediction_qualified=False)
    (OUTPUT / 'B-q-early-boundary-review.json').write_text(json.dumps(report, indent=2)+'\n')
    (OUTPUT / 'README.md').write_text('''本审查只核对固定 `prepare3` 的源码位置、循环覆盖、保存范围、前序开关及 cycle 解码接口。归档及候选的完整 source manifest 均逐项通过 SHA 核对，结果见 [JSON](B-q-early-boundary-review.json)。主线保留 prepare2 的双副本定义编译失败；prepare3 使用 canonical `probes/v06_trace.hpp`，overlay 只 forward。本次不重复主线的 5 项 CPU 测试。

`K_PIPE_MMAS=1`，K4096 对应 64 个 K64 组。prologue 消耗 local0，循环遍历 local1～63。thread256 属于 Consumer1，其首个自有输出（WG local j=0）是 CTA j=1，因此 `C1` 位于 global group65，`C16` 位于 group80；两者之间包含 local2～16（global66～80）的 15 次 full try/wait。stages4/6 的 slot/phase 逐项见 [覆盖表](group-coverage.csv)，count=15、mask=0x1fffc。

候选 H 放在当组 `consumer_try_wait` 前，B 放在 `consumer_wait` 返回后；token 在当前循环现场求值，没有提前为下一组生成 token。try 已返回 WaitDone 时，wait 跳过阻塞，仍须记录 H/B。新增 record 在 B 后，其成本属于 `D−Q`。本审查还删除候选的三个新增块并逐字比较，恢复后的 collective 与归档完全一致；kernel/pipeline 本身逐字一致。

`R13SparseConsumer r13_s` 在 WG scheduler while 外构造，整个 work loop 结束后才 flush。应保留这个范围，只有目标输出更新累计值，后续 WG 输出不重置或覆盖；循环内不写 GMEM。

同线程、同输出、同调用的时间顺序下，`D=C16−C1`，`Q=Σ(B_i−H_i)`，且 `D−Q=Σ[(H_i−C_(i−1))+(C_i−B_i)]`，i=2～16。先逐调用计算这些整数 cycle 差，再统计。Q 是取时后 full try/wait 路径的累计观测，包括软件与取时成本，不能直接当目的 ready、裸 TMA 服务或物理源价格。

最少新增 sum/count/mask/显式开关与唯一 protocol 身份；候选同时保留 H2、B16 和 invalid bits，便于拒绝坏记录。on 必须 count15、mask0x1fffc、invalid0、同 SM，且 `C1≤H2≤B16≤C16`、`0≤Q≤D`。off 字段为零表示未观测，Q 和 residual 应为 None。16 次前序调用始终 observer0，再设置目标 flag0/1；每个 stage 使用自身同一 binary/缓冲区，两个 stage 的执行历史不能据此视为相同。

当前 decoder 已核对上述字段、唯一 marker、SM/work/flag 身份和 on/off 分支；其 `Q≤B16−H2` 比 `Q≤D` 更强，且两个组件的 ns 均为 None。CPU helper 能证明记录算术与拒绝条件，不证明编译后的 GPU 计时顺序。完整采集身份和数值验收仍由主线负责。

同一 binary 的 off/on 只量化运行期开关的增量影响；新增长寿命累计寄存器可能使 off 也不同于旧 progress binary。record 虽在 B 后、归入残余，但会推迟后续等待到达，进而改变后续 Q。因此这是带观测路径的精确分段，不能直接恢复未加观测时的等待量。

源码边界审查通过；尚未核对编译后 SASS。需检查快/慢 token 路径上的 H→实际 barrier try/wait→B 顺序，以及寄存器、spill 和新增软件路径。B 本轮没有 GPU、SSH/Slurm、公共源码修改；CPU 静态通过不授予硬件正确性、观测或预测资格。
''')
    receipt = dict(report_sha256=sha(OUTPUT / 'B-q-early-boundary-review.json'),
        coverage_sha256=sha(OUTPUT / 'group-coverage.csv'), script_sha256=sha(Path(__file__)),
        manifest_source_files=[item['verified_source_files'] for item in manifests],
        coverage_rows=len(rows), findings=report['findings'])
    (OUTPUT / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
