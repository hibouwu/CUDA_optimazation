"""按 r18 probe 的默认规则解析一个 GEMM case 的有效输入，供模型与检查共用。

case 行只需写出与默认值不同的字段；这里补齐默认值、核对 probe 中可离线判断的值域与
组合约束，并在给出构建查询（static_setup 的 setup 事件）时先核对它与 case 属于同一程序，
再以 setup 的程序属性为准。依赖实验构建或设备的约束由 probe 在运行时检查，这里不判定。
字段含义见 Docs/ModelEvaluation/gemm/model/common/inputs.md。
"""

# probe Args 的默认值（probes/r18.cu）。0 表示“由其他字段推导”的项另见 DERIVED。
PROBE_DEFAULTS = dict(
    mode='warm', zero_m=-1, zero_n=-1, swizzle=1, evict=0, sm_count=0, input_mode='dyadic', seed=17,
    warmup_policy='adaptive', history_mode='off', history_source='original', mirror_target_slot=0,
    raster_order='heuristic', work_swap55_91=0, producer_delay_ns=0, first_supply_events=0, ready_observer=0,
    epilogue_defer=0, second_prefill=0, a_row_rotation=0, release_warp_observer=0,
    b_offset_bytes=0, b_allocation_extra_bytes=0)
# 写 0 或缺省时的推导次序与 probe 相同：先行距与存储，再分配行距。
DERIVED = (('lda', 'k'), ('ldb', 'n'), ('ldd', 'n'), ('storage_m', 'm'), ('storage_n', 'n'),
           ('input_map_m', 'm'), ('input_map_n', 'n'), ('alloc_lda', 'lda'), ('alloc_ldb', 'ldb'))
PROBLEM = ('m', 'n', 'k')
PROGRAM = ('config', 'stages')
BOOKKEEPING = ('id', 'set', 'kind', 'group', 'cond', 'candidate', 'original_experiment', 'purpose')
# 历史 case 中的注记：静态信息以构建查询为准，标签与采样安排属于任务协议，均不作为模型输入。
ANNOTATIONS = ('tile', 'cluster', 'schedule', 'grid_ctas', 'ktiles', 'scheduled_units', 'scheduler_status', 'original_m',
               'label', 'pitch', 'input_path', 'ready_generation_bound', 'sampling_order', 'variants')
CONFIG_STAGES = dict(cfg_a=6, cfg_a1=6, cfg_b=6, cfg_c=4, cfg_c1=4)
INTERVENTIONS = ('work_swap55_91', 'producer_delay_ns', 'first_supply_events', 'ready_observer', 'epilogue_defer',
                 'second_prefill', 'a_row_rotation', 'release_warp_observer')
# setup 事件中传给读取方的程序属性与身份。
SETUP_PROGRAM = ('grid', 'tile', 'cluster', 'epilogue_tile', 'stages', 'threads', 'smem', 'smem_epilogue',
                 'max_active_ctas_per_sm', 'scheduler_sm_count', 'gpu_uuid')
# setup 回显、须与 case 有效值一致的字段；setup 的 sm_count 是设备 SM 数，case 的 sm_count 对应 requested_sm_count。
SETUP_ECHO = dict(config='config', m='m', n='n', k='k', lda='lda', ldb='ldb', ldd='ldd', alloc_lda='alloc_lda',
                  alloc_ldb='alloc_ldb', storage_m='storage_m', storage_n='storage_n', zero_m='zero_m', zero_n='zero_n',
                  input_map_m='input_map_m', input_map_n='input_map_n', swizzle='swizzle', evict='evict',
                  input_mode='input_mode', seed='seed', requested_sm_count='sm_count', raster_order='raster_order',
                  b_offset_bytes='b_offset_bytes', b_allocation_extra_bytes='b_allocation_extra_bytes')
# 各读取方需要的有效输入；case 字段可由缺省补齐，setup 字段必须来自构建查询。
REQUIRED = {
    'spec_baseline': ('m', 'n', 'k'),
    'v09': ('config', 'm', 'n', 'k', 'lda', 'ldb', 'swizzle', 'raster_order', 'input_mode', 'zero_m', 'zero_n',
            'input_map_m', 'input_map_n', 'storage_n', 'b_offset_bytes', 'stages', 'gpu_uuid'),
    'c1_models': ('m', 'n', 'k', 'input_mode', 'config', 'stages'),
    'program_graph': ('config', 'm', 'n', 'k', 'swizzle', 'raster_order', 'stages'),
}
REQUIRED_SETUP = {'v09': ('grid', 'stages'), 'program_graph': ('grid', 'stages')}

CHOICES = dict(mode=('warm', 'setup', 'check-work-swap', 'check-producer-delay', 'check-first-supply',
                    'check-ready-observer', 'check-epilogue-order', 'check-second-prefill', 'check-a-row-rotation',
                    'check-release-warps'),
               producer_delay_ns=(0, 512), work_swap55_91=(0, 1), first_supply_events=(0, 1), ready_observer=(0, 1),
               epilogue_defer=(0, 1, 2), second_prefill=(0, 1), a_row_rotation=(0, 1), release_warp_observer=(0, 1),
               mirror_target_slot=(0, 1), swizzle=(1, 8), evict=(0, 1), input_mode=('dyadic', 'zero', 'random'),
               raster_order=('heuristic', 'along_m', 'along_n'), warmup_policy=('adaptive', 'common-sw1-16'),
               history_mode=('off', 'single', 'series'), history_source=('original', 'same', 'mirror'))


def resolve(row, setup=None, gpu_uuid=None):
    """返回 (有效输入, 错误列表)。错误表示记录本身不能解析或违反约束。"""
    eff, errors = dict(PROBE_DEFAULTS), []
    unknown = set(row) - set(PROBE_DEFAULTS) - {d for d, _ in DERIVED} - set(PROBLEM) - set(PROGRAM) - set(BOOKKEEPING) - set(ANNOTATIONS)
    if unknown:
        errors.append('undocumented fields: ' + ', '.join(sorted(unknown)))
    eff.update({key: value for key, value in row.items() if key not in ANNOTATIONS})
    for key in PROBLEM + ('config',):
        if key not in eff:
            errors.append('missing ' + key)
    if errors:
        return eff, errors
    if eff['config'] not in CONFIG_STAGES:
        errors.append('unknown config ' + str(eff['config']))
    for key, source in DERIVED:
        if not eff.get(key):
            eff[key] = eff[source]
    declared = row.get('stages')
    eff['stages'] = declared if declared is not None else CONFIG_STAGES.get(eff['config'])
    m, n, k = eff['m'], eff['n'], eff['k']
    checks = [
        (min(m, n, k) > 0, 'positive dimensions'),
        (eff['stages'] is None or (isinstance(eff['stages'], int) and eff['stages'] > 0), 'positive stage count'),
        (eff['lda'] >= k and eff['ldb'] >= n and eff['ldd'] >= n, 'ld >= logical columns'),
        (eff['storage_m'] >= m and eff['storage_n'] >= n and eff['ldb'] >= eff['storage_n'] and eff['ldd'] >= eff['storage_n'],
         'physical storage covers logical matrix and fits ld'),
        (m <= eff['input_map_m'] <= eff['storage_m'] and n <= eff['input_map_n'] <= eff['storage_n'], 'input map within storage'),
        (eff['alloc_lda'] >= eff['lda'] and eff['alloc_ldb'] >= eff['ldb'], 'allocation pitch >= physical pitch'),
        (eff['seed'] >= 0, 'nonnegative seed'),
        (eff['sm_count'] >= 0, 'nonnegative sm_count'),
        (eff['b_offset_bytes'] % 16 == 0 and eff['b_allocation_extra_bytes'] % 16 == 0
         and 0 <= eff['b_offset_bytes'] <= eff['b_allocation_extra_bytes'], 'B offset 16-byte aligned within extra allocation'),
        (not eff['ready_observer'] or eff['first_supply_events'], 'ready observer requires first supply endpoints'),
        (eff['history_mode'] == 'off' or eff['warmup_policy'] == 'common-sw1-16', 'history observation requires common-sw1-16'),
        (eff['history_source'] != 'original' or eff['mirror_target_slot'] == 0, 'mirror target slot requires paired sources'),
        (eff['history_source'] == 'original' or (eff['history_mode'] == 'series' and eff['swizzle'] == 1),
         'mirror source comparison uses the sw1 history series'),
    ]
    errors += ['violates ' + what for ok, what in checks if not ok]
    errors += [f'{key} must be one of {allowed}' for key, allowed in CHOICES.items() if eff[key] not in allowed]
    if setup is not None:
        mismatch = [f'{skey}={setup[skey]!r} vs case {ekey}={eff.get(ekey)!r}' for skey, ekey in SETUP_ECHO.items()
                    if skey in setup and setup[skey] != eff.get(ekey)]
        if 'stages' in setup and eff['stages'] is not None and setup['stages'] != eff['stages']:
            mismatch.append(f'stages={setup["stages"]} vs expected {eff["stages"]}')
        if mismatch:
            errors.append('setup belongs to a different program or case: ' + '; '.join(mismatch))
        else:
            for key in SETUP_PROGRAM:
                if key in setup:
                    eff[key] = setup[key]
            eff['setup'] = setup
    eff['gpu_uuid'] = eff.get('gpu_uuid') or gpu_uuid
    eff['interventions'] = sorted(key for key in INTERVENTIONS if eff[key])
    return eff, errors


def missing_for(eff, model):
    """某读取方仍缺的有效输入；setup 字段只认构建查询给出的值。"""
    missing = [key for key in REQUIRED[model] if eff.get(key) is None]
    missing += ['setup:' + key for key in REQUIRED_SETUP.get(model, ()) if eff.get('setup', {}).get(key) is None]
    return missing
