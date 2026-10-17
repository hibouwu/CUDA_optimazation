"""cfg_b 启动时的逐 warp 寄存器配额与 CTA 保留池。

计数单位为 32-bit 寄存器；本模块没有时间价格、SM 剩余容量借用或仲裁规则。
setmaxnreg 作用于 warp。同一 warpgroup 的四个 warp 最终必须执行同一指令，
但这不使四次申请成为原子操作；调用顺序只表示调用方给出的事件顺序。
"""


THREADS_PER_WARP = 32
INITIAL_REGISTERS_PER_THREAD = 168
WARP_COUNT = 12
POOL_REGISTERS = THREADS_PER_WARP * WARP_COUNT * INITIAL_REGISTERS_PER_THREAD


def _warp(warp):
    if type(warp) is not int or not 0 <= warp < WARP_COUNT:
        raise ValueError('cfg_b warp must be an integer in 0..11')
    return warp


def _role(warp):
    return 'producer' if warp < 4 else f'consumer{warp // 4 - 1}'


def _target(warp):
    return 40 if warp < 4 else 232


def cfg_b_quota_requirements():
    """固定构建的一次启动需求；不把配额上限当作动态活跃值数量。"""
    warps = []
    for warp in range(WARP_COUNT):
        target = _target(warp)
        delta = THREADS_PER_WARP * (target - INITIAL_REGISTERS_PER_THREAD)
        warps.append(dict(warp=warp, warpgroup=warp // 4, role=_role(warp),
            threads=THREADS_PER_WARP, initial_registers_per_thread=INITIAL_REGISTERS_PER_THREAD,
            target_registers_per_thread=target, operation='dec' if delta < 0 else 'inc',
            released_registers=max(0, -delta), needed_registers=max(0, delta)))
    return dict(schema='gh200-cfg-b-register-demands/1', threads=384, warps=warps,
        initial_pool_registers=POOL_REGISTERS, steady_allocated_registers=POOL_REGISTERS,
        allocation_scope='CTA-reserved pool; no borrowing from remaining SM capacity',
        instruction_scope='warp; all four warps of each warpgroup execute the same instruction',
        lifecycle='once per kernel startup, retained across outputs and Ktiles',
        unit='32-bit register quota counts')


class CfgBRegisterQuota:
    """12 个 warp 共享 64512 个寄存器的配额账本。

    try_set 返回本次是否已满足；失败不更新配额、可用量或成功事件历史。
    已满足的重复调用是无变化的确认，不再次申请。部分完成状态用于描述
    startup 中间事件，不表示可以省略同 warpgroup 其他 warp 的指令。
    """

    qualification = 'cfg_b warp-level CTA quota accounting; no timing or physical RF arbitration'

    def __init__(self, cta=0):
        if type(cta) is not int or cta < 0:
            raise ValueError('cta must be a nonnegative integer')
        self.cta = cta
        self._registers = [INITIAL_REGISTERS_PER_THREAD] * WARP_COUNT
        self._events = []

    @property
    def allocated_registers(self):
        return THREADS_PER_WARP * sum(self._registers)

    @property
    def available_registers(self):
        return POOL_REGISTERS - self.allocated_registers

    @property
    def startup_complete(self):
        return all(value == _target(warp) for warp, value in enumerate(self._registers))

    def try_set(self, warp, target):
        """尝试完成一个 warp 的固定 cfg_b dec/inc；不替调用方选择下个 warp。"""
        warp = _warp(warp)
        if type(target) is not int or target != _target(warp):
            raise ValueError('target must match the fixed cfg_b role: producer 40, consumer 232')
        previous = self._registers[warp]
        delta = THREADS_PER_WARP * (target - previous)
        available = self.available_registers
        satisfied = delta <= available
        unchanged = previous == target
        event = dict(cta=self.cta, warp=warp, warpgroup=warp // 4, role=_role(warp),
            operation='dec' if warp < 4 else 'inc',
            registers_per_thread_before=previous,
            registers_per_thread_after=target if satisfied else previous,
            needed_registers=max(0, delta), released_registers=max(0, -delta),
            available_registers_before=available,
            available_registers=available - delta if satisfied else available,
            success=satisfied, already_satisfied=unchanged, sequence=None)
        if satisfied and not unchanged:
            self._registers[warp] = target
            event['sequence'] = len(self._events)
            self._events.append(dict(event))
        return event

    def release_producer(self, warp):
        """只释放指定 producer warp；没有 warpgroup 原子释放。"""
        warp = _warp(warp)
        if warp >= 4:
            raise ValueError('producer warp must be in 0..3')
        return self.try_set(warp, 40)

    def acquire_consumer(self, warp):
        """只申请指定 consumer warp，合法编号为 4..11。"""
        warp = _warp(warp)
        if warp < 4:
            raise ValueError('consumer warp must be in 4..11')
        return self.try_set(warp, 232)

    def snapshot(self):
        warps = [dict(warp=warp, warpgroup=warp // 4, role=_role(warp),
                      registers_per_thread=value, target_registers_per_thread=_target(warp))
                 for warp, value in enumerate(self._registers)]
        groups = []
        for group in range(3):
            members = range(4*group, 4*group+4)
            groups.append(dict(warpgroup=group, role=_role(4*group),
                instruction_complete=all(self._registers[warp] == _target(warp) for warp in members)))
        return dict(schema='gh200-cfg-b-register-quota/1', cta=self.cta,
            pool_registers=POOL_REGISTERS, allocated_registers=self.allocated_registers,
            available_registers=self.available_registers, warps=warps, warpgroups=groups,
            startup_complete=self.startup_complete, events=[dict(event) for event in self._events],
            qualification=self.qualification)
