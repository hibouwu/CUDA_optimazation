"""cfg_b 输入槽的 mbarrier 生命周期与显式条件返回假设。

计数状态不为硬件操作定价。full_wait 的价格由调用方给出，既有计算
递推仍拥有 wait1 / wait0 后的 release 时刻；本模块只核对该代释放。
"""
import math
from numbers import Real

from supply_model import input_slot_state


def _time(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} requires a finite nonnegative ns value')
    return value


def _count(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f'{name} requires a {"nonnegative" if zero else "positive"} integer')
    return value


class BarrierPhase:
    """一个可重复使用的 barrier；phase 仅在两种待完成计数均为零时翻转。

    tx 计数可暂时为负，因 complete_tx 可以早于 expect_tx。arrive_and_expect_tx
    在同一操作中先登记事务再贡献到达，避免把中间状态误当 phase 完成。
    wait_complete 接收等待者保存的 phase，返回其是否已被翻转；绝对代次
    由 completed_phases 保留，不能只用一位 phase 检查槽复用。
    """

    def __init__(self, expected_arrivals, phase=0):
        self.expected_arrivals = _count(expected_arrivals, 'expected_arrivals')
        if type(phase) is not int or phase not in (0, 1):
            raise ValueError('phase must be 0 or 1')
        self.phase = phase
        self.pending_arrivals = expected_arrivals
        self.pending_tx_bytes = 0
        self.completed_phases = 0

    def _finish(self):
        if self.pending_arrivals == 0 and self.pending_tx_bytes == 0:
            self.phase ^= 1
            self.completed_phases += 1
            self.pending_arrivals = self.expected_arrivals
            return True
        return False

    def arrive(self, count=1):
        count = _count(count, 'arrival count')
        if count > self.pending_arrivals:
            raise ValueError('arrival count exceeds the current phase requirement')
        self.pending_arrivals -= count
        return self._finish()

    def expect_tx(self, transaction_bytes):
        self.pending_tx_bytes += _count(transaction_bytes, 'transaction_bytes', zero=True)
        return self._finish()

    def complete_tx(self, transaction_bytes):
        self.pending_tx_bytes -= _count(transaction_bytes, 'transaction_bytes', zero=True)
        return self._finish()

    def arrive_and_expect_tx(self, transaction_bytes, count=1):
        transaction_bytes = _count(transaction_bytes, 'transaction_bytes', zero=True)
        count = _count(count, 'arrival count')
        if count > self.pending_arrivals:
            raise ValueError('arrival count exceeds the current phase requirement')
        self.pending_tx_bytes += transaction_bytes
        self.pending_arrivals -= count
        return self._finish()

    def wait_complete(self, wait_phase):
        if type(wait_phase) is not int or wait_phase not in (0, 1):
            raise ValueError('wait_phase must be 0 or 1')
        return self.phase != wait_phase

    def snapshot(self):
        return dict(expected_arrivals=self.expected_arrivals, phase=self.phase,
                    pending_arrivals=self.pending_arrivals,
                    pending_tx_bytes=self.pending_tx_bytes,
                    completed_phases=self.completed_phases)


class ConditionSyncService:
    """cfg_b cluster1x1 的 full/empty 状态与条件等待返回工作假设。

    每个 CTA/slot 持续维护 full(1 次到达，32768 B) 和 empty(128 次到达)。
    full_wait 绑定既有目的服务给出的 ready_ns，相当于该代事务已完成；
    它不生成新的 TMA 延迟。ready/blocked 两条返回价格必须显式指定。
    release 批量贡献 128 个逻辑到达，额外耗时仍为零，不推断机器指令数。
    """

    qualification = '工作假设：显式条件返回价格；未由真实 cfg_b 计算/TMA 短链校准'

    def __init__(self, stages, fast_return_ns, blocked_return_ns):
        self.stages = _count(stages, 'stages')
        self.fast_return_ns = _time(fast_return_ns, 'fast_return_ns')
        self.blocked_return_ns = _time(blocked_return_ns, 'blocked_return_ns')
        self._slots = {}

    def _slot(self, cta, slot):
        _count(cta, 'cta', zero=True)
        key = (cta, slot)
        if key not in self._slots:
            self._slots[key] = dict(generation=0, group=None, wait_return_ns=None,
                                    released_ns=None, full=BarrierPhase(1),
                                    empty=BarrierPhase(128))
        return self._slots[key]

    def full_wait(self, enter_ns, ready_ns, cta, group):
        enter_ns = _time(enter_ns, 'enter_ns')
        ready_ns = _time(ready_ns, 'ready_ns')
        _count(group, 'group', zero=True)
        state = input_slot_state(group, self.stages)
        slot = self._slot(cta, state['slot'])
        if slot['generation'] != state['generation'] or slot['group'] is not None:
            raise ValueError('full wait does not match the next unreleased slot generation')
        if slot['released_ns'] is not None and ready_ns < slot['released_ns']:
            raise ValueError('new input readiness precedes the prior slot release')
        if not slot['empty'].wait_complete(state['empty_wait_phase']):
            raise ValueError('producer empty wait is not satisfied for this generation')
        if slot['full'].phase != state['full_wait_phase']:
            raise ValueError('full barrier phase differs from the persistent slot state')
        slot['full'].arrive_and_expect_tx(32768)
        slot['full'].complete_tx(32768)
        if not slot['full'].wait_complete(state['full_wait_phase']):
            raise RuntimeError('full arrival and transaction completion did not satisfy the wait')
        path = 'ready' if enter_ns >= ready_ns else 'blocked'
        price = self.fast_return_ns if path == 'ready' else self.blocked_return_ns
        eligibility = max(enter_ns, ready_ns)
        returned = eligibility + price
        slot['group'] = group
        slot['wait_return_ns'] = returned
        return dict(state, cta=cta, enter_ns=enter_ns, ready_ns=ready_ns,
                    eligibility_ns=eligibility, path=path, return_cost_ns=price,
                    wait_return_ns=returned, qualification=self.qualification)

    def release(self, t, cta, group):
        t = _time(t, 'release time')
        _count(group, 'group', zero=True)
        state = input_slot_state(group, self.stages)
        slot = self._slot(cta, state['slot'])
        if slot['group'] != group or slot['generation'] != state['generation']:
            raise ValueError('release does not match the consumed slot generation')
        if t < slot['wait_return_ns']:
            raise ValueError('slot release precedes full wait return')
        if slot['empty'].phase != state['full_wait_phase']:
            raise ValueError('empty barrier phase differs from the released generation')
        slot['empty'].arrive(128)
        slot['generation'] += 1
        slot['group'] = None
        slot['released_ns'] = t
        return t

    def slot_snapshot(self, cta, slot):
        _count(slot, 'slot', zero=True)
        if slot >= self.stages:
            raise ValueError('slot is outside the input ring')
        record = self._slot(cta, slot)
        return dict(generation=record['generation'], group=record['group'],
                    wait_return_ns=record['wait_return_ns'], released_ns=record['released_ns'],
                    full=record['full'].snapshot(), empty=record['empty'].snapshot())
