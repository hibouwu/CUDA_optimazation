"""Development-only 16 KiB TMA input schedules; all times are CTA clock64 cycles.

The service and completion-tail split is an abstract candidate, not an assignment
of latency to HBM/L2/TMA. Credits return at candidate completion, never at slot
release. No measured ready/elapsed values enter the simulator.
"""
from dataclasses import dataclass
from collections import deque
import math


@dataclass(frozen=True)
class Costs:
    service: float
    completion_tail: float
    issue: float = 0.0
    control: float = 0.0
    ffma: float = 0.0

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0 for x in vars(self).values()):
            raise ValueError('costs must be finite and nonnegative')


@dataclass(frozen=True)
class GroupCosts:
    """Optional explicit software costs; caller supplies evidence/assumptions."""
    before_wait: float = 0.0
    after_issue: float = 0.0

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0 for x in vars(self).values()):
            raise ValueError('group costs must be finite and nonnegative')


def simulate(schedule, items, stages, requests, costs, credits=None,
             consumer=0, trace=False, group_costs=None):
    """Return modeled window and optional request/group event records.

    s14: issue -> wait -> CTA publication, repeated.
    s16: retire oldest if slot occupied -> issue R -> CTA issue publication;
         drain all remaining groups at end. control is split equally between
         issue publication and retirement (a declared simplification).
    r05: prefill up to S before each oldest consumer; drain every eight items.
         control aggregates wait bookkeeping, loads, and the two CTA barriers.
    Initial/final timer/sync overhead is omitted, not inferred as zero hardware
    cost. items for r05 must be a multiple of eight.
    """
    if schedule not in ('s14', 's16', 'r05'):
        raise ValueError('unknown schedule')
    if any(type(x) is not int or x < 1 for x in (items, stages, requests)):
        raise ValueError('positive integer schedule dimensions required')
    if credits is not None and (type(credits) is not int or credits < 1):
        raise ValueError('credits must be a positive integer or None')
    if schedule == 's14' and (stages != 1 or requests != 1):
        raise ValueError('s14 has one slot and one request')
    if schedule == 'r05' and (items % 8 or requests != 1 or stages not in (1, 2, 4)):
        raise ValueError('r05 requires eight-item batches, R=1, S=1/2/4')
    if type(consumer) is not int or consumer < 0:
        raise ValueError('consumer must be a nonnegative integer')
    group_costs = group_costs or GroupCosts()
    now = 0.0
    service_end = 0.0
    active = deque()
    groups = {}
    slots = {}
    events = []
    maximum_active = 0

    def issue(group):
        nonlocal now, service_end, maximum_active
        slot = group % stages
        assert slot not in slots, 'software overwrote an unreleased slot'
        slots[slot] = group
        first = now
        ready = now
        for request in range(requests):
            eligible = now
            while active and active[0] <= now:
                active.popleft()
            if credits is not None and len(active) >= credits:
                now = max(now, active[0])
                while active and active[0] <= now:
                    active.popleft()
            admitted = now
            start = max(admitted, service_end)
            service_end = start + costs.service
            ready = service_end + costs.completion_tail
            active.append(ready)
            maximum_active = max(maximum_active, len(active))
            if trace:
                events.append(dict(kind='request', group=group, slot=slot,
                                   request=request, eligible=eligible, admit=admitted,
                                   service_start=start, service_end=service_end,
                                   ready=ready, active_after_admit=len(active)))
            now += costs.issue
        now += group_costs.after_issue
        if schedule == 's16':
            now += costs.control / 2
        groups[group] = (ready, first, now)

    def retire(group):
        nonlocal now
        ready, first, issue_return = groups.pop(group)
        now += group_costs.before_wait
        wait_begin = now
        wait_return = max(now, ready)
        control = costs.control / 2 if schedule == 's16' else costs.control
        now = wait_return + control + consumer * costs.ffma
        del slots[group % stages]
        if trace:
            events.append(dict(kind='group', group=group, slot=group % stages,
                               issue_begin=first, issue_return=issue_return,
                               ready=ready, wait_begin=wait_begin,
                               wait_return=wait_return, slot_release=now))

    if schedule == 'r05':
        for base in range(0, items, 8):
            issued = base
            for group in range(base, base + 8):
                while issued < min(base + 8, group + stages):
                    issue(issued)
                    issued += 1
                retire(group)
    else:
        for group in range(items):
            if group >= stages:
                retire(group - stages)
            issue(group)
        for group in range(max(0, items - stages), items):
            retire(group)
    return dict(cycles=now, requests=items * requests,
                cycles_per_request=now / (items * requests),
                max_inflight=maximum_active, events=events)
