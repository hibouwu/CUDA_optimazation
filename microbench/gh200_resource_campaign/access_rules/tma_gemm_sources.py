"""把 cfg_b 的源坐标接到 tensor 探针程序，比较源订阅范围。"""
import tma_tensor_multicta as tensor
from tma_gpc_service import GroupedTensorSource


CANDIDATES = ('unmerged', 'gpc_ready', 'global_ready', 'gpc_service_end')


class GemmSource:
    version = 'cfg-b-source-subscription/1'

    def __init__(self, coordinates, cta_groups, candidate, prices):
        self.coordinates = coordinates
        self.cta_groups = cta_groups
        self.candidate = candidate
        self.core = GroupedTensorSource(cta_groups, 2, prices[0], prices[1])
        self.pending = {}
        self.count = 0
        self.work = 0.
        self.busy = 0.

    def submit(self, t, cta, key, weight, partition_demand=None):
        _, k, atom = key
        m, n = self.coordinates[cta]
        address = ('A', m, k) if atom == 'A' else ('B', n*2+(atom == 'B1'), k)
        owner = self.cta_groups[cta] if self.candidate.startswith('gpc_') else 0
        identity = (owner, address)
        previous = self.pending.get(identity)
        boundary = 'service_end' if self.candidate == 'gpc_service_end' else 'source_ready'
        if self.candidate != 'unmerged' and previous is not None and t < previous[boundary]:
            return dict(previous, merged=True, service_start=None, service_end=None, source_key=address)
        served = self.core.submit(t, cta, address, weight)
        self.pending[identity] = served
        self.count, self.work, self.busy = self.core.count, self.core.work, self.core.busy
        return dict(served, source_key=address)


def simulate(coordinates, cta_groups, candidate, prices, control, consumer, *, active, skew_ns, groups=64, ring=64):
    p = tensor.program(len(coordinates), active, 'independent', groups=groups, source_ring=ring)
    p['source_protocol_description'] = 'cfg_b coordinates supplied to GemmSource'
    p['initial_offsets_ns'] = [((cta*skew_ns+31)//32)*32 for cta in range(len(coordinates))]
    p['source_union_bytes'] = (len({m for m,n in coordinates})+len({n for m,n in coordinates}))*ring*16384
    service = GemmSource(coordinates, cta_groups, candidate, prices)
    return tensor.simulate(p, 'shared_source_fifo', source_service=service,
        service=prices[0], latency=prices[1], destination=prices[2], issue=0.,
        control=control, consumer=consumer, trace=True)
