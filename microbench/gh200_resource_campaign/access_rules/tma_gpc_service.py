"""每个声明共享组的有限并行服务候选；组到 GPC 的映射由证据文件给出。"""


class GroupedTensorSource:
    version = 'grouped-parallel-source/1'

    def __init__(self, cta_groups, capacity, service_ns, latency_ns):
        self.cta_groups = list(cta_groups)
        self.capacity = capacity
        self.price = service_ns
        self.latency = latency_ns
        self.frontiers = {group: [0.]*capacity for group in set(cta_groups)}
        self.count = 0
        self.work = 0.
        self.busy = 0.

    def submit(self, t, cta, key, weight, partition_demand=None):
        available = self.frontiers[self.cta_groups[cta]]
        slot = min(range(self.capacity), key=available.__getitem__)
        start = max(t, available[slot])
        end = start+self.price*weight
        available[slot] = end
        source_id = self.count
        self.count += 1
        self.work += weight
        self.busy += end-start
        return dict(source_id=source_id, merged=False, service_start=start,
            service_end=end, source_ready=end+self.latency)
