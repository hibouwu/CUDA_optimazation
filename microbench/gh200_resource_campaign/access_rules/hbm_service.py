"""GPU级共享外存的工作假设；方向服务率由调用方显式提供。"""


class SharedExternalMemory:
    """按可发出时间顺序接收请求；返回资源服务完成时间，单位B、ns。

    单FIFO串行支付读／写工作，尾部返回由调用方连接。
    read_bytes/write_bytes由缓存或写回策略提供。
    """

    version = 'shared-external-fifo/1'
    qualification = 'hypothesis: caller-supplied directional rates'

    def __init__(self, read_bytes_per_ns, write_bytes_per_ns):
        self.read_bytes_per_ns = read_bytes_per_ns
        self.write_bytes_per_ns = write_bytes_per_ns
        self.frontier_ns = 0.
        self.last_arrival_ns = float('-inf')
        self.read_bytes = self.write_bytes = self.count = 0
        self.busy_ns = 0.

    def submit(self, arrival_ns, *, read_bytes=0, write_bytes=0):
        if arrival_ns < self.last_arrival_ns:
            raise ValueError('shared external requests must arrive in event-time order')
        self.last_arrival_ns = arrival_ns
        start = max(arrival_ns, self.frontier_ns)
        service = read_bytes/self.read_bytes_per_ns + write_bytes/self.write_bytes_per_ns
        self.frontier_ns = start + service
        result = dict(request_id=self.count, arrival_ns=arrival_ns, service_start_ns=start,
            service_end_ns=self.frontier_ns, read_bytes=read_bytes, write_bytes=write_bytes)
        self.read_bytes += read_bytes
        self.write_bytes += write_bytes
        self.count += 1
        self.busy_ns += service
        return result
