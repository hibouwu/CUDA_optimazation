"""固定合同的完整 SM 联合窗口；不拆内部端点，不维护仲裁队列。"""
from copy import deepcopy
import math


def joint_service(contract, ready_times_ns, frequency_ghz):
    """同一 SM clock64 的 cycles / GHz = ns；合同覆盖的旧服务须由调用方替换。"""
    fields = ('pair', 'org', 'work', 'roles', 'consumers', 'barriers', 'sass',
              'identity', 'covers', 'boundary', 'qualification')
    if any(not contract.get(field) for field in fields):
        raise ValueError('联合窗口需要完整工作、角色、消费者、同步、SASS及身份合同')
    cycles = contract.get('cycles')
    if (type(cycles) not in (int, float) or not math.isfinite(cycles) or cycles <= 0
            or type(frequency_ghz) not in (int, float) or not math.isfinite(frequency_ghz)
            or frequency_ghz <= 0):
        raise ValueError('完整窗口cycle与同SM频率必须有限且为正')
    ready = list(ready_times_ns)
    if not ready or any(type(value) not in (int, float) or not math.isfinite(value)
                        or value < 0 for value in ready):
        raise ValueError('需要非空、有限且非负的外部前驱时刻')
    begin = max(ready)
    return dict(begin_ns=begin, end_ns=begin+cycles/frequency_ghz,
        cycles=cycles, frequency_ghz=frequency_ghz, pair=contract['pair'], org=contract['org'],
        covers=deepcopy(contract['covers']), boundary=deepcopy(contract['boundary']),
        qualification=deepcopy(contract['qualification']), identity=deepcopy(contract['identity']))
