"""请求级候选的模型活动窗；保留复合服务与在途窗口的不同边界。"""


def union_intervals(intervals):
    merged = []
    for begin, end in sorted((a,b) for a,b in intervals if b > a):
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1],end)
        else:
            merged.append([begin,end])
    return merged


def overlap_ns(left, right):
    a, b = union_intervals(left), union_intervals(right)
    i = j = 0
    total = 0.
    while i < len(a) and j < len(b):
        total += max(0.,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def activity_windows(actor, deliveries, *, prefix_ns=0., detailed=False):
    """只读同版已求解结果；compute service不含排队，input pending含源和目的等待。"""
    compute = []
    for index, end in enumerate(actor['service_frontier']):
        begin = max(actor['starts'][index],actor['service_frontier'][index-1] if index else 0.)
        compute.append([begin+prefix_ns,end+prefix_ns])
    output = {kind: [[a+prefix_ns,b+prefix_ns] for block in actor['outputs']
                    for a,b in zip(block[kind+'_begin'],block[kind+'_end'])]
              for kind in ('prepare','write')}
    pending = [[d['arrival_ns']+prefix_ns,d['ready_ns']+prefix_ns] for d in deliveries]
    events = sorted([(a,1) for a,b in pending if b>a]+[(b,-1) for a,b in pending if b>a])
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak,active)
    summary = dict(composite_compute_service_ns=sum(b-a for a,b in compute),
        epilogue_prepare_ns=sum(b-a for a,b in output['prepare']),
        epilogue_write_ns=sum(b-a for a,b in output['write']),
        compute_prepare_overlap_ns=overlap_ns(compute,output['prepare']),
        compute_write_overlap_ns=overlap_ns(compute,output['write']),
        input_pending_peak=peak,
        qualification='model windows under existing service assumptions')
    if detailed:
        summary['windows_ns'] = dict(composite_compute_service=compute,
            epilogue_prepare=output['prepare'],epilogue_write=output['write'],input_pending=pending)
    return summary
