"""阶段1固定工作集读窗口的两个候选；参数由对应开发批次提供。"""


FORMS = tuple((family, jsat) for family in ("max", "sum") for jsat in (1, 4, 8))


def predict_read_window(logical_bytes, active_sms, chains, service):
    """返回ns；b1/shared单位B/ns，a/floor单位ns。

    max使用局部供给与共享服务的较晚完成；sum使用可辨识的共同窗口项。
    固定B时sum的a和B/Bg合并为floor，不分别拟合两个常量。
    """
    if logical_bytes != service["logical_bytes"]:
        raise ValueError("该L2参数包只支持冻结的逻辑字节量")
    rate = service["b1_bytes_per_ns"] * min(chains, service["jsat"])
    local_ns = logical_bytes / (active_sms * rate)
    if service["family"] == "max":
        shared_ns = logical_bytes / service["shared_bytes_per_ns"]
        return service["a_ns"] + max(local_ns, shared_ns)
    if service["family"] == "sum":
        return service["floor_ns"] + local_ns
    raise ValueError(f"未知L2读窗口候选：{service['family']}")
