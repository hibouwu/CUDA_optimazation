#!/usr/bin/env python3
"""Independent arithmetic/coverage audit for the memory-path campaign."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def check(ok, why):
    if not ok: raise ValueError(why)
def close(a,b,why):
    check(math.isfinite(float(a)) and math.isfinite(float(b)) and
          math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10),why)

def validate(row,case,device,trial):
    shared=case["mode"].startswith("smem")
    check(row["mode"]==case["mode"] and row["trial"]==trial,"case/trial identity")
    check(row["iterations"]==case["iterations"] and row["stride"]==case["stride"],"loop contract")
    check(row["errors"]==0 and row["cache_residency_proven"] is False,"correctness/residency")
    blocks=1 if shared else device["sms"]*4
    check(row["blocks"]==blocks and row["threads"]==256,"grid shape")
    check(row["requested_working_set_bytes"]==case["bytes"],"working-set request")
    check(row["seed"]==3+trial*19,"input seed")
    if shared:
        check(row["allocation_per_array_bytes"]==32768,"SMEM allocation")
        total=blocks*256*8*4*case["iterations"]
        rb=total if case["mode"]=="smem_read" else 0
        wb=total if case["mode"]=="smem_write" else 0
    else:
        per_cta=((case["bytes"]+blocks*256*16-1)//(blocks*256*16))*256
        alloc=per_cta*blocks*16
        check(row["allocation_per_array_bytes"]==alloc,"global allocation rounding")
        rb=alloc*case["iterations"] if case["mode"]!="global_write" else 0
        wb=alloc*case["iterations"] if case["mode"] in ("global_write","global_duplex") else 0
    check((row["read_payload_bytes"],row["write_payload_bytes"])==(rb,wb),"payload accounting")
    details=row["blocks_detail"];check(len(details)==blocks,"CTA records")
    ids={d["smid"] for d in details}
    check(len(ids)==row["observed_sms"],"observed SM field")
    check(len(ids)==(1 if shared else device["sms"]),"SM coverage")
    for d in details:
        check(d["stop_ns"]>d["start_ns"] and d["stop_cycle"]>d["start_cycle"],"invalid clocks")
    first=min(d["start_ns"] for d in details);last=max(d["stop_ns"] for d in details)
    check(row["start_ns"]==first and row["stop_ns"]==last,"grid makespan")
    check(row["max_cta_cycles"]==max(d["stop_cycle"]-d["start_cycle"] for d in details),"clock64 denominator")
    close((rb+wb)/(last-first),row["payload_gbytes_per_second"],"rate arithmetic")
    check((last-first)/1e6<=row["event_ms"]*1.05,"event cross-check")
    return (rb+wb)/row["max_cta_cycles"] if shared else (rb+wb)/(last-first)

def expected_matrix():
    rows = [(f"smem_read_stride{s}", "smem_read", 32768, 8192, s) for s in (1,2,4,8,16,32)]
    rows.append(("smem_write_stride1", "smem_write", 32768, 8192, 1))
    rows += [("global_read_ca_8m", "global_read_ca", 8*1024**2, 64, 1),
             ("global_read_cg_8m", "global_read_cg", 8*1024**2, 64, 1),
             ("global_read_cg_256m", "global_read_cg", 256*1024**2, 16, 1),
             ("global_write_256m", "global_write", 256*1024**2, 16, 1),
             ("global_duplex_128m", "global_duplex", 128*1024**2, 16, 1)]
    return set(rows)

def timed_loops(function):
    instructions=[]
    for line in function.splitlines():
        m=re.match(r"\s*/\*([0-9a-f]+)\*/(.*?) /\*",line)
        if m: instructions.append((int(m[1],16),m[2].strip()))
    timers=[addr for addr,op in instructions if "SR_GLOBALTIMERLO" in op]
    check(len(timers)==2,"two globaltimer boundaries required")
    loops=[]
    for addr,op in instructions:
        branch=re.search(r"\bBRA\s+0x([0-9a-f]+)",op)
        if branch:
            target=int(branch[1],16)
            if timers[0]<target<addr<timers[1]:
                loops.append((target,addr,[op2 for pc,op2 in instructions if target<=pc<=addr]))
    check(loops,"no timed backward branch")
    return loops

def check_memory_loops(function, tokens, exact_count=None):
    loops=timed_loops(function)
    for start,stop,body in loops:
        for token in tokens:
            count=sum(re.search(r"\b"+token+r"(?:\.|\s)",op) is not None for op in body)
            check(count>0,f"{token} absent in timed loop {start:x}..{stop:x}; possible hoist")
            if exact_count is not None:
                check(count==exact_count,f"{token} count in timed loop: {count}")

def check_completion(root,spec):
    summary_hash=sha(root/"summary.json")
    state=json.loads((root/"campaign_status.json").read_text())
    check(state["status"]=="complete" and state["qualification"]=="timing_only","completion state")
    check(state["summary_sha256"]==summary_hash,"completion summary hash")
    check((root/"COMPLETE").read_text()=="summary_sha256="+summary_hash+"\n","COMPLETE binding")
    progress=[json.loads(line) for line in (root/"progress.jsonl").read_text().splitlines()]
    expected={(c["id"],t) for c in spec["cases"] for t in range(spec["trials"])}
    check(len(progress)==len(expected) and {(r["case"],r["trial"]) for r in progress}==expected,"progress coverage")
    check(all(r["status"]=="collected" for r in progress),"progress state")
    check((root/"environment.txt").stat().st_size>0,"missing environment")
    check(len((root/"telemetry.csv").read_text().splitlines())>=3,"insufficient telemetry")
    return summary_hash

def audit(root, require_complete=False):
    spec=json.loads((root/"run_spec.json").read_text())
    expected_summary_hash=check_completion(root,spec) if require_complete else None
    check(spec["target"]=="GH200/SM90a" and spec["trials"]>=10,"target/trials")
    check(len(spec["cases"])==12 and {(c["id"],c["mode"],c["bytes"],c["iterations"],c["stride"]) for c in spec["cases"]}==expected_matrix(),"case matrix")
    for name,expected in spec["source_sha256"].items():
        check(sha(root/"source"/name)==expected,"source hash: "+name)
    check(sha(root/"resource_probe")==spec["binary_sha256"],"binary hash")
    check(sha(root/"sass.txt")==spec["sass_sha256"],"SASS hash")
    compile_command=json.loads((root/"compile_command.json").read_text())
    check("arch=compute_90a,code=sm_90a" in compile_command,"SM90a compile target")
    device=json.loads((root/"device.jsonl").read_text())
    check(device["cc"]=="9.0" and "GH200" in device["name"],"device identity")
    sass=(root/"sass.txt").read_text()
    functions=re.split(r"Function\s*:\s*",sass)[1:]
    for name,tokens in (("global_path",("LDG","STG")),("shared_pathILb0",("LDS",)),("shared_pathILb1",("STS",))):
        matches=[f for f in functions if name in f.splitlines()[0]]
        check(len(matches)==1,"SASS function: "+name)
        for token in tokens:
            check(re.search(r"\b"+token+r"(?:\.|\s)",matches[0]) is not None,"missing "+token)
        check_memory_loops(matches[0], tokens, 8 if name.startswith("shared_path") else None)
    results=[]
    for case in spec["cases"]:
        d=root/"cases"/case["id"];rates=[];raw_hashes=[]
        check(len(list(d.glob("trial_*.jsonl")))==spec["trials"],"raw trial count")
        for trial in range(spec["trials"]):
            raw=d/f"trial_{trial:02d}.jsonl";receipt=json.loads((d/f"trial_{trial:02d}.receipt.json").read_text())
            check(sha(raw)==receipt["raw_sha256"],"raw hash")
            check(receipt["host_stop_ns"]>receipt["host_start_ns"] and receipt["pid"]>0,"process receipt")
            expected_args=[case["mode"],str(case["bytes"]),str(case["iterations"]),str(case["stride"]),str(trial),str(3+trial*19)]
            check(receipt["command"][1:]==expected_args,"trial command")
            data=[json.loads(line) for line in raw.read_text().splitlines()]
            check(len(data)==2 and data[0]==device and data[1].get("type")=="trial","device drift / row count")
            rates.append(validate(data[1],case,device,trial));raw_hashes.append(sha(raw))
        results.append({**case,"unit":"B/clock64 cycle/CTA" if case["mode"].startswith("smem") else "GB/s requested payload/GPU",
                        "median":statistics.median(rates),"min":min(rates),"max":max(rates),
                        "cv":statistics.stdev(rates)/statistics.mean(rates),"trials":len(rates),"raw_sha256":raw_hashes,
                        "scope":"one_CTA" if case["mode"].startswith("smem") else "observed_grid_makespan"})
    ncu=json.loads((root/"ncu_status.json").read_text())
    denied=(root/"ncu.csv").exists() and "ERR_NVGPUCTRPERM" in (root/"ncu.csv").read_text()
    summary={"schema_version":1,"status":"timing_validated","device":device,"cases":results,
             "counter_permission_denied":denied,"ncu_status":ncu,
             "cache_residency_proven":False,"physical_hbm_bytes_proven":False,
             "limitations":["循环、地址运算、同步及写路径 fence 在计时内。",
                            "两次预热不证明缓存命中或长期热稳态。",
                            "global store/duplex 是请求路径吞吐，不是物理 HBM 写带宽。",
                            "SMEM 是一 CTA 的给定 stride 服务，不是整卡物理带宽。",
                            "未测试 TMA、DSM、混合计算流水线。"]}
    summary_text=json.dumps(summary,ensure_ascii=False,indent=2)+"\n"
    if require_complete:
        check(hashlib.sha256(summary_text.encode()).hexdigest()==expected_summary_hash,"recomputed summary changed")
    else:
        (root/"summary.json").write_text(summary_text)
    report=["# GH200 SMEM 与全局访问路径测量","",
            f"12 个配置，每个 {spec['trials']} 次独立进程采样。默认报告计时有效性，不声称缓存驻留或外存物理流量已证实。","",
            "设备属性："+json.dumps(device,ensure_ascii=False),"",
            "| 配置 | 单位 | 中位数 | 最小–最大 | CV |","|---|---|---:|---:|---:|"]
    for r in results:
        report.append(f"| {r['id']} | {r['unit']} | {r['median']:.3f} | {r['min']:.3f}–{r['max']:.3f} | {100*r['cv']:.2f}% |")
    report+=["","SMEM 分配 32 KiB，计时重复 256 线程各 8 次 scalar 访问；stride 改变地址与 bank 映射，也可能改变广播/地址复用。global 工作集按完整 CTA 访问轮次向上对齐，以原始 allocation 字段为准。",
             "","写路径在 device fence 和 CTA 同步之后结束计时；fence 约束内存访问顺序，不单独证明消费者可见或强制刷新到 HBM。输入初始化和输出检查不在设备计时窗口内。",""]
    report += ["- "+x for x in summary["limitations"]]
    report += ["",f"NCU 权限拒绝：{denied}；本报告没有根据工作集大小宣称 L1/L2 hit。"]
    if not require_complete:
        (root/"REPORT.md").write_text("\n".join(report)+"\n")
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("run_dir",type=Path);p.add_argument("--require-complete",action="store_true");a=p.parse_args()
    s=audit(a.run_dir,a.require_complete)
    print(json.dumps({"status":s["status"],"cases":len(s["cases"]),"counter_permission_denied":s["counter_permission_denied"]}))
