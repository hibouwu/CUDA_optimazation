#!/usr/bin/env python3
"""Build the finite meeting pack from published data; never run extraction."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlsplit
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ATLAS = HERE.parent
PIN = "8f50b052e1099fb982392a622caab69b97b63128"

# Each entry selects an already-published independent source entity.
GROUPS = [
    ("ref-host", "Host 调用", [
        ("host.can_implement", "独立检查请求能否实现；initialize 不会自动调用它。"),
        ("host.get_workspace_size", "返回所需字节数，不负责分配。查询、分配、初始化和回收分别核对。"),
        ("host.initialize", "初始化 workspace 后重建 params_，按条件设置共享内存属性；不是执行 GEMM。"),
        ("host.update", "不调用 initialize_workspace，也不保证轻量更新。变更问题或存储之前要重新确认前提。"),
        ("host.run.stream", "使用成员 params_ 的实例重载；与下面的静态 run 保持独立。"),
        ("host.run.params", "提交设备任务并检查启动/API 状态；返回成功不能证明设备完成。"),
        ("host.cluster.launch", "当前已选分支的 Cluster launch 包装。不要推广到其他架构、PDL 或 Host Adapter 分支。"),
        ("host.device_kernel", "CUDA 运行时调度的设备入口；入口内才调用设备 functor。"),
    ]),
    ("ref-build", "编译期选择", [
        ("types.builder_entry", "GEMM CollectiveBuilder 主模板入口；完整模板参数如下。没有适用实现的组合在编译期拒绝。"),
        ("types.builder_sm100", "已选 SM100 偏特化；特化条件决定适用性，CollectiveOp 是产生的类型，不是运行时返回对象。"),
        ("types.atom_call", "MMA_Atom 的一次操作接口。操作数表示必须符合具体 Atom/Traits，不能从 Fragment 名字直接推出存储位置。"),
        ("types.fma", "当前类型选择最终使用的 2SM 硬件包装；其约束不自动适用于其他硬件。"),
    ]),
    ("ref-handoff", "参数和职责交接", [
        ("host.kernel.arguments", "面向调用方的参数，包括 mode、problem_shape、mainloop、epilogue、hw_info、scheduler。"),
        ("host.kernel.params", "设备所用参数，嵌套字段已转换为各组件的 Params；不是 Arguments 的别名。"),
        ("host.kernel.initialize_workspace", "Kernel 将 workspace 初始化交给具体组件；与下面的参数转换操作不同。"),
        ("host.kernel.lower", "划分并对齐 workspace，分别调用三个转换接口；当前实现的 mainloop_workspace 为 nullptr。"),
        ("host.mainloop.lower", "转换 Mainloop 输入；TMA 描述符内部构造尚未由本会议图完整展开。"),
        ("host.epilogue.lower", "转换 Epilogue 输入；C/D、融合参数和空间依赖需要独立核对。"),
        ("host.scheduler.lower", "转换工作分配参数；它不等于 Mainloop Schedule，也不是 MMA Atom 选择器。"),
        ("host.kernel.operator", "设备 functor 收到 Params 与共享内存地址，组织角色和各组件执行。"),
    ]),
    ("ref-sync", "同步与资源使用", [
        ("contract.api.input_acquire", "取得当前输入 stage 的可写条件；参与角色、stage/phase 与事务计数必须一致。"),
        ("contract.api.input_wait", "等待当前输入 stage 可读；发起搬运与完成搬运是不同事件。"),
        ("contract.api.input_release", "把相关 MMA 完成和 empty 通知联系起来；发出通知不等于下一轮 acquire 已成功。"),
        ("contract.api.store_wait", "实际 PTX 是 wait_group.read；源 SMEM 读完不等于全局 D 写完。"),
        ("contract.api.allocate", "双 CTA TMEM 分配有参与 warp 与配对条件；不能把它当作任意线程均可调用的通用分配器。"),
        ("contract.api.free", "释放前要满足拥有者与协作完成条件；单侧 free 发出不等于配对释放完成。"),
    ]),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def references(data: dict) -> list[dict]:
    nodes = {n["id"]: n for n in data["nodes"]}
    files = {f["path"]: f for f in data["files"]}
    chunks = [
        "# 官方接口速查\n",
        f"固定提交 `{PIN}`；库版本 4.6.0，采用其中的 3.x API 体系。以下仅为会议精选接口，不是完整 API 清单。\n",
        "每项保留独立身份、原始完整声明、仓库相对路径和固定行号。在线原文仅作可选核对；离线源码和已有详细图无需联网。具体语义说明适用于会议包所列 Dense 配置。\n",
    ]
    selected = []
    for anchor, title, rows in GROUPS:
        chunks.append(f'<a id="{anchor}"></a>\n\n## {title}\n')
        for node_id, claim in rows:
            n = nodes[node_id]
            path = n["path"]
            assert sha(ATLAS / "snapshot" / path) == files[path]["sha256"], path
            assert n.get("signature"), node_id
            raw = (ATLAS / "snapshot" / path).read_text()
            assert re.sub(r"\s+", "", n["signature"]) in re.sub(r"\s+", "", raw), node_id
            source_url = "../site/" + n["source_url"]
            upstream = f"https://github.com/NVIDIA/cutlass/blob/{PIN}/{path}#L{n['line']}"
            record = {"id": node_id, "path": path, "line": n["line"],
                      "source_sha256": files[path]["sha256"], "claim": claim}
            selected.append(record)
            chunks.extend([
                f'<a id="api-{node_id}"></a>\n\n### {n["name"]}\n',
                f"{claim}\n",
                f"`{path}:{n['line']}`\n",
                f"[离线源码]({source_url}) · [完整接口与独立关系](../site/index.html?api={node_id}) · [固定提交原文（联网）]({upstream})\n",
                f"限定名：`{n.get('full_name') or n['qualified_name']}`\n",
                f"```cpp\n{n['signature']}\n```\n",
            ])
            if node_id in {"host.kernel.arguments", "host.kernel.params"}:
                start = 215 if node_id.endswith("arguments") else 225
                lines = (ATLAS / "snapshot" / path).read_text().splitlines()
                chunks.append("字段定义（固定源码）：\n\n```cpp\n" + "\n".join(lines[start - 1:start + 7]) + "\n```\n")
    chunks.extend([
        "## 同步接口之外还要查什么\n",
        "`Allocator2Sm::release_allocation_lock()` 只是放弃分配许可，不是释放已分配的 TMEM。其声明见 [固定源码第 174 行](../site/source/include/cute/arch/tmem_allocator_sm100.hpp.html#L174)。Kernel 在调用 `free()` 前的独立等待、双 CTA 握手与条件分支见 [第 783 行起](../site/source/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp.html#L783) 和 [TMEM 协议详细图](../site/index.html?protocol=dense.protocol_draft.tmem_lifetime)。\n",
        "`cudaStreamSynchronize(stream)` 是本参考应用的显式等待，不在 Adapter 内部。见 [已保存的示例调用位置](../site/evidence/6c1d941eb5a1b64f-dense_baseline.cu.html#L250)。其他实现应说明自己的等价完成条件；不能只检查执行接口的返回值。\n",
        "本包引用已发布的静态编译材料，但没有新增 GPU 执行。完整 Copy 分派、全部描述符/调度/融合行为和其他配置尚不能由这些接口证明。未展开事项在会议记录中另立问题，不由简图补猜。\n",
    ])
    (HERE / "references.md").write_text("\n".join(chunks))
    return selected


NAV = '<nav aria-label="会议材料"><a href="index.html">图与使用顺序</a><a href="handbook.html">现场手册</a><a href="references.html">官方接口速查</a><a href="review-record-template.html">空白记录</a><a href="review-record-template.md" download>下载记录模板</a><a href="../site/index.html">详细图集</a></nav>'


def render_pages() -> None:
    for src in ["README", "handbook", "references", "review-record-template", "review"]:
        source = HERE / f"{src}.md"
        if not source.exists():
            continue
        result = subprocess.run(["pandoc", str(source), "--from=markdown", "--to=html5"],
                                check=True, text=True, capture_output=True).stdout
        # Only meeting prose pages change extension. All upstream links are retained.
        for name in ["README", "handbook", "references", "review-record-template", "review"]:
            dest = "index" if name == "README" else name
            result = re.sub(rf'href="{re.escape(name)}\.md(?=["#])', f'href="{dest}.html', result)
        title = source.read_text().splitlines()[0].removeprefix("# ")
        from html import escape
        document = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                    f'<meta name="viewport" content="width=device-width, initial-scale=1">'
                    f'<title>{escape(title)}</title><link rel="stylesheet" href="style.css"></head>'
                    f'<body>{NAV}<main>{result}</main><footer>会议精选参考 · CUTLASS 4.6.0 / 3.x API 体系 · 全库图集未完成</footer></body></html>')
        (HERE / ("index.html" if src == "README" else f"{src}.html")).write_text(document)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        for key in ("href", "src", "data"):
            if key in a:
                self.links.append(a[key])


def check_links(data: dict) -> dict:
    total = 0
    target_cache = {}
    routes = {"api": {n["id"] for n in data["nodes"]},
              "view": {v["id"] for v in data["views"]},
              "protocol": {p["id"] for p in data["protocols"]}}
    for file in HERE.glob("*.html"):
        parser = Links()
        parser.feed(file.read_text())
        for url in parser.links:
            parts = urlsplit(url)
            if parts.scheme:
                assert parts.scheme == "https", url
                continue
            target = (file.parent / unquote(parts.path)).resolve() if parts.path else file
            assert target.is_relative_to(ATLAS), url
            assert target.is_file(), (file.name, url)
            if target == (ATLAS / "site/index.html"):
                for key, values in parse_qs(parts.query).items():
                    assert key in routes, (file.name, url)
                    assert all(v in routes[key] for v in values), (file.name, url)
            if parts.fragment and target.suffix == ".html":
                if target not in target_cache:
                    p = Links()
                    p.feed(target.read_text())
                    target_cache[target] = p.ids
                assert unquote(parts.fragment) in target_cache[target], (file.name, url)
            total += 1
    dimensions = {}
    for svg in sorted((HERE / "diagrams").glob("*.svg")):
        root = ET.parse(svg).getroot()
        assert "Syntax Error" not in svg.read_text(), svg
        dimensions[svg.name] = {k: root.attrib.get(k) for k in ("width", "height", "viewBox")}
    assert len(dimensions) == 5
    return {"local_links_checked": total, "svg": dimensions, "actual_browser_verified": False}


def main() -> None:
    data_path = ATLAS / "data/atlas.json"
    before = sha(data_path)
    data = json.loads(data_path.read_text())
    assert data["commit"] == PIN
    selected = references(data)
    env = dict(os.environ)
    env["JAVA_TOOL_OPTIONS"] = "-Djava.awt.headless=true"
    env["PLANTUML_LIMIT_SIZE"] = "24000"
    diagrams = sorted((HERE / "diagrams").glob("*.puml"))
    subprocess.run(["plantuml", "-charset", "UTF-8", "-tsvg", *map(str, diagrams)], check=True, env=env)
    render_pages()
    report = check_links(data)
    assert before == sha(data_path), "Published atlas input changed during meeting build"
    report.update({"commit": PIN, "published_atlas_sha256": before, "selected_api_count": len(selected),
                   "selected_sources": selected, "full_library_complete": False,
                   "scope": "finite_meeting_pack", "source_inputs_verified": True})
    (HERE / "checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("local_links_checked", "selected_api_count", "source_inputs_verified", "actual_browser_verified")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
