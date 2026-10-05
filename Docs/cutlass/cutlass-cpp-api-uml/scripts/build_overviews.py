#!/usr/bin/env python3
"""Add source-linked whole-process navigation to an already published atlas.

This does not rerun extraction, alter atlas.json, or infer member ownership from
graph IDs. API signatures and edge evidence remain in the original detail pages.
Graphviz renders independent API row ports; PlantUML retains the same nodes and
edges as editable, independent operation nodes. Neither output collapses edges.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
RELATIONS = {
    "calls": ("调用", "#375d88", "solid"),
    "type_uses": ("类型", "#7a6095", "dashed"),
    "template_binds": ("模板绑定", "#7a6095", "dashed"),
    "reads": ("读取", "#21746c", "solid"),
    "writes": ("写入", "#21746c", "bold"),
    "launches": ("设备提交", "#9b427c", "bold"),
    "signals": ("通知", "#a6741d", "bold"),
    "issues": ("发起异步操作", "#9b427c", "bold"),
    "waits_for": ("等待条件", "#a6741d", "dashed"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def quoted(value):
    return json.dumps(str(value), ensure_ascii=False)


def href(kind, identifier):
    return "../index.html?" + kind + "=" + quote(identifier, safe="")


def edge_location(edge):
    evidence = edge.get("callsite") or next(iter(edge.get("evidence") or []), {})
    first = evidence.get("start_line") or evidence.get("line")
    if not first:
        return ""
    last = evidence.get("end_line") or first
    location = "L" + str(first) + ("–" + str(last) if last != first else "")
    return " " + ("" if edge.get("callsite") else "证据 ") + location


def short_name(node):
    """Short navigation label only; no change to the complete API declaration."""
    known = {
        "host.app.main": "main()",
        "host.adapter": "GemmUniversalAdapter",
        "host.kernel": "GemmUniversal",
        "contract.type.kernel": "GemmUniversal（设备协议视图）",
        "contract.type.acc_pipeline": "AccumulatorPipeline（类型别名）",
        "contract.api.kernel": "operator()（同一声明：计算/资源角色）",
        "contract.type.recipe": "MainloopSm100TmaUmmaWarpSpecialized",
        "contract.type.mainloop": "CollectiveMma（已选特化）",
        "contract.type.epilogue": "CollectiveEpilogue（已选特化）",
        "host.adapter.arguments": "Arguments = GemmKernel::Arguments",
        "host.adapter.params": "Params = GemmKernel::Params",
        "host.state.params": "params_ : Params",
        "host.run.kernel_params": "kernel_params[] = {&params}",
        "host.call.args": "operator()(Arguments, …)",
        "host.call.stream": "operator()(stream, …)",
        "host.run.args": "run(Arguments, …)",
        "host.run.stream": "run(stream, …)",
        "host.run.params": "static run(Params&, …)",
        "host.grid.args": "get_grid_shape(Arguments, …)",
        "host.grid.params": "get_grid_shape(Params)",
        "host.kernel.operator": "operator()（同一声明：任务交接角色）",
        "host.kernel.smem_size": "SharedStorageSize",
        "host.device_kernel": "device_kernel<Operator>(Params)",
        "host.scheduler": "TileScheduler（已选类型）",
        "scheduler.fetch": "fetch_next_work(WorkTileInfo, Pipeline&, State)",
        "scheduler.fetch_compat": "fetch_next_work(WorkTileInfo)",
        "scheduler.epilogue": "compute_epilogue(WorkTileInfo)",
        "scheduler.epilogue_compat": "compute_epilogue(WorkTileInfo, Params)",
        "scheduler.fixup_state": "fixup<IsComplex>(6个参数)",
        "clc.acquire": "producer_acquire(State, Token)",
        "clc.acquire_stage": "producer_acquire(stage, phase, Token)",
        "clc.wait": "consumer_wait(State, Token)",
        "clc.wait_stage": "consumer_wait(stage, phase, Token)",
        "clc.release": "consumer_release(State)",
        "clc.release_stage": "consumer_release(stage)",
        "clc.commit": "producer_commit(State)",
        "clc.commit_stage": "producer_commit(stage, phase)",
        "clc.expect_remote": "arrive_and_expect_tx(bytes, cta_id, pred)",
        "clc.complete_remote": "complete_transaction(cta_id, bytes, pred)",
    }
    if node["id"] in known:
        return known[node["id"]]
    value = (node.get("qualified_name") or node["name"]).split("::")[-1]
    if node["kind"] == "api" and "(" not in value:
        value += "()"
    return value


def without_templates(value):
    result, depth = [], 0
    for char in value:
        if char == "<":
            depth += 1
        elif char == ">" and depth:
            depth -= 1
        elif not depth:
            result.append(char)
    return "".join(result)


def verify_members(atlas, spec, root):
    """Whitelist + exact source/QName/declaration checks, not ID-prefix ownership."""
    nodes = {n["id"]: n for n in atlas["nodes"]}
    kernel = spec["kernel"]
    owner = nodes[kernel["id"]]
    if owner.get("path") != kernel["path"] or owner.get("qualified_name") != kernel["qualified_name"]:
        raise ValueError("Kernel owner no longer matches reviewed source identity")
    if owner.get("declaration_status") != "linked_to_source_ledger":
        raise ValueError("Kernel type lacks linked source declaration")
    source = (root / "snapshot" / kernel["path"]).read_text().splitlines()
    checked = []
    for member in kernel["members"]:
        node = nodes[member["id"]]
        expected_name = kernel["qualified_name"] + "::" + member["name"]
        actual_name = node.get("qualified_name") or without_templates(node.get("full_name") or "")
        if (node.get("path") != kernel["path"] or actual_name != expected_name
                or node.get("line") != member["line"]
                or node.get("declaration_status") != "linked_to_source_ledger"):
            raise ValueError("Member source identity mismatch: " + member["id"])
        span = node.get("signature_range", {})
        first, last = span.get("start_line", 0), span.get("end_line", 0)
        if span.get("path") != kernel["path"] or not first <= member["line"] <= last:
            raise ValueError("Member lacks matching declaration range: " + member["id"])
        written = "\n".join(source[first - 1:last])
        # operator() has whitespace between the keyword and its argument list.
        token = re.escape(member["name"]).replace(r"\(\)", r"\s*\(\)")
        if not re.search(token, written):
            raise ValueError("Member name missing at fixed source location: " + member["id"])
        checked.append(member["id"])
    for identifier in kernel.get("aliases", []):
        alias = nodes[identifier]
        if (not owner.get("entity_id") or alias.get("entity_id") != owner["entity_id"]
                or alias.get("path") != owner["path"]
                or alias.get("declaration_occurrence_id") != owner.get("declaration_occurrence_id")):
            raise ValueError("Type alias view does not share exact declaration identity: " + identifier)
    operator, alias = nodes["host.kernel.operator"], nodes[kernel["operator_alias"]]
    if (alias.get("entity_id") != operator.get("entity_id")
            or alias.get("declaration_occurrence_id") != operator.get("declaration_occurrence_id")):
        raise ValueError("Operator protocol view has different declaration identity")
    return checked


def models(atlas, spec, members):
    nodes = {n["id"]: n for n in atlas["nodes"]}
    edges = {e["id"]: e for e in atlas["edges"]}
    views = {v["id"]: v for v in atlas["views"]}
    diagrams = []
    for item in spec["views"]:
        view = views[item["id"]]
        if len(view["edge_ids"]) != item["expected_edge_count"]:
            raise ValueError("Reviewed view denominator changed: " + item["id"])
        diagrams.append({"id": "overview-" + item["id"], "title": item["title"],
                         "kind": "overview", "view_id": item["id"],
                         "node_ids": list(view["node_ids"]), "edge_ids": list(view["edge_ids"]),
                         "scope": spec["scope"], "relation_coverage": "all_edges_in_original_view"})
    for view_id, processes in spec.get("view_processes", {}).items():
        owned_edges = []
        for process in processes:
            selected = [eid for eid in views[view_id]["edge_ids"] if edges[eid]["source"] in process["sources"]]
            owned_edges.extend(selected)
            related = {edges[eid][side] for eid in selected for side in ("source", "target")}
            diagrams.append({"id": "process-" + view_id + "." + process["id"], "title": process["title"],
                             "kind": "process", "parent_view_id": view_id, "subject_ids": process["sources"],
                             "node_ids": [n["id"] for n in atlas["nodes"] if n["id"] in related],
                             "edge_ids": selected, "scope": spec["scope"],
                             "relation_coverage": "all_original_view_edges_from_reviewed_responsibility_group"})
        if sorted(owned_edges) != sorted(views[view_id]["edge_ids"]):
            raise ValueError("Semantic processes must partition original view without missing or duplicated edges: " + view_id)
    owner_ids = {spec["kernel"]["id"], *spec["kernel"].get("aliases", []), *members}
    type_ids = {spec["kernel"]["id"], *spec["kernel"].get("aliases", [])}
    incident = [e["id"] for e in atlas["edges"]
                if ((e.get("part") == spec["kernel"]["relation_part"]
                     and (e["source"] in owner_ids or e["target"] in owner_ids))
                    or e["source"] in type_ids or e["target"] in type_ids)]
    all_nodes = owner_ids | {edges[i][side] for i in incident for side in ("source", "target")}
    diagrams.append({"id": "overview-host.kernel", "title": "GemmUniversal：已记录成员与外部接口总图",
                     "kind": "overview", "type_id": "host.kernel",
                     "node_ids": [n["id"] for n in atlas["nodes"] if n["id"] in all_nodes],
                     "edge_ids": incident, "member_ids": members,
                     "scope": "此特化的已记录 Host 交接、Arguments/Params 与设备入口；operator() 内部协议从过程视图继续，不代表完整类成员清单。",
                     "relation_coverage": "all_type_incident_edges_plus_host_part_edges_of_verified_members"})
    # Semantic boundaries follow the source API whose responsibility is reviewed.
    # Incoming adapter calls and every selected member's outgoing relation stay
    # together, unlike the former fixed two-edge slicing.
    processes = [
        ("check", "合法性检查", ["host.kernel.can_implement"]),
        ("workspace", "Workspace 查询与初始化", ["host.kernel.workspace", "host.kernel.initialize_workspace"]),
        ("lowering", "Arguments → Params：参数交接", ["host.kernel.lower"]),
        ("launch-shape", "提交配置与设备入口", ["host.kernel.grid", "host.kernel.block", "host.kernel.operator"]),
    ]
    for name, title, subjects in processes:
        selected = [eid for eid in incident if edges[eid]["source"] in subjects or edges[eid]["target"] in subjects]
        related = {edges[eid][side] for eid in selected for side in ("source", "target")}
        diagrams.append({"id": "process-host.kernel." + name, "title": title, "kind": "process",
                         "type_id": "host.kernel", "subject_ids": subjects,
                         "node_ids": [n["id"] for n in atlas["nodes"] if n["id"] in related],
                         "edge_ids": selected, "scope": spec["scope"],
                         "relation_coverage": "all_host_part_incident_edges_of_process_members"})
    operator_id = spec["kernel"]["operator_alias"]
    operator_aliases = {n['id'] for n in atlas['nodes'] if n.get('entity_id') and n.get('entity_id') == nodes[operator_id].get('entity_id')}
    device_edges = [e["id"] for e in atlas["edges"] if e["source"] in operator_aliases]
    device_nodes = operator_aliases | {edges[eid]["target"] for eid in device_edges}
    diagrams.append({"id": "process-host.kernel.device", "title": "operator()：已记录的设备交接总图",
                     "kind": "process", "type_id": "host.kernel", "subject_ids": sorted(operator_aliases),
                     "node_ids": [n["id"] for n in atlas["nodes"] if n["id"] in device_nodes],
                     "edge_ids": device_edges,
                     "scope": "保留已发布operator()同实体各角色的全部出边；角色节点不是额外重载。各调用点的分支与阶段条件见逐边证据，不以图中位置推断执行顺序，也不宣称完整函数覆盖。",
                     "relation_coverage": "all_published_outgoing_edges_of_operator_entity_aliases"})
    for diagram in diagrams:
        if len(set(diagram["edge_ids"])) != len(diagram["edge_ids"]):
            raise ValueError("Duplicate edge in " + diagram["id"])
        if not all(i in nodes for i in diagram["node_ids"]):
            raise ValueError("Missing node in " + diagram["id"])
        for eid in diagram["edge_ids"]:
            edge = edges[eid]
            if edge["source"] not in diagram["node_ids"] or edge["target"] not in diagram["node_ids"]:
                raise ValueError("Missing edge endpoint: " + eid)
            if edge["relation"] not in RELATIONS:
                raise ValueError("Unhandled relationship style: " + edge["relation"])
    return diagrams


def grouping(node, members):
    """Presentation group labels are not fabricated C++ ownership relations."""
    identifier = node["id"]
    if identifier.startswith('clc.res.'):
        return ('clc-resources','CLC响应与屏障（资源绑定）','#f0f7f6')
    if identifier.startswith('clc.hw.'):
        return ('clc-hardware','CLC异步硬件操作','#fff0f6')
    if (node.get('qualified_name') or '').startswith('cutlass::PipelineCLCFetchAsync'):
        return ('clc-pipeline','PipelineCLCFetchAsync','#f4f0fb')
    if identifier in {"host.kernel", "contract.type.kernel", "contract.api.kernel"} or identifier in members:
        return ("kernel", "GemmUniversal · 已核对的成员 / 类型", "#edf4ff")
    qname = node.get("qualified_name") or ""
    if qname.startswith("cutlass::gemm::device::GemmUniversalAdapter::") or identifier == "host.adapter":
        if identifier in {"host.adapter", "host.adapter.arguments", "host.adapter.params"}:
            return ("adapter-types", "Adapter · 类型入口", "#f3edff")
        if identifier in {"host.call.args", "host.call.stream", "host.run.args", "host.run.stream", "host.run.params"}:
            return ("adapter-run", "Adapter · 重载与提交", "#eef7f4")
        return ("adapter-prepare", "Adapter · 准备 / 查询 / 状态", "#eef7f4")
    if qname.startswith("cutlass::gemm::collective::CollectiveMma::") or node.get("path") == "include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp":
        return ("mainloop", "Mainloop · 已选实现", "#fff4e5")
    if (node.get("path") or "").startswith("include/cutlass/epilogue/"):
        return ("epilogue", "Epilogue · 已选实现", "#fff4e5")
    if (node.get("path") or "").startswith("include/cutlass/gemm/kernel/tile_scheduler") or node.get("path") == "include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp":
        return ("scheduler", "Scheduler · 已选实现", "#fff4e5")
    if identifier == "host.app.main":
        return ("app", "调用方示例", "#f1f5f8")
    if node.get("declaration_status") == "external_boundary":
        return ("cuda", "CUDA API · 外部边界", "#fff0f6")
    if node.get("path") == "include/cutlass/cluster_launch.hpp":
        return ("cluster", "Cluster 提交辅助", "#f3f3fa")
    if identifier == "host.device_kernel":
        return ("device", "设备入口包装", "#f3f3fa")
    return ("other-" + (node.get("path") or "boundary"), "其他已记录接口 / 对象", "#f4f4f4")


def graph_parts(diagram, nodes, members):
    groups = {}
    for nid in diagram["node_ids"]:
        key, label, color = grouping(nodes[nid], members)
        if diagram["id"] == "overview-host.launch" and key == "cluster":
            if nid == "host.cluster.launch":
                key, label = "cluster-entry", "Cluster 提交入口"
            else:
                key, label = "cluster-helpers", "Cluster · 配置与检查"
        if diagram["id"] == "process-host.kernel.device" and nid not in diagram.get('subject_ids', []):
            # This is a presentation list, explicitly not a C++ owner/type box.
            key, label, color = "device-recipients", "被调用接口 / 交接对象（分属不同类型）", "#fff4e5"
        groups.setdefault(key, {"title": label, "color": color, "nodes": []})["nodes"].append(nid)
    paths = sorted({nodes[nid]["path"] for nid in diagram["node_ids"] if nodes[nid].get("path")})
    files = {path: "F" + str(i + 1) for i, path in enumerate(paths)}
    return groups, files


def render_dot(diagram, nodes, edges, members):
    groups, files = graph_parts(diagram, nodes, members)
    # Each API has a separate row port, hyperlink and tooltip. The box is merely
    # its labeled grouping; calls never attach to a whole owner frame.
    legend = ['每行 = 独立接口 / 对象；点击行看完整声明，点击边看条件与源码。',
              '蓝实线：调用；紫虚线：类型 / 模板；绿线：状态；紫粗线：设备提交；棕粗线：通知。',
              '此图不是时间顺序图；API 的生效条件和失败分支随原始关系保存。']
    legend += [key + " = " + path for path, key in files.items()]
    ranksep = "0.35" if diagram["id"] == "overview-host.launch" else "0.75"
    lines = ["digraph overview {", 'graph [rankdir=LR, bgcolor="white", fontname="Noto Sans CJK SC", '
             'fontsize=17, nodesep=0.34, ranksep=' + ranksep + ', pad=0.25, splines=spline, concentrate=false, '
             'outputorder=edgesfirst, labelloc=t, label=<<B>' + html.escape(diagram["title"])
             + '</B><BR/><FONT POINT-SIZE="11">' + '<BR/>'.join(html.escape(s) for s in legend) + '</FONT>>];',
             'node [shape=plain, fontname="Noto Sans CJK SC", fontsize=13];',
             'edge [fontname="Noto Sans CJK SC", fontsize=10, arrowsize=0.7, penwidth=1.15, target="_top"];']
    ports = {}
    for gi, group in enumerate(groups.values()):
        group_id = "g" + str(gi)
        rows = ['<TR><TD BGCOLOR="' + group["color"] + '"><B>' + html.escape(group["title"]) + '</B></TD></TR>']
        for ni, nid in enumerate(group["nodes"]):
            node = nodes[nid]
            port = "p" + str(ni)
            ports[nid] = group_id + ":" + port
            loc = (files[node["path"]] + ":L" + str(node.get("line") or 1)) if node.get("path") else "外部声明边界"
            label = html.escape(short_name(node))
            tip = node.get("signature") or node.get("qualified_name") or node["name"]
            rows.append('<TR><TD PORT="' + port + '" ALIGN="LEFT" HREF="' + html.escape(href("api", nid), quote=True)
                        + '" TARGET="_top" TOOLTIP="' + html.escape(tip, quote=True).replace("\n", "&#10;")
                        + '">' + label + ' <FONT POINT-SIZE="10" COLOR="#677587">' + loc + '</FONT></TD></TR>')
        lines.append(group_id + ' [label=<<TABLE BORDER="1" COLOR="#a9b8c8" CELLBORDER="0" CELLSPACING="0" CELLPADDING="7">'
                     + "".join(rows) + '</TABLE>>];')
    for i, eid in enumerate(diagram["edge_ids"]):
        edge = edges[eid]
        label, color, style = RELATIONS[edge["relation"]]
        status = " [待展开]" if edge.get("resolution") == "pending" else ""
        edge_label = "E" + str(i + 1).zfill(2) + " " + label + edge_location(edge) + status
        tip = eid + "\n" + str(edge.get("source_expression") or "") + "\n条件：" + str(edge.get("condition") or "见详细关系") + "\n解析状态：" + str(edge.get("resolution") or "见详细关系")
        # These relations remain real visible edges, but do not force unrelated
        # compile-time types and the functor's frame to the end of the Host chain.
        layout = ', constraint=false' if diagram["id"] == "overview-host.launch" and (
            edge["relation"] in {"type_uses", "template_binds"} or eid == "host.device.call_functor.123") else ''
        lines.append(ports[edge["source"]] + ":e -> " + ports[edge["target"]] + ':w [id=' + quoted(eid)
                     + ', label=' + quoted(edge_label) + ', URL=' + quoted(href("edge", eid))
                     + ', tooltip=' + quoted(tip) + ', labeltooltip=' + quoted(tip)
                     + ', color=' + quoted(color) + ', fontcolor=' + quoted(color) + ', style=' + quoted(style) + layout + '];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_plantuml(diagram, nodes, edges, members):
    groups, files = graph_parts(diagram, nodes, members)
    aliases = {nid: "n" + str(i) for i, nid in enumerate(diagram["node_ids"])}
    lines = ["@startuml", "!pragma layout smetana", "left to right direction", "skinparam shadowing false",
             "skinparam defaultFontName Noto Sans CJK SC", "skinparam defaultFontSize 13", "title " + diagram["title"]]
    for group in groups.values():
        lines.append('rectangle "' + group["title"] + '" ' + group["color"] + " {")
        for nid in group["nodes"]:
            node = nodes[nid]
            loc = files[node["path"]] + ":L" + str(node.get("line") or 1) if node.get("path") else "外部声明边界"
            label = (short_name(node) + "\\n" + loc).replace('"', '<U+0022>')
            lines.append('rectangle "' + label + '" as ' + aliases[nid] + " [[" + href("api", nid) + "]] ")
        lines.append("}")
    for i, eid in enumerate(diagram["edge_ids"]):
        edge = edges[eid]
        label, color, style = RELATIONS[edge["relation"]]
        arrow = "-[" + color + (",dashed" if style == "dashed" else ",bold" if style == "bold" else "") + "]->"
        lines.append("' source relationship: " + eid)
        lines.append(aliases[edge["source"]] + " " + arrow + " " + aliases[edge["target"]] + " : [[" + href("edge", eid)
                     + " E" + str(i + 1).zfill(2) + " " + label + "]]")
    lines += ["legend bottom", "每个操作节点与每条边均保留独立身份；总图不表示执行先后。",
              "调用 / 状态 / 编译期类型关系使用不同线型；完整签名、条件及源码在链接页。"]
    lines += [key + " = " + path for path, key in files.items()]
    lines += ["endlegend", "@enduml"]
    return "\n".join(lines) + "\n"


def svg_dimensions(path):
    svg = ET.parse(path).getroot()
    return {key: float(re.sub(r"[^0-9.]", "", svg.attrib[key])) for key in ("width", "height")}


def build(root=ROOT):
    root = Path(root)
    atlas_path, spec_path = root / "data/atlas.json", root / "data/overview-spec.json"
    atlas = json.loads(atlas_path.read_text())
    spec = json.loads(spec_path.read_text())
    members = verify_members(atlas, spec, root)
    diagrams = models(atlas, spec, members)
    nodes, edges = ({n["id"]: n for n in atlas["nodes"]}, {e["id"]: e for e in atlas["edges"]})
    out = root / "site/diagrams"
    out.mkdir(parents=True, exist_ok=True)
    for diagram in diagrams:
        stem = diagram["id"]
        dot, puml, svg = (out / (stem + suffix) for suffix in (".dot", ".puml", ".svg"))
        dot.write_text(render_dot(diagram, nodes, edges, members))
        puml.write_text(render_plantuml(diagram, nodes, edges, members))
        subprocess.run(["dot", "-Tsvg", str(dot), "-o", str(svg)], check=True, capture_output=True, text=True)
        diagram.update({"svg": "diagrams/" + svg.name, "plantuml": "diagrams/" + puml.name,
                        "dot": "diagrams/" + dot.name, "svg_dimensions": svg_dimensions(svg)})
    aliases = [spec["kernel"]["id"], *spec["kernel"].get("aliases", [])]
    existing_views = {v["id"] for v in atlas["views"]}
    report = {
        "schema_version": 1, "generated_by": "scripts/build_overviews.py", "scope": spec["scope"],
        "base_atlas_sha256": sha(atlas_path), "commit": atlas["commit"],
        "input_sha256": {"scripts/build_overviews.py": sha(Path(__file__)), "data/overview-spec.json": sha(spec_path)},
        "diagrams": diagrams,
        "view_overviews": {d["view_id"]: d["id"] for d in diagrams if "view_id" in d},
        "type_overviews": {i: "overview-host.kernel" for i in aliases},
        "type_members": {i: members for i in aliases},
        "type_process_views": {i: [v for v in spec["kernel"]["process_views"] if v in existing_views] for i in aliases},
        "type_process_diagrams": {i: [d["id"] for d in diagrams if d["kind"] == "process" and d.get("type_id") == "host.kernel"] for i in aliases},
        "view_process_diagrams": {view_id: [d["id"] for d in diagrams if d.get("parent_view_id") == view_id]
                                  for view_id in spec.get("view_processes", {})},
        "member_validation": "explicit_whitelist_path_qname_line_signature_range_and_snapshot_name",
        "actual_browser_verified": False, "global_completion_claimed": False,
    }
    dump(root / "data/overviews.json", report)
    assets = root / "site/assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "overview-data.js").write_text("window.ATLAS_OVERVIEWS=" + json.dumps(report, ensure_ascii=False, separators=(",", ":")) + ";\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = build(args.root)
    print(json.dumps({"diagrams": len(report["diagrams"]), "base_atlas_sha256": report["base_atlas_sha256"],
                      "sizes": {d["id"]: d["svg_dimensions"] for d in report["diagrams"]}}, ensure_ascii=False))
