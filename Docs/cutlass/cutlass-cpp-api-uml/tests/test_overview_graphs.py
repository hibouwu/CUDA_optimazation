"""Whole-view navigation must preserve identities and the published denominator."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("overview_builder", ROOT / "scripts/build_overviews.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class OverviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atlas = json.loads((ROOT / "data/atlas.json").read_text())
        cls.spec = json.loads((ROOT / "data/overview-spec.json").read_text())
        cls.nodes = {n["id"]: n for n in cls.atlas["nodes"]}
        cls.edges = {e["id"]: e for e in cls.atlas["edges"]}
        cls.views = {v["id"]: v for v in cls.atlas["views"]}
        cls.members = builder.verify_members(cls.atlas, cls.spec, ROOT)
        cls.diagrams = builder.models(cls.atlas, cls.spec, cls.members)
        cls.by_id = {d["id"]: d for d in cls.diagrams}

    def test_three_original_views_preserve_every_edge_and_node(self):
        for vid, count in [("host.lifecycle", 25), ("host.preparation", 24), ("host.launch", 24)]:
            overview = self.by_id["overview-" + vid]
            self.assertEqual(overview["node_ids"], self.views[vid]["node_ids"])
            self.assertEqual(overview["edge_ids"], self.views[vid]["edge_ids"])
            self.assertEqual(len(overview["edge_ids"]), count)

    def test_model_is_deterministic(self):
        self.assertEqual(self.diagrams, builder.models(self.atlas, self.spec, self.members))

    def test_member_list_includes_accumulator_alias_not_nested_field(self):
        self.assertIn("contract.type.acc_pipeline", self.members)
        self.assertNotIn("contract.res.tmem_ptr", self.members)
        self.assertEqual(len(self.members), 11)

    def test_wrong_member_qname_is_rejected(self):
        atlas = deepcopy(self.atlas)
        next(n for n in atlas["nodes"] if n["id"] == "host.kernel.lower")["qualified_name"] = "not_kernel::to_underlying_arguments"
        with self.assertRaisesRegex(ValueError, "Member source identity"):
            builder.verify_members(atlas, self.spec, ROOT)

    def test_wrong_member_path_is_rejected(self):
        atlas = deepcopy(self.atlas)
        next(n for n in atlas["nodes"] if n["id"] == "host.kernel.lower")["path"] = "include/fake.hpp"
        with self.assertRaisesRegex(ValueError, "Member source identity"):
            builder.verify_members(atlas, self.spec, ROOT)

    def test_wrong_member_line_is_rejected(self):
        spec = deepcopy(self.spec)
        spec["kernel"]["members"][0]["line"] = 2
        with self.assertRaisesRegex(ValueError, "Member source identity"):
            builder.verify_members(self.atlas, spec, ROOT)

    def test_type_alias_requires_exact_declaration(self):
        atlas = deepcopy(self.atlas)
        next(n for n in atlas["nodes"] if n["id"] == "contract.type.kernel")["declaration_occurrence_id"] = "other-specialization"
        with self.assertRaisesRegex(ValueError, "exact declaration identity"):
            builder.verify_members(atlas, self.spec, ROOT)

    def test_operator_alias_requires_same_entity(self):
        atlas = deepcopy(self.atlas)
        next(n for n in atlas["nodes"] if n["id"] == "contract.api.kernel")["entity_id"] = "wrong-overload"
        with self.assertRaisesRegex(ValueError, "Operator protocol"):
            builder.verify_members(atlas, self.spec, ROOT)

    def test_denominator_change_is_not_silently_accepted(self):
        atlas = deepcopy(self.atlas)
        atlas["views"][0]["edge_ids"].pop()
        with self.assertRaisesRegex(ValueError, "denominator changed"):
            builder.models(atlas, self.spec, self.members)

    def test_kernel_includes_all_type_relations(self):
        expected = {e["id"] for e in self.atlas["edges"] if e["source"] in {"host.kernel", "contract.type.kernel"}
                    or e["target"] in {"host.kernel", "contract.type.kernel"}}
        self.assertTrue(expected <= set(self.by_id["overview-host.kernel"]["edge_ids"]))
        self.assertEqual(len(self.by_id["overview-host.kernel"]["edge_ids"]), 37)

    def test_device_process_keeps_all_outgoing_edges_of_same_entity_roles(self):
        entity = self.nodes['contract.api.kernel']['entity_id']
        aliases = {n['id'] for n in self.atlas['nodes'] if n.get('entity_id') == entity}
        expected = [e["id"] for e in self.atlas["edges"] if e["source"] in aliases]
        self.assertEqual(expected, self.by_id["process-host.kernel.device"]["edge_ids"])
        self.assertTrue({'host.kernel.operator','contract.api.kernel'} <= aliases)
        self.assertIn('scheduler.call.fetch.741',expected)
        self.assertIn("contract.edge.kernel_load", expected)
        self.assertIn("contract.edge.kernel_load_rest", expected)

    def test_process_partition_is_semantic_and_not_two_edge_slicing(self):
        lower = self.by_id["process-host.kernel.lowering"]
        self.assertEqual(lower["subject_ids"], ["host.kernel.lower"])
        for eid in lower["edge_ids"]:
            e = self.edges[eid]
            self.assertTrue(e["source"] == "host.kernel.lower" or e["target"] == "host.kernel.lower")
        self.assertGreater(len(lower["edge_ids"]), 2)

    def test_launch_processes_partition_all_original_edges(self):
        processes = [d for d in self.diagrams if d.get("parent_view_id") == "host.launch"]
        selected = [eid for d in processes for eid in d["edge_ids"]]
        self.assertEqual(len(processes), 3)
        self.assertEqual(sorted(selected), sorted(self.views["host.launch"]["edge_ids"]))
        self.assertEqual(len(set(selected)), len(selected))
        self.assertEqual([len(d["edge_ids"]) for d in processes], [11, 9, 4])

    def test_same_named_run_overloads_remain_independent(self):
        diagram = self.by_id["overview-host.lifecycle"]
        for nid in ("host.run.args", "host.run.params", "host.run.stream"):
            self.assertIn(nid, diagram["node_ids"])
        dot = builder.render_dot(diagram, self.nodes, self.edges, self.members)
        for nid in ("host.run.args", "host.run.params", "host.run.stream"):
            self.assertIn(builder.href("api", nid), dot)

    def test_no_fabricated_initialize_check_or_update_workspace(self):
        diagram = self.by_id["overview-host.lifecycle"]
        selected = [self.edges[eid] for eid in diagram["edge_ids"]]
        self.assertFalse(any(e["source"] == "host.initialize" and e["target"] == "host.can_implement" for e in selected))
        self.assertFalse(any(e["source"] == "host.update" and e["target"] == "host.kernel.initialize_workspace" for e in selected))

    def test_dot_has_one_individual_statement_per_edge(self):
        for diagram in self.diagrams:
            dot = builder.render_dot(diagram, self.nodes, self.edges, self.members)
            edges = [line for line in dot.splitlines() if " -> " in line]
            self.assertEqual(len(edges), len(diagram["edge_ids"]))
            for eid in diagram["edge_ids"]:
                self.assertEqual(dot.count('id="' + eid + '"'), 1)
                self.assertIn(builder.href("edge", eid), dot)
            self.assertIn("concentrate=false", dot)
            self.assertIn('target="_top"', dot)
            self.assertTrue(all(re.match(r"g\d+:p\d+:e -> g\d+:p\d+:w", line) for line in edges))

    def test_dot_distinguishes_type_call_launch_and_state(self):
        dot = builder.render_dot(self.by_id["overview-host.launch"], self.nodes, self.edges, self.members)
        self.assertIn('style="dashed"', dot)
        self.assertIn('color="#375d88"', dot)
        self.assertIn('color="#9b427c"', dot)
        self.assertIn('color="#21746c"', dot)

    def test_evidence_ranges_are_not_mislabeled_as_exact_callsites(self):
        self.assertEqual(builder.edge_location(self.edges["host.device.call_functor.123"]), " L122–123")
        self.assertEqual(builder.edge_location(self.edges["host.adapter.init_smem.338"]), " 证据 L329–342")
        self.assertEqual(builder.edge_location(self.edges["host.kernel.lower_mainloop.290"]), " L290")

    def test_unfinished_protocol_boundary_remains_visible(self):
        dot = builder.render_dot(self.by_id["process-host.kernel.device"], self.nodes, self.edges, self.members)
        self.assertIn("[待展开]", dot)
        self.assertIn("解析状态：pending", dot)

    def test_plantuml_has_exact_same_node_and_edge_identities(self):
        for diagram in self.diagrams:
            puml = builder.render_plantuml(diagram, self.nodes, self.edges, self.members)
            for nid in diagram["node_ids"]:
                self.assertEqual(puml.count(builder.href("api", nid) + "]]"), 1)
            for eid in diagram["edge_ids"]:
                self.assertEqual(puml.count("' source relationship: " + eid + "\n"), 1)

    def test_source_paths_remain_visible_in_each_figure(self):
        for diagram in self.diagrams:
            dot = builder.render_dot(diagram, self.nodes, self.edges, self.members)
            for nid in diagram["node_ids"]:
                if self.nodes[nid].get("path"):
                    self.assertIn(self.nodes[nid]["path"], dot)

    def test_generated_report_has_current_inputs_and_no_completion_claim(self):
        report = json.loads((ROOT / "data/overviews.json").read_text())
        self.assertEqual(report["base_atlas_sha256"], builder.sha(ROOT / "data/atlas.json"))
        self.assertEqual(report["input_sha256"]["scripts/build_overviews.py"], builder.sha(ROOT / "scripts/build_overviews.py"))
        self.assertFalse(report["actual_browser_verified"])
        self.assertFalse(report["global_completion_claimed"])
        self.assertEqual(report["type_overviews"]["host.kernel"], report["type_overviews"]["contract.type.kernel"])

    def test_rendered_svg_keeps_clickable_rows_edges_and_top_target(self):
        ns = {"svg": "http://www.w3.org/2000/svg"}
        href_attr = "{http://www.w3.org/1999/xlink}href"
        report = json.loads((ROOT / "data/overviews.json").read_text())
        for diagram in report["diagrams"]:
            xml = ET.parse(ROOT / "site" / diagram["svg"]).getroot()
            links = xml.findall(".//svg:a", ns)
            urls = {a.get(href_attr) for a in links}
            for nid in diagram["node_ids"]:
                self.assertIn(builder.href("api", nid), urls)
            for eid in diagram["edge_ids"]:
                self.assertIn(builder.href("edge", eid), urls)
            self.assertTrue(all(a.get("target") == "_top" for a in links))
            graph_edges = [g for g in xml.findall(".//svg:g", ns) if g.get("class") == "edge"]
            self.assertEqual(len(graph_edges), len(diagram["edge_ids"]))


if __name__ == "__main__":
    unittest.main()
