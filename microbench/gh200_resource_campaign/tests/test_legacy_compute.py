"""CPU checks of generated contracts and host reference; no CUDA qualification."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from auditors.legacy_compute import validate_contract, arithmetic, fma_reference, resolve_iterations
spec = importlib.util.spec_from_file_location("legacy_generator", BASE / "probes/generate_legacy_compute.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class LegacyComputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contracts = {name: json.loads((BASE/"contracts"/(name+".json")).read_text())
                         for name in generator.FAMILIES}

    def test_all_contracts_and_source_generation(self):
        expected = {"legacy_fma": (93,24), "legacy_mma": (55,16), "legacy_wgmma": (85,25)}
        for name, contract in self.contracts.items():
            validate_contract(contract)
            self.assertEqual((len(contract["cases"]),len({generator.symbol(c) for c in contract["cases"]})),expected[name])
            self.assertEqual(generator.generate(name), (BASE/"probes"/(name+".cu")).read_text())
        total = [c["legacy"]["mapping_id"] for v in self.contracts.values() for c in v["cases"] if c.get("legacy")]
        self.assertEqual(len(total), 229)
        self.assertEqual(len(set(total)), 229)

    def test_changed_shape_chain_batch_and_zero_work_rejected(self):
        for name in ("legacy_fma","legacy_mma","legacy_wgmma"):
            original = self.contracts[name]
            for field in ("chains","batch"):
                changed = copy.deepcopy(original)
                changed["cases"][0]["parameters"][field] *= 2
                with self.assertRaises(ValueError):
                    validate_contract(changed)
        for contract in self.contracts.values():
            for c in contract["cases"]:
                demand = arithmetic(c, 3)
                if c["parameters"]["phase"] == "empty_control":
                    self.assertEqual(demand["work"],0)
                else:
                    self.assertGreater(demand["work"],0)

    def test_resolver_endpoints_and_wrong_pilot(self):
        device={"schema_version":2,"type":"device","cc":"9.0","name":"NVIDIA GH200 120GB",
                "uuid":"GPU-43269fbc-449d-3e0f-908a-9c81229546d3","sms":132,
                "driver_version":13010,"runtime_version":12090}
        for contract in self.contracts.values():
            case=next(c for c in contract["cases"] if c["iteration_policy"]["kind"]=="calibrated")
            policy=case["iteration_policy"]
            for ms in (0.001,0.01,0.5,100,200,1000):
                row={"case_id":case["id"],"iterations":8192,"event_ms":ms}
                r=resolve_iterations(row,case,device)
                expected=max(8192,min(policy["max_iterations"],int(8192*100/max(ms,.01))))
                self.assertEqual(r["resolved_iterations"],expected)
            with self.assertRaises(ValueError):
                resolve_iterations({"case_id":case["id"],"iterations":128,"event_ms":1},case,device)

    def test_fragment_coordinates_cover_logical_matrices(self):
        for m,n,k in ((16,8,16),(16,8,8),(8,8,4)):
            coords=[]
            for lane in range(32):
                for e in range(m*n//32):
                    row=lane//4 if m==8 else lane//4+8*(e//2)
                    col=2*(lane%4)+(e if m==8 else e%2)
                    coords.append((row,col))
            self.assertEqual(set(coords),{(i,j) for i in range(m) for j in range(n)})
            self.assertEqual(len(coords),len(set(coords)))
        a=[];d=[]
        for t in range(128):
            for e in range(32):
                rc=(16*(t//32)+(t%32)//4+8*((e//2)%2),2*(t%4)+e%2+8*(e//4))
                d.append(rc)
                if e<8:a.append(rc)
        self.assertEqual(set(d),{(i,j) for i in range(64) for j in range(64)})
        self.assertEqual(set(a),{(i,j) for i in range(64) for j in range(16)})
        offsets=[(row%8)*8+(row//8)*64+k%8+(k//8)*512 for row in range(64) for k in range(16)]
        self.assertEqual(sorted(offsets),list(range(1024)))

    def test_actual_cpp_reference_against_independent_integer_oracle(self):
        compiler=shutil.which("c++")
        if not compiler:self.skipTest("host C++ compiler unavailable; no runtime claim")
        with tempfile.TemporaryDirectory(prefix="gh200-reference-test-") as tmp:
            d=Path(tmp)
            (d/"driver.cpp").write_text(r'''
#include <iostream>
#include <iomanip>
#include "legacy_compute_reference.hpp"
int main() {
  std::string kind;int t,c,l,it,b;unsigned seed;bool varied;
  std::cout<<std::setprecision(17);
  while(std::cin>>kind>>t>>c>>l>>seed>>it>>b>>varied)
    std::cout<<legacy_reference::fma(kind,t,c,l,seed,it,b,varied)<<"\n";
}
''')
            subprocess.run([compiler,"-std=c++17","-O2","-I",str(BASE/"common"),
                            str(d/"driver.cpp"),"-o",str(d/"driver")],check=True,capture_output=True,timeout=30)
            rng=random.Random(20261001);rows=[]
            for kind in ("f32","f64","f16","f16x2","bf16","bf16x2"):
                for it in (0,1,2,128,2048,65536,1048576):
                    for varied in (False,True):
                        rows.append((kind,rng.randrange(256),rng.randrange(8),
                                     rng.randrange(2) if kind.endswith("x2") else 0,
                                     rng.choice((3,193,4294967295)),it,16,varied))
            stdin="\n".join(" ".join(str(int(x)) if type(x) is bool else str(x) for x in row) for row in rows)+"\n"
            result=subprocess.run([str(d/"driver")],input=stdin,text=True,capture_output=True,check=True,timeout=30)
            actual=[float(x) for x in result.stdout.splitlines()]
            self.assertEqual(len(actual),len(rows))
            for row,value in zip(rows,actual):
                self.assertEqual(value,fma_reference(*row),msg=str(row))


if __name__ == "__main__":
    unittest.main()
