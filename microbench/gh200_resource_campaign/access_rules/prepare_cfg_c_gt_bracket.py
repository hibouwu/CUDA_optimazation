#!/usr/bin/env python3
"""Prepare one fixed cfg_c GT-C-GT observer binary from the frozen tail source."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_MANIFEST_SHA = "11eab809368ab860736fab64d5871d21df712c1e00d5ab9eb76961bf9517d7db"
VERSION = "cfg-c-gt-bracket-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("source binding changed: " + old[:100])
    return text.replace(old, new, 1)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(base, output):
    manifest = base / "source_hashes.json"
    if sha(manifest) != BASE_MANIFEST_SHA:
        raise ValueError("wrong base source manifest")
    hashes = json.loads(manifest.read_text())
    for name, expected in hashes.items():
        if sha(base / name) != expected:
            raise ValueError("base source changed: " + name)
    if output.exists():
        raise ValueError("output must be new")
    output.mkdir(parents=True)
    shutil.copytree(base / "source", output / "source")
    (output / "build").mkdir()
    source = output / "source"
    shutil.copy2(HERE / "cfg_c_gt_bracket.cuh", source / "probes/cfg_c_gt_bracket.cuh")
    header = source / "probes/v06_trace.hpp"
    if (source / "overlay/v06_trace.hpp").read_text() != '#pragma once\n#include "../probes/v06_trace.hpp"\n':
        raise ValueError("active overlay include binding changed")
    text = header.read_text()
    text = replace_once(text, "#include <cstdint>", '''#include <cstdint>
#include "cfg_c_gt_bracket.cuh"
constexpr int CfgCGtShadowWords = 9; // 2 WG x 2 output ordinals x 2 events + overflow
__constant__ uint64_t* cfg_c_gt_shadow_ptr;
__constant__ unsigned cfg_c_gt_observer;''')
    old = '''  int role = threadIdx.x / 128;
  uint64_t c = v06_clock();
#ifdef R18_DUAL_CLOCK
  uint64_t ns = v06_ns();
#endif'''
    new = '''  int role = threadIdx.x / 128;
  uint64_t c, ns, before=0;
  bool bracket = cfg_c_gt_observer && (event==V06_FIRST_MMA || event==V06_MAIN_END);
  if (bracket) {
    auto stamp=cfg_c_gt_bracket();
    before=stamp.before; c=stamp.cycles; ns=stamp.after;
  } else { c=v06_clock(); ns=v06_ns(); }
  if (bracket) {
    unsigned block=blockIdx.x+gridDim.x*(blockIdx.y+gridDim.y*blockIdx.z);
    auto shadow=cfg_c_gt_shadow_ptr+size_t(block)*CfgCGtShadowWords;
    if (tile<0 || tile>=2 || role<1 || role>2) shadow[8]=1;
    else shadow[((role-1)*2+tile)*2+event]=before;
  }'''
    text = replace_once(text, old, new)
    header.write_text(text)
    probe = source / "probes/r18.cu"
    text = probe.read_text()
    text = text.replace("check-producer-tail", "check-gt-bracket")
    text = text.replace('"producer_tail_check"', '"gt_bracket_check"')
    text = replace_once(text, 'kTraceVersion="cfg-c-producer-tail-v2"', 'kTraceVersion="'+VERSION+'"')
    text = replace_once(text, "  unsigned producer_tail_observer = 0;", "  unsigned gt_bracket_observer = 0;\n  unsigned producer_tail_observer = 0;")
    text = replace_once(text, '      else if(key=="--producer-tail-observer")', '      else if(key=="--gt-bracket-observer")gt_bracket_observer=std::stoul(v);\n      else if(key=="--producer-tail-observer")')
    text = replace_once(text, "o.producer_tail_observer>1 ||", "o.producer_tail_observer!=0 || o.gt_bracket_observer>1 ||")
    text = replace_once(text, '"producer-tail observer requires fixed cfg_c geometry and history"', '"GT bracket observer requires fixed cfg_c geometry and history"')
    text = replace_once(text, "    unsigned producer_tail_off=0;", "    unsigned gt_bracket_off=0;\n    CUDA_CHECK(cudaMemcpyToSymbol(cfg_c_gt_observer,&gt_bracket_off,sizeof(unsigned)));\n    unsigned producer_tail_off=0;")
    text = replace_once(text, "    DeviceBuffer<uint64_t> trace(size_t(ctas) * kTraceWords);", "    DeviceBuffer<uint64_t> trace(size_t(ctas) * kTraceWords);\n    DeviceBuffer<uint64_t> gt_shadow(size_t(ctas)*CfgCGtShadowWords);\n    CUDA_CHECK(cudaMemcpyToSymbol(cfg_c_gt_shadow_ptr,&gt_shadow.pointer,sizeof(gt_shadow.pointer)));\n    CUDA_CHECK(cudaMemset(gt_shadow.pointer,0,gt_shadow.count*8));")
    text = replace_once(text, "    auto timed = [&](typename Gemm::Params& params, bool use_plain_prefix=false) {", "    auto timed = [&](typename Gemm::Params& params, bool use_plain_prefix=false) {\n      CUDA_CHECK(cudaMemset(gt_shadow.pointer,0,gt_shadow.count*8));")
    text = replace_once(text, "    auto set_target_epilogue = [&]() {", "    auto set_target_epilogue = [&]() {\n      CUDA_CHECK(cudaMemcpyToSymbol(cfg_c_gt_observer,&o.gt_bracket_observer,sizeof(unsigned)));")
    check = '''        if (o.mode == "check-gt-bracket") {
          CUDA_CHECK(cudaMemcpyToSymbol(r18_producer_tail_observer,&value,sizeof(value)));'''
    text = replace_once(text, check, check.replace("r18_producer_tail_observer", "cfg_c_gt_observer"))
    metadata = '                << ",\\\"producer_tail_observer\\\":" << o.producer_tail_observer'
    if text.count(metadata) != 2:
        raise ValueError("setup/call metadata sites changed")
    extra = '''                << ",\\\"gt_bracket_protocol\\\":\\\"cfg-c-gt-bracket-v1\\\""
                << ",\\\"gt_bracket_observer\\\":" << o.gt_bracket_observer
                << ",\\\"gt_bracket_prefix_flag\\\":0,\\\"gt_shadow_words_per_cta\\\":9"
                << ",\\\"gt_shadow_bytes\\\":" << gt_shadow.count*8
'''
    text = text.replace(metadata, extra+metadata)
    text = replace_once(text, '      print_u64("trace", host);', '''      print_u64("trace", host);
      std::vector<uint64_t> gt_host(gt_shadow.count);
      CUDA_CHECK(cudaMemcpy(gt_host.data(),gt_shadow.pointer,gt_shadow.count*8,cudaMemcpyDeviceToHost));
      print_u64("gt_shadow",gt_host);''')
    probe.write_text(text)
    commands = json.loads((base / "build/commands.json").read_text())
    write_json(output / "build/commands.json", commands)
    shutil.copy2(base / "run.sh", output / "run.sh")
    shutil.copy2(Path(__file__), source / Path(__file__).name)
    write_json(output / "source_hashes.json", {str(p.relative_to(output)):sha(p) for p in sorted(source.rglob("*")) if p.is_file()})
    write_json(output / "contract.json", dict(version=VERSION, base_manifest_sha256=BASE_MANIFEST_SHA,
        source_files_verified=len(hashes), fixed_grid=[66,2,1], shadow_words_per_cta=9,
        shadow_bytes=132*9*8, original_cta_words=1296, prefix_calls=16, prefix_flag=0,
        target_flags=[0,1], events=["FIRST_MMA","MAIN_END"], writer_threads=[128,256],
        shadow_index="((role-1)*2+output_ordinal)*2+event; slot 8 overflow",
        original_fields="cycles and GT_after retain original event slots; GT_before in separate shadow",
        invalid="zero prerequisite returns three zeros; no retry; zero/nonmonotonic/overflow rejects record",
        qualification=dict(source_prepared=True,compiled=False,sass_dependency_checked=False,
            hardware_sampling_enclosure=False,physical_ns_accuracy=False,correctness=False,
            observer_perturbation=False,null_gate=False,prediction=False),
        calibration="New binary identity. Old prefix rule cannot be applied as if its binary constraint matched."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.base, args.output)
