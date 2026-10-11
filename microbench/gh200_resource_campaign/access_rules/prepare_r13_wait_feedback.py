"""Derive four runtime modes from the immutable cumulative-wait source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
BASE_SHA='b57bbb9852ce7da56f87ad5eac6be5412589e0d32c369e0499b41414a0b8021d'
VERSION='r13-wait-feedback-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('source anchor changed: '+old[:80])
    return text.replace(old,new,1)


def prepare(base,output):
    if sha(base/'source_hashes.json')!=BASE_SHA:raise ValueError('wrong base source manifest')
    hashes=json.loads((base/'source_hashes.json').read_text())
    for name,digest in hashes.items():
        if sha(base/name)!=digest:raise ValueError('changed base source: '+name)
    if output.exists():raise ValueError('new output required')
    output.mkdir(parents=True)
    shutil.copytree(base/'source',output/'source')
    (output/'build').mkdir()
    shutil.copyfile(base/'build/commands.json',output/'build/commands.json')
    source=output/'source'
    shutil.copyfile(HERE/'r13_wait_feedback.cuh',source/'probes/r13_wait_sum.cuh')
    header=source/'probes/v06_trace.hpp'
    assert header.read_bytes()==(source/'probes/r18_trace.hpp').read_bytes()
    assert (source/'overlay/v06_trace.hpp').read_text()=='#pragma once\n#include "../probes/v06_trace.hpp"\n'
    text=replace_once(header.read_text(),'b[31]=0x52313309;','b[31]=0x5231330a;')
    text=replace_once(text,'b[39]=s.wait_sum.invalid; b[40]=r13_wait_sum_observer;',
        'b[39]=s.wait_sum.invalid; b[40]=r13_wait_sum_observer;\n  b[41]=s.wait_sum.group2_cycles; b[42]=s.wait_sum.group16_cycles;')
    text=text.replace('r13_wait_sum_observer','r13_wait_sum_mask')
    header.write_text(text);(source/'probes/r18_trace.hpp').write_text(text)
    collective=source/'overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp'
    text=replace_once(collective.read_text(),
        'const bool r13_sum_selected = r13_sum_enabled && r13_k>=2 && r13_k<=16;',
        'const bool r13_sum_selected = r13_sum_enabled && r13_wait_feedback_selected(r13_wait_sum_mask, r13_k);')
    text=replace_once(text,'r13_target && r13_wait_sum_observer','r13_target && r13_wait_sum_mask')
    collective.write_text(text)
    probe=source/'probes/r18.cu';text=probe.read_text()
    assert text.count('r13-early-wait-sum-v1')==3
    text=text.replace('r13-early-wait-sum-v1',VERSION)
    text=replace_once(text,'o.wait_sum_observer>1','o.wait_sum_observer>3')
    text=replace_once(text,'unsigned wait_sum_off=0;',
        'unsigned wait_sum_mask_target=r13_wait_feedback_mask(o.wait_sum_observer);\n    unsigned wait_sum_off=0;')
    assert text.count('cudaMemcpyToSymbol(r13_wait_sum_observer,')==2
    text=text.replace('cudaMemcpyToSymbol(r13_wait_sum_observer,','cudaMemcpyToSymbol(r13_wait_sum_mask,')
    text=replace_once(text,'&o.wait_sum_observer,sizeof(unsigned)','&wait_sum_mask_target,sizeof(unsigned)')
    needle='<< ",\\\"wait_sum_observer\\\":" << o.wait_sum_observer'
    assert text.count(needle)==2
    text=text.replace(needle,needle+'\n                << ",\\\"wait_sum_selection_mask\\\":" << wait_sum_mask_target')
    probe.write_text(text)
    shutil.copyfile(Path(__file__),source/'prepare_r13_wait_feedback.py')
    (output/'source_hashes.json').write_text(json.dumps({str(p.relative_to(output)):sha(p)
        for p in sorted(source.rglob('*')) if p.is_file()},indent=2)+'\n')
    changed=[name for name,digest in hashes.items() if sha(output/name)!=digest]
    contract=dict(version=VERSION,base_source_manifest_sha256=BASE_SHA,base_source_files_verified=len(hashes),
        changed_base_files=changed,config='cfg_b',stages=[4,6],shape=[2816,1536,4096],
        cta=0,physical_thread=256,cta_output_ordinal=1,WG_local_output_ordinal=0,
        modes={'0':'no selected waits','1':'all local groups 2..16','2':'only group2','3':'only group16'},
        group_masks={'0':0,'1':0x1fffc,'2':4,'3':65536},prefix_calls=16,prefix_mode=0,
        original_progress_fields_preserved=True,trace_words_per_CTA=784,
        fields={34:'selected sum',35:'count',36:'mask',37:'first selected H',38:'last selected B',
                39:'invalid bits',40:'declared selection mask',41:'Q2 when selected',42:'Q16 when selected'},
        interval='Same iteration H before consumer_try_wait, B after consumer_wait; unchanged token path.',
        collector='All updates after B, final global flush after full consumer loop.',
        missingness='off Q/R absent; single-group sum is not Q_early; unselected Q2/Q16 absent, not zero.',
        contrast='Compare Q16(all) vs Q16(only16); group2 all vs only2 is a no-prior-collector control.',
        limits=['Selector/control instructions and register allocation are part of the new observer.',
                'Not the pure price of fourteen arithmetic updates; current-wait timestamp overhead remains.',
                'Different stages run different off-prefix kernels; no component ns conversion.'],
        compiled=False,hardware_correctness=False,observation_qualified=False,prediction_qualified=False)
    (output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    print(json.dumps(dict(source_manifest_sha256=sha(output/'source_hashes.json'),changed_base_files=changed),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.base.resolve(),args.output.resolve())
