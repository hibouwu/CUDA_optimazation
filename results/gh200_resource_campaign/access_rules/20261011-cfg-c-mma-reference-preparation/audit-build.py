from pathlib import Path
import hashlib,json,re
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((root/'receipt.json').read_text());assert receipt['returncode']==0 and sha(root/'output.tar.gz')==receipt['sha256']
for name,digest in json.loads((root/'freeze.json').read_text())['files'].items():assert sha(root/name)==sha(root/'returned'/name)==digest
sources=json.loads((root/'source_hashes.json').read_text())
for name,digest in sources.items():assert sha(root/name)==digest
sass=root/'returned/build/reference.sass';sections=sass.read_text().split('Function : ')
main=next(s for s in sections if 'Li4ELb1ELb0EE' in s.splitlines()[0])
short=next(s for s in sections if 'Li1ELb1ELb0EE' in s.splitlines()[0])
instructions=[line.strip() for line in main.splitlines() if 'HGMMA.' in line]
assert len(instructions)==8 and all('HGMMA.64x128x16.F32' in x and '.tnspB' in x for x in instructions)
assert sum(' R88,' in x for x in instructions)==4 and sum(' R24,' in x for x in instructions)==4
assert '/*1650*/                   WARPGROUP.ARRIVE' in main and 'BRA 0x1650' in main
assert '/*1700*/                   WARPGROUP.DEPBAR.LE gsb0, 0x1' in main
assert '/*1720*/                   WARPGROUP.DEPBAR.LE gsb0, 0x0' in main
assert len([s for s in short.splitlines() if 'HGMMA.' in s])==2
log=(root/'returned/build/compile.log').read_text()
assert log.count('Used 154 registers, used 1 barriers, 1152 bytes smem')==2
assert log.count('0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads')==3
source=(root/'source/cfg_c_mma_reference.cu').read_text()
assert 'for (int i = 0; i < 16; ++i)' in source and 'for (int i = 0; i < 30; ++i)' not in source
assert source.index('  asm volatile("wgmma.wait_group.sync.aligned 0;')<source.index('  drain[threadIdx.x] = sum;')<source.index('    uint64_t end_cycle = clock64()')
result=dict(job_id=740094,gpu_executed=False,source_files_verified=len(sources),binary_sha256=sha(root/'returned/build/cfg_c_mma_reference'),sass_sha256=sha(sass),mainloop_hgmma=instructions,per_wg_instructions_per_group=8,wait_group=1,final_wait_group=0,registers=154,barriers=1,static_smem_bytes=1152,dynamic_smem_bytes=32768,spill_bytes=0,observation_qualified=False,numerical_verified=False,limits=['Compiler injects WARPGROUP.ARRIVE at loop head; retained in reference cost.','Stop endpoint follows accumulator checksum/drain and CTA sync, unlike GEMM MAIN; no intercept subtraction assumed.','Static build checks only; no same-card instruction timing measured.'])
(root/'build-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='mainloop_hgmma'}))
