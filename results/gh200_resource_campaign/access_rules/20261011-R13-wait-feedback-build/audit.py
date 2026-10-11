"""Bind the CPU build and inspect the single mask-selected wait path."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import tarfile

root=Path(__file__).resolve().parent
run=root/'attempts/prepare2';returned=run/'returned'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(run/'source_hashes.json')=='ab9dc741f4e72da413833b6e984b58054e9c37d05232e6f88bbfcfdb0b334b9e'
hashes=json.loads((run/'source_hashes.json').read_text())
for name,digest in hashes.items():assert sha(run/name)==digest
assert (returned/'source_hashes.json').read_bytes()==(run/'source_hashes.json').read_bytes()
assert sha(run/'output.tar.gz')=='966dd267a5ff54c22df59d6c34b8e209236957dab20fbd4b48b6739236876fd7'
assert json.loads((run/'receipt.json').read_text())['returncode']==0
allocation=(returned/'allocation.txt').read_text()
assert 'AllocTRES=cpu=4,mem=16G,node=1,billing=4' in allocation and 'JOB_GRES=(null)' in allocation
assert 'V12.9.41' in (returned/'compiler.txt').read_text()
records=[]
with tarfile.open(run/'output.tar.gz') as archive:
    for stage,name,H,B,slow,end in [(4,'cfg_b_s4_stamped',0x1ed0,0x1f80,0x5a40,0x22c0),
                                  (6,'cfg_b_stamped',0x1dd0,0x1e80,0x5960,0x21c0)]:
        for suffix in ('','.sass','.resources','.log'):
            path='build/'+name+suffix
            assert archive.extractfile('./'+path).read()==(returned/path).read_bytes()
        instructions={};active=False
        for line in (returned/'build'/f'{name}.sass').read_text().splitlines():
            if 'Function :' in line:
                if active:break
                active='_ZN7cutlass13device_kernel' in line
            match=re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\*',line)
            if active and match:instructions[int(match[1],16)]=match[2]
        assert instructions[H]=='@!P3 CS2R R180, SR_CLOCKLO'
        assert instructions[B]=='CS2R R182, SR_CLOCKLO'
        group_register='UR17' if stage==4 else 'UR21'
        assert instructions[H-0x20]==f'UISETP.GT.U32.OR UP0, UPT, {group_register}, 0x3f, !UP0'
        assert 'UR29' in instructions[H-0x40] # dynamic mask participates in bit selection
        assert instructions[H+0x50].startswith('SYNCS.PHASECHK.TRANS64.TRYWAIT P3,')
        assert instructions[H+0x70]==instructions[H+0x50]
        assert instructions[H+0x60]==f'@P3 BRA {H+0x90:#x}'
        assert instructions[H+0x80]==f'@!P3 BRA {slow:#x}'
        assert instructions[B-0x10]==f'@!P2 BRA {end:#x}'
        assert instructions[slow+0x20].startswith('SYNCS.PHASECHK.TRANS64.TRYWAIT P3,')
        assert instructions[slow+0x30].startswith('@!P3 NANOSLEEP.SYNCS')
        assert instructions[slow+0x40].startswith('@!P3 SYNCS.PHASECHK.TRANS64')
        assert instructions[slow+0x50]==f'@!P3 BRA {slow+0x10:#x}'
        assert instructions[slow+0x60]==f'BRA {H+0x90:#x}'
        assert instructions[end]=='BSYNC B1' and instructions[end+0x20]=='WARPGROUP.ARRIVE'
        body=[(a,i) for a,i in instructions.items() if H<=a<=end or slow<=a<=slow+0x60]
        assert not any(re.search(r'\b(?:STG|STL|STS|LDL|HGMMA)(?:\.|\s)',i) for a,i in body)
        assert [a for a,i in body if 'SR_CLOCKLO' in i]==[H,B]
        assert sum('HGMMA.' in i for i in instructions.values())==16
        assert any('STG.E.64' in i and '+0x148]' in i for i in instructions.values())
        assert any('STG.E.64' in i and '+0x150]' in i for i in instructions.values())
        assert 'REG:168 STACK:0 SHARED:1024 LOCAL:0' in (returned/'build'/f'{name}.resources').read_text()
        spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',
                          (returned/'build'/f'{name}.log').read_text())
        assert spills and all(x==('0','0','0') for x in spills)
        excerpt=[(a,i) for a,i in instructions.items() if H-0xc0<=a<=end+0x20 or slow<=a<=slow+0x60]
        (root/f'stage{stage}-instructions.txt').write_text('\n'.join(f'{a:#06x} {i}' for a,i in excerpt)+'\n')
        records.append(dict(stage=stage,name=name,binary_sha256=sha(returned/'build'/name),
            sass_sha256=sha(returned/'build'/f'{name}.sass'),H=hex(H),B=hex(B),
            try_address=hex(H+0x50),wait_address=hex(H+0x70),slow_entry=hex(slow),
            collector_end=hex(end),static_B_through_collector_end=(end-B)//16+1,
            registers=168,spill_bytes=0,static_HGMMA=16))
tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','microbench/gh200_resource_campaign/access_rules',
    '-p','test_r13_wait_feedback.py','-v'],capture_output=True,text=True)
(root/'cpu-tests.json').write_text(json.dumps(dict(returncode=tests.returncode,stdout=tests.stdout,stderr=tests.stderr),indent=2)+'\n')
assert tests.returncode==0 and 'Ran 5 tests' in tests.stderr
report=dict(job=740367,CPU_only=True,CUDA='12.9.41',source_files_verified=len(hashes),
    source_manifest_sha256=sha(run/'source_hashes.json'),output_archive_sha256=sha(run/'output.tar.gz'),
    binaries=records,selected_clock_pair_shared_across_modes=True,collector_after_B_no_loop_stores=True,
    limits=['Same selected instruction addresses do not prove equal scheduling or background state.',
            'Group2 controls no prior selected observation; it is not a pure collector-cost subtraction.',
            'New binary and protocol are separate from the earlier cumulative-wait measurement.'],
    hardware_correctness=False,observation_qualified=False,prediction_qualified=False)
(root/'build-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
