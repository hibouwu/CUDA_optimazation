from pathlib import Path
import re,json,hashlib,subprocess
root=Path(__file__).resolve().parent
results=[]
for stage in (4,6):
 name=f'cfg_b_s{stage}_stamped';binary=root/'returned/build'/name
 elf=subprocess.check_output(['cuobjdump','--dump-elf',str(binary)],text=True)
 symbol=next(l for l in elf.splitlines() if re.search(r'0x6\s+0\s+0x4\s+0x1.*r13_release_warp_observer',l))
 sass=binary.with_suffix('.sass').read_text().split('Function :')[1]
 lines=[l.split(';')[0].strip()+';' for l in sass.splitlines() if re.match(r'\s*/\*[0-9a-f]+\*/\s+[^/]',l) and ';' in l]
 k=next(i for i,l in enumerate(lines) if 'UISETP.NE.AND' in l and ', 0x11,' in l)
 pre=next(i for i in range(k,len(lines)) if 'CS2R' in lines[i] and 'SR_CLOCKLO' in lines[i])
 arrival=next(i for i in range(pre,len(lines)) if 'SYNCS.ARRIVE' in lines[i])
 post=next(i for i in range(arrival,len(lines)) if 'CS2R' in lines[i] and 'SR_CLOCKLO' in lines[i])
 assert '@P2 CS2R' in lines[pre] and '@P2 CS2R' in lines[post]
 assert '@' not in lines[arrival] and 'RED.A1T0' in lines[arrival]
 assert not any('BRA ' in l or 'STG.' in l or 'STS.' in l or 'BAR.' in l for l in lines[pre:post+1])
 sel=next(i for i,l in enumerate(lines) if 'ULDC ' in l and 'c[0x3][0x0]' in l)
 results.append(dict(stage=stage,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),constant_symbol=symbol,
 selector_sass=lines[sel-1:sel+54],release_sass=lines[k-1:post+1],
 bracket_same_predicate=True,arrival_not_observer_predicated=True,no_branch_or_store_inside_pair=True,
 interpretation='Source and static issue-path evidence only; active lanes and physical completion are not measured.'))
(root/'static-release-audit.json').write_text(json.dumps(results,indent=2)+'\n')
print([(r['stage'],r['release_sass'][-1]) for r in results])
