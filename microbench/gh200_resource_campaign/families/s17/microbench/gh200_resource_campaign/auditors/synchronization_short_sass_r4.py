"""Exact41 actual wrapper targets; structural roles remain independently reviewed."""
import hashlib,json,re
from common.suite_io import require
from auditors.synchronization_short_baseline_r4 import BASELINE
from auditors import synchronization_lowering_v3

def target_rows(text,symbol):
    matches=[p for p in re.split(r'Function\s*:\s*',text)[1:] if p.splitlines()[0].strip()==symbol];require(len(matches)==1,'S11 missing/duplicate exact role target '+symbol);rows=[]
    for line in matches[0].splitlines()[1:]:
        head=re.match(r'\s*/\*([0-9a-fA-F]+)\*/(.*?);',line)
        if head:rows.append({'pc':int(head[1],16),'instruction':' '.join(head[2].split()),'encoding':re.findall(r'/\* (0x[0-9a-fA-F]{16}) \*/',line)})
        elif rows:rows[-1]['encoding'].extend(re.findall(r'/\* (0x[0-9a-fA-F]{16}) \*/',line))
    require(rows and all(len(r['encoding'])==2 for r in rows),'S11 full128bit role instructions');return rows

def audit_sass(text,contract):
    from auditors.synchronization_short_r4 import LEGACY_BINDINGS,PAIR_BINDINGS
    expected={v['kernel_symbol'] for roles in LEGACY_BINDINGS.values() for v in roles.values()}|set(PAIR_BINDINGS)
    require(expected==set(BASELINE) and len(expected)==41,'S11 fixed39+2 roles')
    out=[]
    for symbol in sorted(expected):
        rows=target_rows(text,symbol);digest=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        require(len(rows)==BASELINE[symbol]['instruction_count'] and digest==BASELINE[symbol]['target_identity_sha256'],'S11 complete role/outlined helper identity drift '+symbol)
        require(not any(re.search(r'\b(?:LDL|STL)(?:\.|\s)',r['instruction']) for r in rows),'S11 local spill forbidden')
        out.append({'symbol':symbol,'instruction_count':len(rows),'target_identity_sha256':digest,'registers_per_thread':BASELINE[symbol]['registers_per_thread'],'static_smem_bytes':BASELINE[symbol]['static_smem_bytes'],'GPU_qualification':False})
    if contract.get('adapter_id')=='synchronization_lowering_v3':synchronization_lowering_v3.audit_sass(text,contract)
    warp=target_rows(text,'warp_divergent_pair_v2');cta=target_rows(text,'cta_divergent_pair_v2');ops=[r['instruction'] for r in warp];cops=[r['instruction'] for r in cta]
    require(sum('WARPSYNC.COLLECTIVE' in op for op in ops)==32 and sum('WARPSYNC.ALL' in op for op in ops)==1 and sum('BRA.DIV' in op for op in ops)==32,'S11 native outlined warp candidate lowering')
    require(sum('BAR.SYNC' in op for op in cops)==18,'S11 CTA same-task pair lowering')
    return {'status':'actual41_targets_match_candidate','targets':out,'hardware_qualification':False,'GPU_execution':False,'dynamic_pairs_per_iteration':8,'barriers_per_pair_per_participant':2,'static_count_is_not_dynamic_work':True}

def audit_resources(text):
    out={}
    for symbol,expected in BASELINE.items():
        marker="Compiling entry function '"+symbol+"'";require(text.count(marker)==1,'S11 exact compiled resource target')
        body=text.split(marker,1)[1].split('Compiling entry function',1)[0];require('0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in body,'S11 zero stack/spill')
        used=re.findall(r'Used (\d+) registers, used (\d+) barriers(?:, (\d+) bytes smem)?',body);require(len(used)==1,'S11 complete ptxas resources')
        regs,bars,shared=used[0];actual={'registers_per_thread':int(regs),'barriers':int(bars),'static_smem_bytes':int(shared or 0)}
        require(all(actual[k]==expected[k] for k in actual),'S11 perrole resources differ');out[symbol]=actual
    return out
