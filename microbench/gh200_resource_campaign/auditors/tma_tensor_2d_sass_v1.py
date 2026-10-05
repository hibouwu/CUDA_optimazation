"""S15 actual731028 candidate baseline; independent B2 review still required."""
import hashlib,json,re
from common.suite_io import require
from auditors.tma_tensor_2d_sass_baseline_v1 import BASELINE

def target_rows(text,symbol):
    parts=re.split(r'Function\s*:\s*',text)[1:];matches=[part for part in parts if part.splitlines()[0].strip()==symbol]
    require(len(matches)==1,'S15 missing/duplicate target '+symbol);rows=[]
    for line in matches[0].splitlines()[1:]:
        head=re.match(r'\s*/\*([0-9a-fA-F]+)\*/(.*?);',line)
        if head:rows.append({'pc':int(head[1],16),'instruction':' '.join(head[2].split()),'encoding':re.findall(r'/\* (0x[0-9a-fA-F]{16}) \*/',line)})
        elif rows:rows[-1]['encoding'].extend(re.findall(r'/\* (0x[0-9a-fA-F]{16}) \*/',line))
    require(rows and all(len(row['encoding'])==2 for row in rows),'S15 complete128bit instructions '+symbol)
    return rows

def identity(rows):return {'instruction_count':len(rows),'full_instructions_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()}

def audit_sass(text,contract):
    expected={f'tt_{"g2s" if c["parameters"]["direction"]=="gmem_to_smem" else "s2g"}_q{c["parameters"]["payload_bytes"]}_{"sw128" if c["parameters"]["swizzle"]=="SW128" else "none"}' for c in contract['cases']}
    require(len(contract['cases'])==68 and expected==set(BASELINE),'S15 original68/22 targets')
    out=[]
    for symbol in sorted(expected):
        rows=target_rows(text,symbol);require(identity(rows)==BASELINE[symbol],'S15 actual instruction/predicate/operand/encoding drift '+symbol)
        ops=[(r['pc'],r['instruction']) for r in rows];g2s='_g2s_' in symbol
        tma=[(pc,op) for pc,op in ops if 'UTMALDG.2D' in op or 'UTMASTG.2D' in op]
        require(len(tma)==1 and tma[0][1].startswith('UTMALDG.2D' if g2s else 'UTMASTG.2D'),'S15 one native2D request per target loop')
        pc=tma[0][0];bars=[at for at,op in ops if op.startswith('BAR.SYNC')];timers=[at for at,op in ops if 'SR_GLOBALTIMERLO' in op]
        require(any(at<pc for at in bars) and any(at>pc for at in bars) and len(timers)==(4 if g2s else 2),'S15 timing/CTA gates; internal timeout reads separated')
        require(not any(re.search(r'\b(?:LDL|STL)(?:\.|\s)',op) for _,op in ops),'S15 no descriptor local copy/spill')
        require(any('MEMBAR.ALL.CTA' in op for _,op in ops),'S15 shared generic/async proxy publication')
        if g2s:
            expect=[at for at,op in ops if 'SYNCS.ARRIVE.TRANS64.RED.A0TR' in op];arrive=[at for at,op in ops if 'SYNCS.ARRIVE.TRANS64.A1T0' in op];wait=[at for at,op in ops if 'SYNCS.PHASECHK' in op];invalidate=[at for at,op in ops if 'SYNCS.CCTL.IV' in op]
            require(len(expect)==len(arrive)==len(invalidate)==1 and expect[0]<pc<arrive[0]<min(wait)<invalidate[0],'S15 expect/request/token/acquire/invalidate order')
            require(any('BPT.TRAP' in op for _,op in ops),'S15 fatal timeout path')
        else:
            commit=[at for at,op in ops if op=='UTMACMDFLUSH'];wait=[at for at,op in ops if op=='DEPBAR.LE SB0, 0x0'];full=[at for at,op in ops if op=='CCTL.IVALL'];fence=[at for at,op in ops if op=='FENCE.VIEW.ASYNC.G']
            require(len(commit)==len(wait)==len(full)==1 and pc<commit[0]<wait[0]<full[0] and len(fence)==2 and fence[0]<pc<full[0]<fence[1],'S15 store fullwait and global reset/consumer proxy fences')
        out.append({'symbol':symbol,'identity':identity(rows),'tensor_request_pc':pc,'globaltimer_pcs':timers,'CTA_barrier_pcs':bars,'independent_B2_required':True,'GPU_qualification':False})
    return out

def audit_compile_resources(text):
    resources={}
    for symbol in BASELINE:
        marker="Compiling entry function '"+symbol+"'";require(text.count(marker)==1,'S15 actual compile target multiplicity')
        body=text.split(marker,1)[1].split('Compiling entry function',1)[0]
        require('0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in body,'S15 actual no-stack/no-spill')
        used=re.findall(r'Used (\d+) registers, used (\d+) barriers(?:, (\d+) bytes smem)?',body);require(len(used)==1,'S15 complete ptxas resources')
        regs,bars,shared=used[0];require(int(regs)==40 and int(bars)==1 and int(shared or 0)==0,'S15 actual40regs/1barrier/0static shared')
        resources[symbol]={'registers_per_thread':40,'barriers':1,'static_smem_bytes':0,'stack_bytes':0,'spill_store_bytes':0,'spill_load_bytes':0}
    return resources
