"""Finite job731114 CUDA12.9/sm90a full-code identity and lifecycle checks.

Static request sites, uniform issuer retry lowering and dynamic item work are
separate. This module cannot grant GPU numerical or performance qualification.
"""
import hashlib,json,re
from common.suite_io import require
from auditors.tma_stage_request_sass_baseline_v1 import BASELINE

def target_rows(text,symbol):
    matches=[part for part in re.split(r'Function\s*:\s*',text)[1:] if part.splitlines()[0].strip()==symbol]
    require(len(matches)==1,'S16 missing/duplicate complete target '+symbol)
    rows=[]
    for line in matches[0].splitlines()[1:]:
        head=re.match(r'\s*/\*([0-9a-fA-F]+)\*/(.*?);',line)
        if head:rows.append({'pc':int(head[1],16),'instruction':' '.join(head[2].split()),'encoding':re.findall(r'/\* (0x[0-9a-fA-F]{16}) \*/',line)})
        elif rows:rows[-1]['encoding'].extend(re.findall(r'/\* (0x[0-9a-fA-F]{16}) \*/',line))
    require(rows and all(len(row['encoding'])==2 for row in rows),'S16 complete128bit code')
    return rows

def identity(rows):
    return {'instruction_count':len(rows),'full_instructions_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()}

def audit_target(rows,symbol):
    expected=BASELINE[symbol];require(all(identity(rows)[k]==expected[k] for k in ('instruction_count','full_instructions_sha256')),'S16 exact code/predicate/operand/encoding drift '+symbol)
    match=re.fullmatch(r'ts_(g2s|s2g)_s([124])_r([124])',symbol);require(match is not None,'S16 finite target');g=match[1]=='g2s';s=int(match[2]);r=int(match[3])
    ops=[(row['pc'],row['instruction']) for row in rows]
    pcs=lambda token:[pc for pc,op in ops if token in op]
    requests=pcs('UBLKCP.S.G' if g else 'UBLKCP.G.S');require(len(requests)==r,'S16 exactlyR static request sites per item body')
    require(not pcs('UBLKCP.G.S' if g else 'UBLKCP.S.G'),'S16 direction lowering')
    require(not any(re.search(r'\b(?:LDL|STL)(?:\.|\s)',op) for _,op in ops),'S16 no local token array/spill')
    clock=pcs('SR_CLOCKLO');require(len(clock)==2 and clock[0]<min(requests)<=max(requests)<clock[1],'S16 primary clock boundaries')
    bars=pcs('BAR.SYNC');shared_fences=pcs('FENCE.VIEW.ASYNC.S');require(bars and shared_fences and min(shared_fences)<clock[0],'S16 initial shared proxy publication')
    arrive=pcs('SYNCS.ARRIVE.TRANS64.A1T0');commit=pcs('UTMACMDFLUSH');end=arrive[0] if g and arrive else commit[0] if commit else -1
    edges=[]
    for pc,op in ops:
        branch=re.search(r'\bBRA(?:\.\S+)?\s+(0x[0-9a-fA-F]+)',op)
        if branch and int(branch[1],16)<pc:edges.append((int(branch[1],16),pc))
    enclosing=[edge for edge in edges if edge[0]<=min(requests) and max(requests)<end<edge[1]];require(enclosing,'S16 request/commit-or-arrive stays in an item backedge');outer=min(enclosing,key=lambda edge:edge[1]-edge[0])
    before=[pc for pc in bars if outer[0]<=pc<min(requests)];require(len(before)==4,'S16 steady completion publication/consumer-done/release/reset-publication CTA gates')
    after=[pc for pc in bars if end<pc<outer[1]];require(len(after)==1,'S16 issue publication CTA gate')
    record={'symbol':symbol,'identity':identity(rows),'resource':{k:expected[k] for k in ('registers_per_thread','barriers','static_smem_bytes')},'static_request_site_pcs':requests,'item_loop_backedge':list(outer),'primary_clock_pcs':clock,'steady_CTA_roles':dict(zip(('completion_publication','consumer_done','slot_release','destination_reset_publication'),before)),'issue_publication_pc':after[0],'dynamic_transport_requests':'I*R per CTA','dynamic_transport_bytes':'I*R*16384 per CTA','static_site_count_is_not_dynamic_count':True,'GPU_qualification':False}
    if g:
        init=pcs('SYNCS.EXCH.64');inval=pcs('SYNCS.CCTL.IV');expect=pcs('SYNCS.ARRIVE.TRANS64.RED.A0TR');phase=pcs('SYNCS.PHASECHK');require(len(init)==len(inval)==s and len(expect)==len(arrive)==1 and len(phase)==8 and not commit,'S16 Sbarriers/RQexpect/onearrive/two boundedpoll sites')
        require(max(init)<max(pc for pc in shared_fences if pc<clock[0])<clock[0] and clock[1]<min(inval),'S16 init-fence-publication and final invalidation after export')
        require(expect[0]<min(requests)<=max(requests)<arrive[0] and any(pc<outer[0] or outer[0]<=pc<before[0] for pc in phase),'S16 expect/request/arrive and acquire before consumer gate')
        tail=[pc for pc in phase if pc>outer[1]];require(len(tail)==4 and max(tail)<clock[1] and pcs('BPT.TRAP'),'S16 tail acquire and fatal bounded timeout')
        tailbars=[pc for pc in bars if max(tail)<pc<clock[1]];require(len(tailbars)>=4,'S16 tail consumers/consumer-done/release/final timing gate')
        record.update(mbarrier_init_pcs=init,mbarrier_invalidate_pcs=inval,expect_tx_pc=expect[0],arrive_token_pc=arrive[0],steady_acquire_pcs=[pc for pc in phase if pc<outer[1]],tail_acquire_pcs=tail,tail_CTA_roles=dict(zip(('completion_publication','consumer_done','slot_release','final_timing_gate'),tailbars[:4])),dynamic_arrivals='I per CTA',dynamic_expected_tx_bytes='I*R*16384 per CTA',token='opaque register result; no local spill or encoding assumption')
    else:
        waits=[(pc,op) for pc,op in ops if op.startswith('DEPBAR.LE SB0')];full=pcs('CCTL.IVALL');fences=pcs('FENCE.VIEW.ASYNC.G');require(len(commit)==1 and len(waits)==len(full)==2 and len(fences)==3,'S16 onecommit/two full-wait sites/three global proxy paths')
        require(waits[0][1]==f'DEPBAR.LE SB0, 0x{s-1:x}' and waits[1][1]=='DEPBAR.LE SB0, 0x0','S16 steady S-1 and tail0 group thresholds')
        require(waits[0][0]<full[0]<before[0]<min(requests)<commit[0]<outer[1]<waits[1][0]<full[1]<clock[1],'S16 full completion rather than source-read-only wait')
        tailbars=[pc for pc in bars if full[1]<pc<clock[1]];require(len(tailbars)>=5,'S16 tail drain/consumer/consumer-done/release/final timing gates')
        record.update(commit_pc=commit[0],steady_full_wait_pc=waits[0][0],tail_full_wait_pc=waits[1][0],full_visibility_pcs=full,global_proxy_pcs=fences,tail_CTA_roles=dict(zip(('tail_drain_publication','completion_publication','consumer_done','slot_release','final_timing_gate'),tailbars[:5])),dynamic_commits='I per CTA',dynamic_steady_full_wait_calls='max(0,I-S) per CTA',dynamic_tail_full_wait_calls='1 per CTA')
    return record

def audit_sass(text,contract):
    symbols=[part.splitlines()[0].strip() for part in re.split(r'Function\s*:\s*',text)[1:]]
    require(len(symbols)==18 and set(symbols)==set(BASELINE),'S16 exact18 complete target set')
    cases=contract['cases'];require(len(cases)==36,'S16 original36 nominal coordinates')
    expected={f'{direction}_16kib_s{s}_r{r}_{scope}' for direction in ('gmem_to_smem','smem_to_gmem') for s in (1,2,4) for r in (1,2,4) for scope in ('one_cta','all_gpu')};require({c['id'] for c in cases}==expected,'S16 finite coordinate set')
    return [audit_target(target_rows(text,symbol),symbol) for symbol in sorted(BASELINE)]

def audit_compile_resources(text):
    output={};names=re.findall(r"Compiling entry function '([^']+)'",text);require(len(names)==18 and set(names)==set(BASELINE),'S16 exact18 compiled resources')
    for symbol,expected in BASELINE.items():
        body=text.split("Compiling entry function '"+symbol+"'",1)[1].split('Compiling entry function',1)[0];require('0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in body,'S16 stack/spill forbidden')
        entries=re.findall(r'Used (\d+) registers, used (\d+) barriers(?:, (\d+) bytes smem)?',body);require(len(entries)==1,'S16 completeptxas resources');regs,bars,shared=entries[0];actual={'registers_per_thread':int(regs),'barriers':int(bars),'static_smem_bytes':int(shared or 0)};require(all(actual[k]==expected[k] for k in actual),'S16 exact pertarget resource drift');output[symbol]=actual
    return output
