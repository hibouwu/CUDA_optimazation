"""Finite actual731187 full128-bit identities and explicit cluster lifecycle roles."""
import hashlib,json,re
from common.suite_io import require
from auditors.cluster_dsm_sass_baseline_v1 import BASELINE

def target_rows(text,symbol):
 matches=[p for p in re.split(r'Function\s*:\s*',text)[1:] if p.splitlines()[0].strip()==symbol];require(len(matches)==1,'S17 unique complete target '+symbol);rows=[]
 for line in matches[0].splitlines()[1:]:
  head=re.match(r'\s*/\*([0-9a-fA-F]+)\*/(.*?);',line)
  if head:rows.append({'pc':int(head[1],16),'instruction':' '.join(head[2].split()),'encoding':re.findall(r'/\* (0x[0-9a-fA-F]{16}) \*/',line)})
  elif rows:rows[-1]['encoding'].extend(re.findall(r'/\* (0x[0-9a-fA-F]{16}) \*/',line))
 require(rows and [x['pc'] for x in rows]==list(range(0,len(rows)*16,16)) and all(len(x['encoding'])==2 for x in rows),'S17 contiguous full128-bit target')
 return rows

def identity(rows):return {'instruction_count':len(rows),'full_instructions_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()}

def backedges(rows):
 out=[]
 for x in rows:
  m=re.search(r'\bBRA(?:\.\S+)?\s+(0x[0-9a-fA-F]+)',x['instruction'])
  if m and int(m[1],16)<x['pc']:out.append((int(m[1],16),x['pc']))
 return out

def audit_target(rows,symbol):
 expected=BASELINE[symbol];require(all(identity(rows)[k]==expected[k] for k in ('instruction_count','full_instructions_sha256')),'S17 exact instructions/predicates/operands/complete machine encoding drift '+symbol)
 m=re.fullmatch(r's17_(local_read|dsm_read|local_write|dsm_write|cluster_sync|bulk_single|bulk_all)_c([248])',symbol);require(m is not None,'S17 finite target');form=m[1];c=int(m[2]);bulk=form.startswith('bulk_');write='write' in form
 pcs=lambda token:[x['pc'] for x in rows if token in x['instruction']]
 clock=pcs('SR_CLOCKLO');arv=pcs('UCGABAR_ARV');wait=pcs('UCGABAR_WAIT');bars=pcs('BAR.SYNC');edges=backedges(rows)
 require(len(clock)==2 and not any(re.search(r'\b(?:LDL|STL)(?:\.|\s)',x['instruction']) for x in rows),'S17 two CTA clock sites and no local spill')
 require(len(arv)==len(wait)==(4 if bulk or write else 3),'S17 init/item/exit cluster synchronization sites')
 require(arv[0]<wait[0]<clock[0]<clock[1]<arv[-1]<wait[-1],'S17 live cluster initialization and separate exit after timedloop')
 timed_arv=[p for p in arv if clock[0]<p<clock[1]];timed_wait=[p for p in wait if clock[0]<p<clock[1]]
 require(len(timed_arv)==len(timed_wait)==(2 if bulk or write else 1),'S17 per-item collective stage count')
 enclosing=[e for e in edges if clock[0]<e[0]<=min(timed_arv) and max(timed_wait)<e[1]<clock[1]];require(enclosing,'S17 cluster stages remain inside timed item backedge');outer=min(enclosing,key=lambda e:e[1]-e[0])
 record={'symbol':symbol,'identity':identity(rows),'compiler_resources':{k:expected[k] for k in ('registers_per_thread','barriers','static_smem_bytes','local_size_bytes')},'primary_same_CTA_clock_sites':clock,'item_loop_backedge':list(outer),'initial_live_cluster_gate':[arv[0],wait[0]],'timed_cluster_gates':list(zip(timed_arv,timed_wait)),'independent_exit_gate':[arv[-1],wait[-1]],'static_site_count_is_not_dynamic_count':True,'GPU_numerical_qualified':False,'performance_qualified':False}
 if bulk:
  request=pcs('UBLKCP.S.G.MULTICAST');init=pcs('SYNCS.EXCH.64');expect=pcs('SYNCS.ARRIVE.TRANS64');phase=pcs('SYNCS.PHASECHK');inval=pcs('SYNCS.CCTL.IV');fence=pcs('FENCE.VIEW.ASYNC.S')
  require(len(request)==len(init)==len(expect)==len(inval)==1 and len(phase)==4 and len(fence)==3,'S17 one multicast/init/expect/invalidate, bounded token wait and three proxy sites')
  require(init[0]<max(p for p in fence if p<clock[0])<arv[0]<clock[0],'S17 init/generic reset publication before initial cluster gate')
  require(outer[0]<expect[0]<fence[-1]<timed_arv[0]<timed_wait[0]<request[0]<min(phase)<=max(phase)<timed_arv[1]<timed_wait[1]<outer[1],'S17 expect/publish/armed/issue/acquire/consumer-done/release sequence')
  publication=[p for p in bars if max(phase)<p<timed_arv[1]];require(len(publication)==2,'S17 acquire to CTA consumers then CTA consumer-done publication')
  require(wait[-1]<inval[0] and pcs('BPT.TRAP'),'S17 invalidate only after exit gate; bounded fatal timeout')
  # Fixed observed compiler control literal; no general ISA decoding claim.
  mask=(1<<c)-1 if form=='bulk_all' else 1<<(c-1);literal=(mask<<16)|0x400
  controls=[x for x in rows if x['pc']<request[0] and x['instruction']==f'UMOV UR6, 0x{literal:x}'];require(len(controls)==1 and request[0]-controls[0]['pc']==32,'S17 exact observed request control literal')
  retries=[e for e in edges if e[0]<=request[0]<e[1] and e[1]-e[0]<=80];require(len(retries)==1 and retries[0]!=outer,'S17 issuer election retry is separate from item repetition')
  consume=[p for p in pcs('STG.E') if publication[0]<p<publication[1]];require(consume,'S17 full trace consumer stores between acquire/consumer-done CTA gates')
  done=[x['pc'] for x in rows if timed_wait[1]<x['pc']<outer[1] and 'STG.E' in x['instruction'] and ('+0x28]' in x['instruction'] or '+0x2c]' in x['instruction'])];require(len(done)==2,'S17 actual consumer_done/release lifecycle stores after cluster done gate')
  token=[p for p in pcs('STG.E.64') if expect[0]<p<fence[-1]];require(len(token)==1,'S17 opaque arrival token preserved before armed publication')
  record.update(multicast_request_pc=request[0],selected_mask=mask,observed_control_literal_pc=controls[0]['pc'],observed_control_literal=f'0x{literal:x}',issuer_election_retry=list(retries[0]),arrival_and_expect_pc=expect[0],opaque_token_store_pc=token[0],receiver_wait_pcs=phase,CTA_consumer_publication_gates=publication,consumer_done_release_stores=done,mbarrier_init_pc=init[0],mbarrier_invalidate_pc=inval[0],proxy_publication_pcs=fence,dynamic_source_requests='G*I',dynamic_source_request_bytes='G*I*16384',dynamic_received_bytes=f'G*I*{mask.bit_count()}*16384',opaque_token='preserved domain-only; no guessed encoding')
 elif form!='cluster_sync':
  operations=pcs('ST.E.STRONG.SYS' if write else 'LD.E.STRONG.SYS');require(len(operations)==8 and outer[0]<min(operations)<=max(operations)<timed_arv[0],'S17 eight mapped self/remote lane operations within loop before collective gate')
  require(not pcs('UBLKCP') and not pcs('SYNCS.ARRIVE'),'S17 DSM form has no TMA/mbarrier work')
  if write:
   consumer=[p for p in pcs('LDS ') if timed_wait[0]<p<timed_arv[1]];require(len(consumer)==8,'S17 full local target tile consumed after producer gate before consumer-done gate');record.update(target_consumer_read_pcs=consumer)
  record.update(mapped_access_opcode='ST.E.STRONG.SYS' if write else 'LD.E.STRONG.SYS',mapped_access_pcs=operations,local_is_mapped_self_comparator=True,dynamic_requested_bytes='G*C*I*4096',cluster_stages_per_item=2 if write else 1)
 else:
  require(not pcs('UBLKCP') and not pcs('LD.E.STRONG.SYS') and not pcs('ST.E.STRONG.SYS'),'S17 synchronization form no transport');record.update(dynamic_collective_stages='G*I',participants_per_cluster=c)
 return record

def audit_sass(text,contract):
 require('arch = sm_90a' in text,'S17 actual sm90a image');symbols=[p.splitlines()[0].strip() for p in re.split(r'Function\s*:\s*',text)[1:]];require(len(symbols)==21 and set(symbols)==set(BASELINE),'S17 exact21 full target set')
 expected={f'{f}_c{c}_{scope}' for f in ('local_read','dsm_read','local_write','dsm_write','cluster_sync','bulk_single_target','bulk_all_targets') for c in (2,4,8) for scope in ('one_cluster','cluster_grid')};require(len(contract['cases'])==42 and {x['id'] for x in contract['cases']}==expected,'S17 finite42 coordinate contract')
 return [audit_target(target_rows(text,s),s) for s in sorted(BASELINE)]

def audit_compile_resources(text):
 names=re.findall(r"Compiling entry function '([^']+)'",text);require(len(names)==21 and set(names)==set(BASELINE),'S17 exact21 ptxas entries');out={}
 for s,e in BASELINE.items():
  body=text.split("Compiling entry function '"+s+"'",1)[1].split('Compiling entry function',1)[0];require('0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in body,'S17 zero stack/spill')
  entries=re.findall(r'Used (\d+) registers, used (\d+) barriers(?:, (\d+) bytes smem)?',body);require(len(entries)==1,'S17 unique resources');reg,bar,smem=entries[0];a={'registers_per_thread':int(reg),'barriers':int(bar),'static_smem_bytes':int(smem or 0),'local_size_bytes':0};require(all(a[k]==e[k] for k in a),'S17 exact actual resource drift');out[s]=a
 return out
