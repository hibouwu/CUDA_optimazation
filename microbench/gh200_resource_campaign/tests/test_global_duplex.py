"""S13 independent address/work/resource and conservative SASS mutation checks."""
import copy,json,re,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from auditors import global_duplex as audit
CONTRACT=json.loads((ROOT/'contracts/global_duplex.json').read_text())
DEVICE={'schema_version':2,'type':'device','cc':'9.0','name':'NVIDIA GH200','uuid':'GPU-00000000-0000-0000-0000-000000000000',
        'sms':132,'driver_version':13010,'runtime_version':12090,'l2_cache_bytes':62914560,'global_memory_bytes':102005473280,
        'registers_per_sm':65536,'smem_per_sm_bytes':233472,'smem_per_cta_optin_bytes':232448}


def sass_fixture(case):
    """Synthetic parser fixture only; never actual target-compile evidence."""
    p=case['parameters'];ops=[(0,'CS2R R0, SR_GLOBALTIMERLO'),(16,'BAR.SYNC 0x0'),(32,'MOV R60, RZ')]
    pc=48
    for i in range(p['read_requests_per_group']):
        reg=4+4*i;ops.append((pc,f'LDG.E.128.STRONG.{"SM" if p["read_cache_policy"]=="ca" else "GPU"} R{reg}, [R40]'));pc+=16
        ops.append((pc,f'IADD3 R60, R60, R{reg}, R{reg+1}'));pc+=16
        ops.append((pc,f'IADD3 R60, R60, R{reg+2}, R{reg+3}'));pc+=16
    for i in range(p['write_requests_per_group']):ops.append((pc,f'STG.E.128.STRONG.SM [R42], R{4 if p["mode"]=="copy" else 20+4*i}'));pc+=16
    ops.append((pc,'@P0 BRA 0x30'));pc+=16
    ops.append((pc,'@P1 BRA 0x20'));pc+=16
    ops.append((pc,'STS [R50], R60'));pc+=16
    if p['write_requests_per_group']:ops.append((pc,'MEMBAR.SC.GPU'));pc+=16
    ops.append((pc,'BAR.SYNC 0x0'));pc+=16
    ops.append((pc,'CS2R R0, SR_GLOBALTIMERLO'))
    return 'Function : '+audit.kernel_symbol(case)+'\n'+'\n'.join(f'/*{at:04x}*/ {op}; /* encoded */' for at,op in ops)


class GlobalDuplexTests(unittest.TestCase):
    def test_original18_case_work_geometry_and_uint64(self):
        audit.validate_contract(CONTRACT);audit.validate_device(DEVICE)
        for case in CONTRACT['cases']:
            for occupancy in (1,2,3,4,8):
                plan=audit.allocation(case,DEVICE,occupancy);p=case['parameters']
                quantum=plan['blocks']*256*16
                self.assertEqual(plan['array_bytes']%quantum,0)
                self.assertGreaterEqual(plan['array_bytes'],plan['requested_array_bytes'])
                self.assertEqual(plan['aggregate_array_bytes'],plan['array_bytes']*(int(p['read_requests_per_group']>0)+int(p['write_requests_per_group']>0)))
                if p['working_set_class']=='large' and p['read_requests_per_group']+p['write_requests_per_group']>=2:self.assertGreater(16*plan['array_bytes']*(p['read_requests_per_group']+p['write_requests_per_group']),2**32)
        for key,value in [('l2_cache_bytes',0),('registers_per_sm',0),('global_memory_bytes',0)]:
            with self.assertRaises(ValueError):audit.validate_device({**DEVICE,key:value})
        with self.assertRaises(ValueError):audit.allocation(CONTRACT['cases'][4],{**DEVICE,'l2_cache_bytes':1024},4)
        with self.assertRaises(ValueError):audit.allocation(CONTRACT['cases'][-1],{**DEVICE,'global_memory_bytes':1024},4)

    def test_owner_arithmetic_independent_of_issue_order(self):
        for T in (1,7,256):
            for groups in (1,2,3,17):
                size=T*groups*16
                for r in (0,1,2,4):
                    for n in (1,2,16):
                        seed=0xa5a5a5a5;thread=T-1
                        actual=audit.read_checksum(size,T,thread,r,n,seed)
                        values=[]
                        for group in range(groups):
                            for request in range(r):
                                vector=((group*r+request)*T+thread)%(size//16)
                                values.extend((17*(4*vector+lane)+seed)&0xffffffff for lane in range(4))
                        self.assertEqual(actual,(sum(values)*n)&0xffffffff)

    def test_compile_actual_device_wrapping_helper_on_CPU(self):
        text=(ROOT/'probes/global_duplex.cu').read_text()
        helper=text[text.index('template<int Ratio>'):text.index('template<int Reads')].replace('__device__','').replace('__forceinline__','')
        source='#include <iostream>\nnamespace gh {using u64=unsigned long long;}\n'+helper
        source+='\nint main(){unsigned long long g;int r;while(std::cin>>g>>r)for(unsigned long long j=0;j<g;++j)for(int i=0;i<r;++i)std::cout<<(r==1?gd_slot<1>(j,i,g):r==2?gd_slot<2>(j,i,g):gd_slot<4>(j,i,g))<<"\\n";}\n'
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'slot.cpp';binary=Path(temp)/'slot';path.write_text(source)
            subprocess.run(['g++','-std=c++17','-O2',str(path),'-o',str(binary)],check=True,timeout=20,capture_output=True)
            pairs=[(g,r) for g in (1,2,3,17,255) for r in (1,2,4)]
            run=subprocess.run([str(binary)],input=''.join(f'{g} {r}\n' for g,r in pairs),capture_output=True,text=True,check=True,timeout=10)
            expected=[(j*r+i)%g for g,r in pairs for j in range(g) for i in range(r)]
            self.assertEqual(list(map(int,run.stdout.splitlines())),expected)

    def test_resource_and_ownership_mutations_rejected(self):
        case=CONTRACT['cases'][-1];plan=audit.allocation(case,DEVICE,4)
        resource={'kernel_symbol':audit.kernel_symbol(case),'registers_per_thread':32,'static_smem_bytes':1024,'dynamic_smem_bytes':0,'local_size_bytes':0,'occupancy_limit_ctas_per_sm':4,'extensions':{k:v for k,v in plan.items() if k!='blocks'}}
        self.assertEqual(audit.validate_resources(resource,case,DEVICE),plan)
        for field,value in [('occupancy_limit_ctas_per_sm',9),('registers_per_thread',128),('static_smem_bytes',0),('local_size_bytes',4),('dynamic_smem_bytes',1024)]:
            changed=copy.deepcopy(resource);changed[field]=value
            with self.assertRaises(ValueError):audit.validate_resources(changed,case,DEVICE)
        for field,value in [('mode','copy'),('read_requests_per_group',3),('vector_bytes',4),('write_cache_policy','wt')]:
            changed=copy.deepcopy(case);changed['parameters'][field]=value
            with self.assertRaises(ValueError):audit.case_identity(changed)

    def test_synthetic_sass_width_ratio_loop_fence_and_drain_rejection(self):
        for case in CONTRACT['cases'][:9]:
            text=sass_fixture(case);result=audit.audit_sass(text,{'cases':[case]})
            self.assertTrue(result[0]['requires_independent_actual_cache_and_dataflow_review'])
            mutations=[text.replace('.128','.64',1),text.replace('BAR.SYNC 0x0','NOP',1),
                       text.replace('STS [R50], R60','NOP'),text.replace('@P0 BRA 0x30','NOP')]
            if case['parameters']['read_requests_per_group']:mutations.append(text.replace('LDG.E.128.STRONG.GPU','LDG.E.128.STRONG.SM') if case['parameters']['read_cache_policy']=='cg' else text.replace('LDG.E.128.STRONG.SM','LDG.E.128.STRONG.GPU'))
            if case['parameters']['read_requests_per_group']:mutations.append(text.replace('IADD3 R60, R60, R6, R7','IADD3 R60, R60, R6, RZ'))
            if case['parameters']['mode']=='copy':mutations.append(text.replace('STG.E.128.STRONG.SM [R42], R4','STG.E.128.STRONG.SM [R42], R20'))
            if case['parameters']['mode']=='independent':mutations.append(text.replace('STG.E.128.STRONG.SM [R42], R20','STG.E.128.STRONG.SM [R42], R4'))
            if case['parameters']['write_requests_per_group']:mutations.append(text.replace('MEMBAR.SC.GPU','NOP'))
            for changed in mutations:
                with self.assertRaises(ValueError):audit.audit_sass(changed,{'cases':[case]})
            token='LDG' if case['parameters']['read_requests_per_group'] else 'STG'
            with self.assertRaises(ValueError):audit.audit_sass(text.replace(token+'.E.128','NOP',1),{'cases':[case]})

    def test_actual_cuda129_sass_and_semantic_mutations(self):
        actual=ROOT.parents[1]/'results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/target-compile-global-duplex-a/diagnostics/global_duplex/sass.txt'
        text=actual.read_text();evidence=audit.audit_sass(text,CONTRACT)
        self.assertEqual(len(evidence),18);self.assertEqual(len({x['symbol'] for x in evidence}),9)
        self.actual_mutations=[]
        for case,entry in zip(CONTRACT['cases'][:9],evidence[:9]):
            body=next(f for f in re.split(r'Function\s*:\s*',text)[1:] if f.splitlines()[0].strip()==entry['symbol'])
            def changed_line(pc,mutate):
                lines=body.splitlines();chosen=next(line for line in lines if f'/*{pc:04x}*/' in line)
                return text.replace(body,body.replace(chosen,mutate(chosen),1),1)
            mutants=[]
            memory=(entry['accesses']['LDG'] or entry['accesses']['STG'])[0]['pc']
            mutants.append(('wrong_width',changed_line(memory,lambda line:line.replace('.128','.64',1))))
            mutants.append(('deleted_access',changed_line(memory,lambda line:re.sub(r'(?:LDG|STG)\.E\.128\.STRONG\.(?:SM|GPU)','NOP',line))))
            mutants.append(('missing_drain',changed_line(entry['drain'][0][0],lambda line:line.replace('STS','NOP',1))))
            mutants.append(('never_executed_drain',changed_line(entry['drain'][0][0],lambda line:line.replace('STS','@!PT STS',1))))
            if entry['accesses']['STG']:
                mutants.append(('never_executed_store',changed_line(entry['accesses']['STG'][0]['pc'],lambda line:line.replace('STG.E.128','@!PT STG.E.128',1))))
            if entry['accesses']['LDG']:
                mutants.append(('never_executed_load',changed_line(entry['accesses']['LDG'][0]['pc'],lambda line:line.replace('LDG.E.128','@!PT LDG.E.128',1))))
            mutants.append(('missing_group_loop',changed_line(entry['group_loop'][1],lambda line:line.replace('BRA','NOP',1))))
            # The copy dependency is verified through exact lane origins. Replacing
            # only the source register preserves counts, widths and addresses.
            if case['parameters']['mode']=='copy':
                store=entry['accesses']['STG'][0]['pc']
                mutants.append(('copy_no_loaded_data',changed_line(store,lambda line:re.sub(r', R[0-9]+ ;',', RZ ;',line))))
            if entry['accesses']['LDG']:
                load=entry['accesses']['LDG'][0]['pc']
                mutants.append(('wrong_cache',changed_line(load,lambda line:line.replace('.STRONG.GPU','.STRONG.SM') if '.STRONG.GPU' in line else line.replace('.STRONG.SM','.STRONG.GPU'))))
                mutants.append(('loaded_values_not_drained',changed_line(entry['drain'][0][0],lambda line:re.sub(r', R[0-9]+ ;',', RZ ;',line))))
            if entry['fences']:
                mutants.append(('missing_write_fence',changed_line(entry['fences'][0][0],lambda line:line.replace('MEMBAR.SC.GPU','NOP'))))
                mutants.append(('CTA_scope_write_fence',changed_line(entry['fences'][0][0],lambda line:line.replace('MEMBAR.SC.GPU','MEMBAR.SC.CTA'))))
                mutants.append(('predicated_write_fence',changed_line(entry['fences'][0][0],lambda line:line.replace('MEMBAR.SC.GPU','@P0 MEMBAR.SC.GPU'))))
            if case['parameters']['mode']=='independent':
                store=entry['accesses']['STG'][0]['pc']
                drain_reg=entry['drain'][0][1].rsplit(',',1)[1].strip()
                mutants.append(('store_uses_read_sum',changed_line(store,lambda line:re.sub(r', R[0-9]+ ;',', '+drain_reg+' ;',line))))
            for reason,mutant in mutants:
                with self.subTest(case=case['id'],reason=reason):
                    self.assertNotEqual(text,mutant)
                    with self.assertRaises(ValueError):audit.audit_sass(mutant,{'cases':[case]})
                    self.actual_mutations.append({'case_id':case['id'],'mutation':reason,'status':'rejected'})


if __name__=='__main__':unittest.main()
