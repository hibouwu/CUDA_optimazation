"""S14 independent full-word/lifecycle/coverage-model negatives, no GPU."""
import copy,hashlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from auditors import tma_bulk as audit
from auditors.tma_bulk_validation import validation_argv
CONTRACT=json.loads((ROOT/'contracts/tma_bulk.json').read_text())
PROFILES=json.loads((ROOT/'contracts/tma_bulk_validation_profiles_v1.json').read_text())['profiles']


def pair(values):return [piece for v in values for piece in (v&0xffffffff,v>>32)]


def arrays_for(case,profile,blocks=2):
    q=case['parameters']['payload_bytes'];words=q//4;g2s=case['parameters']['direction']=='gmem_to_smem'
    length=profile['target_iterations'][0];release=profile['role']=='source_release';seed=profile['required_seed']
    layouts=audit.artifact_layouts(case,profile,blocks,seed);arrays={name:{'shape':shape,'values':[]} for name,shape in layouts.items()}
    arrays['bulk_guards.u32le']['values']=list(audit.GUARDS)
    counts=[]
    for _ in range(blocks):counts.extend([length,length if g2s else 0,0,int(release),0 if g2s else length])
    arrays['bulk_completion.u32le']['values']=pair(counts)
    if release:
        arrays['bulk_release_clocks.u32le']['values']=pair([v for b in range(blocks) for v in (2**53+b*10+1,2**53+b*10+1,2**53+b*10+3)])
        arrays['bulk_source_after.u32le']['values']=[((29*(b*words+w)+seed)&0xffffffff)^0xffffffff for b in range(blocks) for w in range(words)]
    else:
        arrays['bulk_trace.u32le']['values']=[((17*((b*32+i%32)*words+w) if g2s else 29*(b*words+w))+seed)&0xffffffff for b in range(blocks) for i in range(length) for w in range(words)]
    if not g2s:
        visited=set(i%32 for i in range(length))
        arrays['bulk_ring.u32le']['values']=[((29*(b*words+w)+seed)&0xffffffff)^(0 if slot in visited else 0xffffffff) for b in range(blocks) for slot in range(32) for w in range(words)]
    return arrays


class TmaBulkTests(unittest.TestCase):
    def test_work_uint64_bounds_and_original_matrix(self):
        audit.validate_contract(CONTRACT)
        self.assertEqual(audit.work_count(1,65536,65536),4294967296)
        self.assertEqual(audit.allocation_bytes(4097,65536),(4097*32*65536)+32)
        for args in ((0,1,1024),(1,0,1024),(1,65537,1024),(1,1,2048),(2**63,2,65536)):
            with self.assertRaises(ValueError):audit.work_count(*args)
        for field,value in [('expected_arrivals',128),('expected_transaction_bytes_per_phase',128),('global_slots_per_cta',2),('source_alignment_bytes',4),('issuer_thread',1),('tensor_map',True)]:
            c=copy.deepcopy(CONTRACT['cases'][0]);c['parameters'][field]=value
            with self.assertRaises(ValueError):audit.case_identity(c)

    def test_ring_reference_singleword_and_guard_counter_corruptions(self):
        for direction in ('gmem_to_smem','smem_to_gmem'):
            case=next(c for c in CONTRACT['cases'] if c['parameters']['direction']==direction and c['parameters']['payload_bytes']==1024)
            for profile in PROFILES:
                if direction=='gmem_to_smem' and profile['role']=='source_release':continue
                seed=profile['required_seed'];arrays=arrays_for(case,profile)
                result=audit.audit_artifact_values(case,profile,seed,arrays);self.assertTrue(result['full_value_replay'])
                for name in arrays:
                    bad=copy.deepcopy(arrays)
                    if name=='bulk_release_clocks.u32le':bad[name]['values'][4:6]=[0,0]
                    else:bad[name]['values'][-1]^=1
                    with self.assertRaises(ValueError,msg=(direction,profile['id'],name)):
                        audit.audit_artifact_values(case,profile,seed,bad)
                for name in arrays:
                    missing=copy.deepcopy(arrays);del missing[name]
                    with self.assertRaises(ValueError):audit.audit_artifact_values(case,profile,seed,missing)

    def test_64kib_last_word_and_uint32_address_truncation_rejected(self):
        case=next(c for c in CONTRACT['cases'] if c['id']=='gmem_to_smem_64kib_one_cta');profile=PROFILES[2]
        arrays=arrays_for(case,profile,1);self.assertEqual(audit.audit_artifact_values(case,profile,profile['required_seed'],arrays)['checked_elements'],33*16384+8)
        arrays['bulk_trace.u32le']['values'][-1]^=1
        with self.assertRaises(ValueError):audit.audit_artifact_values(case,profile,profile['required_seed'],arrays)
        # The address itself, unlike word values, cannot truncate to uint32.
        full=(4096*32+31)*65536;self.assertGreater(full,2**32);self.assertNotEqual(full,full&0xffffffff)

    def test_finite_family_gate_never_accepts_partial_or_release_substitution(self):
        identity={k:hashlib.sha256(k.encode()).hexdigest() for k in ('contract_sha256','profiles_sha256','reference_sha256','source_sha256','normalized_target_set_sha256','compiler_identity_sha256')}
        identity['device_uuid']='GPU-00000000-0000-0000-0000-000000000000';records=[]
        for case in CONTRACT['cases']:
            for profile in PROFILES:
                if profile['role']=='source_release' and case['parameters']['direction']=='gmem_to_smem':continue
                key=case['id']+profile['id'];records.append({'case_id':case['id'],'profile_id':profile['id'],'seed':profile['required_seed'],'status':'pass','full_value_replay':True,'semantic_identity':copy.deepcopy(identity),'receipt_sha256':hashlib.sha256(key.encode()).hexdigest()})
        result=audit.complete_profile_set(CONTRACT,records);self.assertEqual(result['short_receipts'],72);self.assertEqual(result['source_release_receipts'],12);self.assertFalse(result['pilot_authorized'])
        with self.assertRaises(ValueError):audit.complete_profile_set(CONTRACT,records[:-1])
        changes=[lambda r:r[0].update(seed=3),lambda r:r[0].update(profile_id=audit.RELEASE_PROFILE),
                 lambda r:r[0].update(status='failed'),lambda r:r[0].update(full_value_replay=False),
                 lambda r:r[0]['semantic_identity'].update(source_sha256='f'*64),
                 lambda r:r[1].update(receipt_sha256=r[0]['receipt_sha256'])]
        for change in changes:
            altered=copy.deepcopy(records);change(altered)
            with self.assertRaises(ValueError):audit.complete_profile_set(CONTRACT,altered)

    def test_source_release_three_local_clocks_and_complete_source_overwrite(self):
        case=next(c for c in CONTRACT['cases'] if c['id']=='smem_to_gmem_1kib_all_gpu');profile=PROFILES[3];arrays=arrays_for(case,profile,2)
        result=audit.audit_artifact_values(case,profile,3,arrays)
        self.assertEqual(result['source_release_cycles_per_CTA'],[0,0]);self.assertEqual(result['full_completion_cycles_per_CTA'],[2,2]);self.assertFalse(result['performance_eligible'])
        bad=copy.deepcopy(arrays);bad['bulk_source_after.u32le']['values'][0]^=0xffffffff
        with self.assertRaises(ValueError):audit.audit_artifact_values(case,profile,3,bad)
        for field in (0,2,3,4):
            bad=copy.deepcopy(arrays);bad['bulk_completion.u32le']['values'][field*2]^=1
            with self.assertRaises(ValueError):audit.audit_artifact_values(case,profile,3,bad)


if __name__=='__main__':unittest.main()
