import copy
import hashlib
import unittest
from joint_records import assemble
from extract_joint import validate_pair_identities


class JointRecordTests(unittest.TestCase):
    def setUp(self):
        self.protocol=dict(series=['S','L'],repeats_per_arm=2,gpu_uuid='gpu',binary_sha256='binary')
        self.records=[]
        for cohort in ('off_on','off_off'):
            for arm in (0,1):
                for block in range(2):
                    identity=f'{cohort}/{arm}/{block}'
                    self.records.append(dict(id=identity,cohort=cohort,arm=arm,block=block,
                        flag=arm if cohort=='off_on' else 0,sm_by_cta={'0':5},
                        raw_sha256=hashlib.sha256(identity.encode()).hexdigest(),gpu_uuid='gpu',binary_sha256='binary',
                        series={'S':[block+arm,None],'L':[10-block,20-block]}))

    def test_shuffled_records_preserve_each_joint_vector(self):
        samples,ordered=assemble(self.protocol,list(reversed(self.records)))
        self.assertEqual(samples['off_on']['S']['arm1'],[1,2])
        self.assertEqual(samples['off_on']['L']['arm1'],[10,9])
        for row in ordered:self.assertEqual(row['series']['S'][0]+row['series']['L'][0],10+row['arm'])

    def test_duplicate_block_or_process_or_raw_rejected(self):
        for field in ('block','id','raw_sha256'):
            rows=copy.deepcopy(self.records);rows[1][field]=rows[0][field]
            with self.assertRaises(ValueError):assemble(self.protocol,rows)

    def test_missing_coordinate_and_wrong_binary_rejected(self):
        rows=copy.deepcopy(self.records);del rows[0]['series']['S']
        with self.assertRaises(ValueError):assemble(self.protocol,rows)
        rows=copy.deepcopy(self.records);rows[0]['binary_sha256']='wrong'
        with self.assertRaises(ValueError):assemble(self.protocol,rows)

    def test_false_block_and_wrong_ns_type_rejected(self):
        for change in ('block','ns'):
            rows=copy.deepcopy(self.records)
            if change=='block':rows[0]['block']=False
            else:rows[0]['series']['S'][0]=True
            with self.assertRaises(ValueError):assemble(self.protocol,rows)

    def test_wrong_arm_flag_and_sm_remap_rejected(self):
        _,ordered=assemble(self.protocol,self.records)
        validate_pair_identities(ordered)
        for change in ('flag','sm'):
            rows=copy.deepcopy(ordered)
            if change=='flag':rows[0]['flag']=1
            else:rows[0]['sm_by_cta']['0']=6
            with self.assertRaises(ValueError):validate_pair_identities(rows)


if __name__=='__main__':unittest.main()
