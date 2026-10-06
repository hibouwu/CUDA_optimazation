"""One-pass finite S14 payload and target-code observer; retains no large arrays."""
import hashlib,json
from common.suite_io import require
from common.family_b3 import target_identity
from common.s14_packed_values import S14WordObserver
from auditors.tma_bulk import audit_sass


class S14EvidenceObserver:
    def __init__(self,descriptors,sass_members,contract):
        self.numeric=S14WordObserver(descriptors)
        self.sass_members=sass_members
        self.contract=contract
        self.targets={}
        self.active_name=None
        self.active_bytes=bytearray()
        self.symbols=[f'tb_{mode}_q{q}' for mode in ('g2s','s2g','release') for q in (1024,4096,8192,16384,32768,65536)]
        require(len(sass_members)==84 and all(0<size<=16*1024*1024 for size in sass_members.values()),'fixed84 bounded SASS inputs')

    def __call__(self,name,offset,chunk):
        self.numeric(name,offset,chunk)
        if name not in self.sass_members:return
        require(name not in self.targets,'SASS member repeated')
        if self.active_name is None:
            require(offset==0,'complete SASS stream begins at zero');self.active_name=name
        require(self.active_name==name and offset==len(self.active_bytes),'one contiguous bounded SASS member')
        self.active_bytes.extend(chunk)
        require(len(self.active_bytes)<=self.sass_members[name],'SASS member exceeds fixed length')
        if len(self.active_bytes)==self.sass_members[name]:
            text=self.active_bytes.decode('utf-8');audit_sass(text,self.contract)
            full=target_identity(text,self.symbols)
            facts={symbol:{'instruction_count':len(rows),'target_identity_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()} for symbol,rows in full.items()}
            require(len(facts)==18,'all18 actual encoded targets')
            if self.targets:require(facts==next(iter(self.targets.values())),'S14 complete target encoding drift across original84')
            self.targets[name]=facts
            self.active_bytes=bytearray();self.active_name=None

    def finish(self):
        require(self.active_name is None and not self.active_bytes and set(self.targets)==set(self.sass_members),'complete all84 SASS member consumption')
        return {'complete_values':self.numeric.finish(),'target_facts':self.targets,'source_kind':'actual complete original SASS,including PC/opcode/predicate/operands/both64bit words',
                'full_SASS_members_checked':84,'actual_targets_per_member':18,'full_SASS_retained_in_pack':True,'large_arrays_retained_in_RAM':False,'bridge_qualified':False}
