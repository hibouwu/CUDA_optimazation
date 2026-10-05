import unittest,struct
from common.s14_packed_values import S14WordObserver


class CompleteWordStreamTests(unittest.TestCase):
    def descriptor(self,role,steps=1,seed=3,g2s=False,release=False):
        shapes={'bulk_guards.u32le':[2,4],'bulk_trace.u32le':[1,steps,256],
                'bulk_ring.u32le':[1,32,256],'bulk_source_after.u32le':[1,256],
                'bulk_completion.u32le':[1,5,2],'bulk_release_clocks.u32le':[1,3,2]}
        shape=shapes[role];count=1
        for n in shape:count*=n
        return {'run':'synthetic','role':role,'shape':shape,'bytes':count*4,'blocks':1,'words':256,'iterations':steps,'seed':seed,'g2s':g2s,'release':release}

    def feed(self,descriptor,values,chunk=7):
        name='synthetic/'+descriptor['role'];obs=S14WordObserver({name:descriptor});data=struct.pack('<'+'I'*len(values),*values)
        for offset in range(0,len(data),chunk):obs(name,offset,data[offset:offset+chunk])
        return obs

    def test_g2s33_wrap_and_unsigned_seed(self):
        d=self.descriptor('bulk_trace.u32le',steps=33,seed=4294967295,g2s=True)
        values=[(17*((i//256%32)*256+i%256)+4294967295)&0xffffffff for i in range(33*256)]
        r=self.feed(d,values).finish();self.assertEqual(r['payload_guard_words'],33*256)

    def test_s2g_ring_unvisited_poison(self):
        d=self.descriptor('bulk_ring.u32le');values=[]
        for slot in range(32):
            for w in range(256):
                value=29*w+3;values.append(value if slot==0 else value^0xffffffff)
        self.assertEqual(self.feed(d,values).finish()['payload_guard_words'],8192)
        values[-1]^=1
        with self.assertRaisesRegex(ValueError,'word=8191'):self.feed(d,values)

    def test_source_release_complement(self):
        d=self.descriptor('bulk_source_after.u32le',release=True)
        values=[(29*w+3)^0xffffffff for w in range(256)];self.feed(d,values).finish()
        values[255]^=1
        with self.assertRaises(ValueError):self.feed(d,values)

    def test_lifecycle_low_high_pairs(self):
        d=self.descriptor('bulk_completion.u32le',g2s=True);pairs=[1,2**32+17,0,0,0]
        values=[n for v in pairs for n in (v&0xffffffff,v>>32)];r=self.feed(d,values).finish();self.assertEqual(r['lifecycle_words'],10)
        values[0]=0
        with self.assertRaises(ValueError):self.feed(d,values).finish()

    def test_release_clocks_high_bits_and_order(self):
        d=self.descriptor('bulk_release_clocks.u32le',release=True);pairs=[2**53+1,2**53+2,2**53+4]
        values=[n for v in pairs for n in (v&0xffffffff,v>>32)];self.feed(d,values).finish()
        pairs=[2**53+4,2**53+2,2**53+1];values=[n for v in pairs for n in (v&0xffffffff,v>>32)]
        with self.assertRaises(ValueError):self.feed(d,values).finish()

    def test_missing_truncated_or_noncontiguous_bytes(self):
        d=self.descriptor('bulk_guards.u32le');name='synthetic/'+d['role'];values=[0xd15ea5e0+i for i in range(8)];data=struct.pack('<8I',*values)
        obs=S14WordObserver({name:d});obs(name,0,data[:-1])
        with self.assertRaises(ValueError):obs.finish()
        obs=S14WordObserver({name:d})
        with self.assertRaises(ValueError):obs(name,1,data)

if __name__=='__main__':unittest.main()
