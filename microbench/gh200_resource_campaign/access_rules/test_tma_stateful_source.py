import unittest

from tma_stateful_source import StatefulTensorSource, G3GemmSource
from supply_model import shared_stage_events


class StatefulSourceTest(unittest.TestCase):
    def source(self, cache_bytes=512):
        return StatefulTensorSource([[(0,0),(1,0)],[(0,1),(1,1)]], [0,1], [1.,10.,1.],
            cache_bytes=cache_bytes, hit_ns=5., home_ns=20., link_ns=10., hbm_service_ns=128.)

    def test_pending_fill_is_not_an_immediate_hit(self):
        source=self.source()
        self.assertEqual(source.read(0.,0,0),21.)
        self.assertEqual(source.read(1.,0,0),21.)
        self.assertEqual(source.hbm_lines,1)
        self.assertEqual(source.pending_lines,1)
        self.assertEqual(source.read(30.,0,0),35.)

    def test_remote_subscriber_keeps_own_copy(self):
        source=self.source()
        source.read(0.,0,0)
        self.assertEqual(source.read(2.,1,0),31.)
        self.assertEqual(source.hbm_lines,1)
        # Evict the home copy; the completed remote copy remains readable.
        for line in (1,2,3,4):source.put(0,line,30.)
        self.assertNotIn(0,source.caches[0])
        self.assertEqual(source.read(40.,1,0),45.)
        self.assertEqual(source.hbm_lines,1)

    def test_actual_A_and_B_row_addresses(self):
        source=self.source(cache_bytes=4*1024*1024)
        source.begin(0,1000.)
        a=source.submit(0.,1,(1,2,'A'),1.)
        b=source.submit(20.,1,(1,2,'B1'),.5)
        self.assertEqual(source.line_sets['A',0,2],[(row*8192+256)//128 for row in range(128)])
        self.assertEqual(source.line_sets['B',3,2],[(22*1024*1024+(128+row)*3072+384)//128 for row in range(64)])
        self.assertGreaterEqual(a['source_ready'],10.)
        self.assertGreaterEqual(b['source_ready'],30.)
        self.assertEqual(len(source.diagnostics),2)

    def test_native_allocations_and_measured_group_stride(self):
        addresses=dict(A=0x40029a000000,B=0x40029b600000,records=0x400299e00a00,
            ctas=0x400299e0da00,worklist=0x400299e00000,gate=0x400299e0f800)
        source=StatefulTensorSource([[(0,0),(1,0)],[(0,1),(1,1)]], [0,1], [1.,10.,1.],
            cache_bytes=4*1024*1024,hit_ns=5.,home_ns=20.,link_ns=10.,hbm_service_ns=128.,
            addresses=addresses)
        writes=[]
        source.store=lambda t,d,a,n:writes.append((a,n))
        source.begin(0,1000.,groups=1)
        source.memory_observer(20.,1,0,'issue_record')
        self.assertEqual(writes[-1],(addresses['records']+80,32))
        source.memory_observer(30.,1,1,'cta_record')
        self.assertEqual(writes[-1],(addresses['ctas']+56,56))
        source.submit(0.,1,(1,2,'A'),1.)
        self.assertEqual(source.line_sets['A',0,2],
            [(addresses['A']+row*8192+256)//128 for row in range(128)])

    def test_output_source_observer_preserves_event_results(self):
        parameters=dict(service_ns=2.,read_latency_ns=10.,compute_ns=3.,
            epilogue_prepare_ns=1.,epilogue_write_ns=1.,trace_requests=True)
        expected=shared_stage_events(4,3,3,**parameters)
        observed=[]
        actual=shared_stage_events(4,3,3,output_source_observer=lambda t,c,j:observed.append((t,c,j)),
            **parameters)
        self.assertEqual(expected,actual)
        self.assertEqual(len(observed),3*2*8)
        self.assertEqual([r[0] for r in observed],sorted(r[0] for r in observed))
        for t,c,j in observed:
            output,subtile=divmod(j,8)
            self.assertEqual(t,actual['ctas'][c]['outputs'][output]['source_done'][subtile])

    def test_gemm_output_uses_epi_m_inner_order(self):
        source=G3GemmSource([[(0,0),(11,0)]],[0],[1.,10.,1.],cache_bytes=4*1024*1024,
            hit_ns=5.,home_ns=20.,link_ns=10.,hbm_service_ns=128.)
        source.begin(0,1000.)
        writes=[]
        source.store=lambda t,d,a,n:writes.append((t,a,n))
        for j in range(8):source.output(20.+j,0,j)
        self.assertEqual(len(writes),512)
        self.assertEqual(source.output_bytes,65536)
        self.assertEqual(writes[64][1],source.output_base+64*1536*4)
        self.assertEqual(writes[128][1],source.output_base+32*4)
        self.assertEqual(len({a for t,a,n in writes}),512)


if __name__ == '__main__':
    unittest.main()
