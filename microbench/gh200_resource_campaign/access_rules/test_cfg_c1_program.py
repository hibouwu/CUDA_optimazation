import unittest
import cfg_c1_program as program
import output_model


class CfgC1ProgramTests(unittest.TestCase):
    def graph(self):
        row=dict(id='synthetic-c1',config='cfg_c1',m=320,n=192,k=128,swizzle=1,input_mode='dyadic')
        model=program.build_program(row,dict(config='cfg_c1',m=320,n=192,k=128,grid=[1,1,1],stages=4,tile=[256,128,64],cluster=[1,1]))
        return model,program.expand_cta(model,0)

    def test_partial_source_and_output_conservation(self):
        model,graph=self.graph()
        self.assertEqual(sum(r['bytes'] for r in graph['requests']),len(graph['groups'])*49152)
        self.assertEqual(sum(r['valid_bytes']+r['fill_bytes'] for r in graph['requests']),sum(r['bytes'] for r in graph['requests']))
        self.assertEqual(sum(s['valid_bytes'] for s in graph['output_subtiles']),320*192*4)
        self.assertTrue(any(r['fill_bytes'] for r in graph['requests']))
        self.assertEqual({r['atom']:r['bytes'] for r in graph['requests']},dict(A=32768,B0=8192,B1=8192))

    def test_local_arrivals_and_slot_reuse(self):
        _,graph=self.graph();routes=graph['empty_arrival_routes'];nodes=graph['nodes']
        self.assertEqual(sum(r['arrival_count'] for r in routes),256)
        self.assertEqual([t for r in routes for t in r['physical_threads']],list(range(128,384)))
        self.assertEqual(set(nodes['r0.g4.acquire'])-{'r0.g3.B1_issue'},
            {r['visibility_template'].format(group=0) for r in routes})
        self.assertIn('r0.w0.g1.commit',nodes['r0.w0.g0.warp0.release_emit'])
        self.assertIn('r0.w0.g0.retired',nodes['r0.w0.g0.warp0.release_emit'])

    def test_trace_marker_is_not_all_warps_arriving(self):
        _,graph=self.graph();nodes=graph['nodes'];seen=set()
        def visit(n):
            if n in seen:return
            seen.add(n)
            for p in nodes[n]:visit(p)
        visit('r0.w0.t0.MAIN')
        self.assertIn('r0.w0.g1.warp0.release_emit',seen)
        self.assertNotIn('r0.w0.g1.warp1.release_emit',seen)
        self.assertNotIn('r0.w0.g1.warp0.empty_visible',seen)
        self.assertIn('r0.w0.t0.DONE',nodes['r0.w0.g2.warp0.wait_return'])

    def test_reverse_observed_order_rejected(self):
        _,graph=self.graph()
        with self.assertRaises(ValueError):
            output_model.dependency_bounds(graph,{'r0.w0.t0.MAIN':(20,20),'r0.w0.t0.DONE':(10,10)})

if __name__=='__main__':unittest.main()
