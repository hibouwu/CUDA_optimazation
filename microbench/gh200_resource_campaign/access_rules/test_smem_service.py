"""完整SMEM地址输入、实际cfg_b需求与复合写块替换的CPU检查。"""
import json
import unittest

import cfg_b_program
import cfg_b_request_model as model
from smem_service import ordinary_accesses, with_output_write_cost


class SmemServiceTests(unittest.TestCase):
    def program(self, ktiles=4, stages=4):
        return cfg_b_program.build_program(
            dict(config='cfg_b',m=768,n=256,k=64*ktiles,stages=stages),
            dict(grid=[4,1,1],stages=stages))

    def test_lane_order_survives_equal_full_warp_bank_word_sets(self):
        contiguous = [[(lane//4)*4 for lane in range(32)] for _ in range(8)]
        interleaved = [[(lane%8)*4 for lane in range(32)] for _ in range(8)]
        first = ordinary_accesses(contiguous,16,'read')
        second = ordinary_accesses(interleaved,16,'read')
        self.assertEqual(first['requested_bytes'],4096)
        self.assertEqual(first['distinct_address_bytes'],128)
        self.assertEqual([a['bank_words'] for a in first['accesses']],
                         [a['bank_words'] for a in second['accesses']])
        self.assertNotEqual(first['accesses'],second['accesses'])

    def test_wide_same_address_keeps_every_word_and_requesting_lane(self):
        description = ordinary_accesses([[64]*32 for _ in range(8)],16,'read')
        access = description['accesses'][0]
        self.assertEqual(access['lane_words'],[[64,65,66,67]]*32)
        self.assertEqual(access['same_word_lanes'],
            [dict(word=word,lanes=list(range(32))) for word in range(64,68)])
        self.assertEqual(description['distinct_address_bytes'],16)
        self.assertEqual(description['requested_bytes'],4096)
        self.assertEqual(description['warp_instructions'],8)

    def test_store_aliases_are_recorded_without_broadcast_work_reduction(self):
        rows = [[0]*32 for _ in range(8)]
        read = ordinary_accesses(rows,8,'read')
        write = ordinary_accesses(rows,8,'write')
        self.assertEqual(write['direction'],'write')
        self.assertEqual(write['requested_bytes'],read['requested_bytes'])
        self.assertEqual(write['accesses'][0]['same_word_lanes'][0]['lanes'],list(range(32)))

    def test_logical_paths_have_separate_coverage_and_byte_units(self):
        program = self.program(13,6)
        result = cfg_b_program.smem_demands(program)
        self.assertEqual(result['output_tiles'],12)
        paths={p['path']:p for p in result['paths']}
        self.assertEqual(len({p['covered_by'] for p in paths.values()}),4)
        self.assertEqual(paths['tma_input_destination']['total_bytes'],12*13*32768)
        self.assertEqual(paths['wgmma_operand_references']['total_operand_reference_bytes'],12*13*49152)
        write=paths['ordinary_output_store']
        self.assertEqual(write['opcode'],'STS.64')
        self.assertEqual(write['threads']*write['instructions_per_thread_per_subtile']*write['bytes_per_lane'],8192)
        self.assertEqual(write['total_bytes'],paths['tma_output_source']['total_bytes'])
        self.assertEqual(write['total_bytes'],12*128*128*4)
        json.dumps(result,allow_nan=False)

    def test_actual_sts64_sequence_covers_each_output_buffer_once(self):
        for subtile in range(8):
            addresses=cfg_b_program.epilogue_smem_addresses(subtile)
            self.assertEqual(len(addresses),8)
            self.assertEqual(set(map(len,addresses)),{128})
            words=[word+i for row in addresses for word in row for i in (0,1)]
            first=(0x800+(subtile%2)*0x2000)//4
            self.assertEqual(sorted(words),list(range(first,first+2048)))
            demand=ordinary_accesses(addresses,8,'write')
            self.assertEqual((demand['warp_instructions'],demand['requested_bytes'],
                              demand['distinct_address_bytes']),(32,8192,8192))
            self.assertTrue(all(not access['same_word_lanes'] for access in demand['accesses']))

    def test_write_block_price_replaces_the_old_price(self):
        parameters=dict(epilogue_write_ns=100.,compute_ns=7.)
        actual=with_output_write_cost(parameters,cycles_per_subtile=12.,frequency_ghz=2.)
        self.assertEqual(actual,dict(epilogue_write_ns=6.,compute_ns=7.))
        self.assertEqual(parameters['epilogue_write_ns'],100.)

    def test_six_programs_preserve_default_events_and_output_buffer_reuse(self):
        parameters=dict(service_ns=2.,read_latency_ns=10.,compute_ns=7.,commit_lead_ns=2.,
            output_latency_ns=19.,epilogue_prepare_ns=3.,epilogue_write_ns=4.)
        for ktiles in (4,13,32):
            for stages in (4,6):
                with self.subTest(ktiles=ktiles,stages=stages):
                    program=self.program(ktiles,stages)
                    direct=model.simulate(program,parameters,detailed_ctas=range(4))
                    equivalent=with_output_write_cost(parameters,cycles_per_subtile=8.,frequency_ghz=2.)
                    self.assertEqual(direct,model.simulate(program,equivalent,detailed_ctas=range(4)))
                    changed=with_output_write_cost(parameters,cycles_per_subtile=80.,frequency_ghz=2.)
                    slower=model.simulate(program,changed,detailed_ctas=range(4))
                    self.assertGreater(slower['consumer_span_ns'],direct['consumer_span_ns'])
                    self.assertEqual(slower['accounting']['input_delivery_count'],
                                     direct['accounting']['input_delivery_count'])
                    for cta in slower['details']:
                        for output in cta['output_service']:
                            for subtile in range(8):
                                self.assertEqual(output['write_end'][subtile]-output['write_begin'][subtile],40.)
                                self.assertGreaterEqual(output['submitted'][subtile],output['write_end'][subtile])
                                if subtile>=2:
                                    self.assertGreaterEqual(output['write_begin'][subtile],output['source_done'][subtile-2])


if __name__=='__main__':
    unittest.main()
