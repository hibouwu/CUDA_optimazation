import unittest
from test_cfg_c_producer_tail_packet import packet as original_packet
from decode_cfg_c_gt_bracket import decode, VERSION


def packet(flag=1):
    setup, call = original_packet(0)
    fixed = dict(gt_bracket_protocol=VERSION, gt_bracket_prefix_flag=0,
        gt_shadow_words_per_cta=9, gt_shadow_bytes=9504, gt_bracket_observer=flag,
        warmup_swizzle=1, warmup_fixed_calls=16, history_protocol="none",
        history_source="original", source_copies=1, mirror_target_slot=0,
        release_warp_observer=0, release_warp_warmup=0, epilogue_defer=0,
        epilogue_warmup_defer=0, second_prefill=0, second_prefill_warmup=0,
        a_row_rotation=0, first_supply_events=0, producer_delay_ns=0, work_swap55_91=0,
        b_offset_bytes=0, b_allocation_extra_bytes=0, b_payload_elements=4096*3072,
        b_buffer_ptr=2048)
    setup.update(fixed)
    call.update(fixed)
    setup.update(trace_version=VERSION, mode="warm", threads=384, scheduler_sm_count=132,
        requested_sm_count=0, max_active_ctas_per_sm=1, raster_order="heuristic",
        resolved_raster_order="along_m", evict=0, history_capacity_bytes=0, trace_tile_capacity=64)
    call.update(label="warm", target_swizzle=1, target_calls=1, preceding_target_calls=0,
        elapsed_us=1, gt_shadow=[0]*(132*9))
    for cta in range(132):
        for wg in (0, 1):
            count=call["trace"][cta*1296+6+3*wg]
            for ordinal in range(count):
                if flag:
                    start=10+30*ordinal
                    pos=cta*9+wg*4+ordinal*2
                    call["gt_shadow"][pos:pos+2]=[start-2,start+7]
    return setup,call


class GtPacketTests(unittest.TestCase):
    def test_both_arms_full_fixed_work(self):
        for flag in (0,1):
            result=decode(*packet(flag),'synthetic')
            self.assertTrue(result['record_valid'])
            self.assertEqual(len(result['records']),132)
            self.assertFalse(result['observation_qualified'])
            self.assertFalse(result['hardware_sampling_enclosure'])
            tile=result['records'][0]['roles'][0]['tiles'][0]
            if flag:
                self.assertEqual(tile['conditional_elapsed_gt'],[7,12])
                self.assertEqual(tile['post_gt_elapsed'],10)
                self.assertEqual(tile['post_gt_error_bound'],3)
            else:self.assertIsNone(tile['gt_brackets'])

    def test_structural_and_shadow_failures_retained(self):
        for field,index,value in [('trace',11,2),('trace',20,999),('trace',16,0),
                ('gt_shadow',0,0),('gt_shadow',0,11),('gt_shadow',1,10),
                ('gt_shadow',1,21),('gt_shadow',8,1)]:
            setup,call=packet();call[field][index]=value
            result=decode(setup,call,'synthetic')
            self.assertFalse(result['record_valid'])
            self.assertTrue(any(e['cta']==0 for e in result['errors']))

    def test_disabled_and_unused_shadow(self):
        setup,call=packet(0);call['gt_shadow'][0]=1
        self.assertFalse(decode(setup,call,'synthetic')['record_valid'])
        setup,call=packet()
        cta=next(i for i in range(132) if call['trace'][i*1296+6]==1)
        call['gt_shadow'][cta*9+2]=1
        self.assertFalse(decode(setup,call,'synthetic')['record_valid'])

    def test_wrong_protocol_or_types_rejected(self):
        for target,key,value in [('setup','trace_version','cfg-c-producer-tail-v2'),
                ('setup','gpu_uuid','other'),('setup','threads',256),
                ('call','gt_bracket_observer',True),('call','gt_bracket_observer',0),
                ('call','gt_bracket_prefix_flag',1),('call','producer_tail_observer',1),
                ('call','second_prefill',1),('call','preceding_target_calls',1),
                ('call','target_swizzle',2),('call','b_buffer_ptr',4096),
                ('call','gt_shadow_words_per_cta',True),('call','elapsed_us',float('nan'))]:
            setup,call=packet();(setup if target=='setup' else call)[key]=value
            with self.assertRaises(ValueError):decode(setup,call,'synthetic')

    def test_shadow_missing_short_and_noninteger_rejected(self):
        for bad in (None,[0]*1187,[True]*1188,[2**64]*1188):
            setup,call=packet();call['gt_shadow']=bad
            with self.assertRaises(ValueError):decode(setup,call,'synthetic')


if __name__=='__main__':unittest.main()
