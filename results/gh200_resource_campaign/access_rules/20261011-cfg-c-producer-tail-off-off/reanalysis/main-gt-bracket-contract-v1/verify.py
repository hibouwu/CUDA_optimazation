#!/usr/bin/env python3
"""Independent endpoint-box corner checks; no device or archived timer input."""
import copy
from fractions import Fraction as Q
import json
from pathlib import Path

from r09_gt_brackets import GtClockStamp,error_intervals,relative_error_status,window_interval


def check_values(result,clock_delta,expected_lower,expected_upper,predicted_cycles,predicted_rate):
    assert result['cycles']==clock_delta
    assert result['elapsed_lower_ns']==max(0,expected_lower)
    assert result['elapsed_upper_ns']==expected_upper
    assert result['elapsed_lower_open']==(expected_lower<=0)
    assert result['rate_lower_cycle_per_ns']==Q(clock_delta)/expected_upper
    expected_rate_upper=Q(clock_delta)/expected_lower if expected_lower>0 else None
    assert result['rate_upper_cycle_per_ns']==expected_rate_upper
    assert result['cycle_error']==Q(predicted_cycles)/clock_delta-1
    assert result['conversion_error_lower']==Q(clock_delta)/expected_upper/predicted_rate-1
    assert result['free_error_lower']==Q(predicted_cycles)/predicted_rate/expected_upper-1
    if expected_lower>0:
        assert result['conversion_error_upper']==Q(clock_delta)/expected_lower/predicted_rate-1
        assert result['free_error_upper']==Q(predicted_cycles)/predicted_rate/expected_lower-1
    else:
        assert result['conversion_error_upper'] is None and result['free_error_upper'] is None


def main():
    cases=empty=positive_corners=unbounded=0
    for begin_width in (0,1,7,32):
        for end_width in (0,3,17):
            for gap in (-8,0,4,40,128):
                for error in (Q(0),Q(1,2),Q(4)):
                    for clock_delta in (1,150,65536):
                        begin=GtClockStamp('GPU-synthetic',2,10000,100000,10000+begin_width)
                        end=GtClockStamp('GPU-synthetic',2,10000+gap,100000+clock_delta,10000+gap+end_width)
                        start_box=(Q(begin.gt_before)-error,Q(begin.gt_after)+error)
                        end_box=(Q(end.gt_before)-error,Q(end.gt_after)+error)
                        corners=[t_end-t_begin for t_begin in start_box for t_end in end_box]
                        low,high=min(corners),max(corners);cases+=1
                        if high<=0:
                            try:window_interval(begin,end,gt_error_ns=error)
                            except ValueError:empty+=1
                            else:raise AssertionError('nonpositive endpoint box accepted')
                            continue
                        predicted_cycles=Q(3*clock_delta,4)+1;predicted_rate=Q(3,2)
                        result=error_intervals(begin,end,gt_error_ns=error,
                            predicted_cycles=predicted_cycles,predicted_rate=predicted_rate)
                        check_values(result,clock_delta,low,high,predicted_cycles,predicted_rate)
                        unbounded+=low<=0
                        for elapsed in corners:
                            if elapsed<=0:continue
                            conversion=Q(clock_delta)/elapsed/predicted_rate-1
                            free=predicted_cycles/predicted_rate/elapsed-1
                            assert (1+result['cycle_error'])*(1+conversion)==1+free
                            assert result['free_error_lower']<=free
                            if result['free_error_upper'] is not None:assert free<=result['free_error_upper']
                            positive_corners+=1
                        assert result['observation_qualified'] is False
    # Two false certificates retain apparent simplicity or algebraic closure.
    begin=GtClockStamp('GPU-synthetic',2,100,1000,108)
    end=GtClockStamp('GPU-synthetic',2,200,1150,212)
    result=error_intervals(begin,end,gt_error_ns=2,predicted_cycles=144,predicted_rate=Q(3,2))
    altered=copy.deepcopy(result);altered['elapsed_lower_ns']+=1
    try:check_values(altered,150,Q(88),Q(116),Q(144),Q(3,2))
    except AssertionError:pass
    else:raise AssertionError('incorrect elapsed bound accepted')
    swapped=copy.deepcopy(result)
    swapped['cycle_error'],swapped['conversion_error_lower']=swapped['conversion_error_lower'],swapped['cycle_error']
    assert (1+swapped['cycle_error'])*(1+swapped['conversion_error_lower'])==1+swapped['free_error_lower']
    try:check_values(swapped,150,Q(88),Q(116),Q(144),Q(3,2))
    except AssertionError:pass
    else:raise AssertionError('closed but swapped errors accepted')
    classification=relative_error_status(result['free_error_lower'],result['free_error_upper'],tolerance=Q(1,20))
    assert classification['status']=='undetermined'
    report=dict(passed=True,synthetic_endpoint_boxes=cases,inconsistent_boxes_rejected=empty,
        positive_corner_error_checks=positive_corners,open_zero_elapsed_cases=unbounded,
        negative_certificates_rejected=['incorrect elapsed bound','closed but swapped labeled errors'],
        sample=dict(cycles=150,elapsed_ns=['88','116'],rate_cycle_per_ns=['75/58','75/44'],
            predicted_cycles='144',predicted_rate='3/2',predicted_ns='96',
            cycle_error='-1/25',conditional_conversion_error=['-4/29','3/22'],
            free_time_error=['-5/29','1/11'],status_at_five_percent='undetermined'),
        scope='Pure conditional interval arithmetic; no actual GT-CLOCK-GT samples, accuracy, enclosure or observation qualification.',
        observation_qualified=False,prediction_qualified=False)
    (Path(__file__).resolve().parent/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
