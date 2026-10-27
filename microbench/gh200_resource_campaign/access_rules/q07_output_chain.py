"""cfg_b八子块、两源槽输出链；复用完整引擎的状态转换。"""
import argparse
import heapq
import json
from pathlib import Path

from supply_model import pingpong_output_state, pingpong_output_step


def simulate(available_ns, *, prepare_ns, write_ns, control_ns, tail_ns, source_read_service):
    """available_ns给每输出的上游许可；source_read_service(t,out,j)给源读取完成。"""
    outputs=[pingpong_output_state() for _ in available_ns]
    events=[];serial=0

    def push(t,kind,index):
        nonlocal serial
        heapq.heappush(events,(t,serial,kind,index));serial+=1

    def source_read(t,index):
        out,j=divmod(index,8)
        return source_read_service(t,out,j)

    outputs[0]['scheduled']=True
    push(available_ns[0],'epi_begin',0)
    while events:
        t,_,kind,index=heapq.heappop(events)
        pingpong_output_step(t,kind,index,outputs,prepare_ns=prepare_ns,write_ns=write_ns,
            control_ns=control_ns,tail_ns=tail_ns,push=push,
            reserve=lambda t,duration,kind,index:None,source_read=source_read)
        if kind=='epi_end' and index+1<len(outputs):
            outputs[index+1]['scheduled']=True
            push(max(t,available_ns[index+1]),'epi_begin',index+1)
    subtiles=[]
    for out,e in enumerate(outputs):
        for j in range(8):
            if j>=2:previous=e['source_done'][j-2]
            elif out:previous=outputs[out-1]['source_done'][6+j]
            else:previous=None
            subtiles.append(dict(output=out,subtile=j,bytes=8192,source_buffer=j%2,
                source_generation=out*4+j//2,
                prepare_begin_ns=e['prepare_begin'][j],prepare_end_ns=e['prepare_end'][j],
                write_begin_ns=e['write_begin'][j],write_end_ns=e['write_end'][j],
                submitted_ns=e['submitted'][j],source_done_ns=e['source_done'][j],
                reuse_predecessor_ns=previous))
    return dict(unit='ns',qualification='声明服务价格的独立输出链',
        endpoint='store_tail read0 + declared tail cost',
        outputs=outputs,subtiles=subtiles,end_ns=outputs[-1]['end'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    given=json.loads(args.input.read_text())
    result=simulate(given['available_ns'],prepare_ns=given['prepare_ns'],
        write_ns=given['write_ns'],control_ns=given['control_ns'],tail_ns=given['tail_ns'],
        source_read_service=lambda t,out,j:t+given['source_read_latency_ns'])
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
