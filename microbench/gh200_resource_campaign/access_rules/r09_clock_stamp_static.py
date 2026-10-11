#!/usr/bin/env python3
"""Extract the candidate inline PTX and audit ONLY its isolated SASS fixture.

This is a compiler control-flow check, not a sampling enclosure, runtime
accuracy, CUTLASS resource, perturbation or prediction qualification.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def inline_fixture(header,sites=1):
    if sites not in (1,2):
        raise ValueError('one/two-site fixture only')
    fragment=header.split('asm volatile(',1)[1].split(': "=l"',1)[0]
    body=''.join(ast.literal_eval(s) for s in re.findall(r'"(?:\\.|[^"\\])*"',fragment)).replace('%%','%')
    for i,name in enumerate(('before_out','ns_out','after_out')):
        body=body.replace('%'+str(i),name)
    if any(x in body for x in ('%0','%1','%2')):
        raise ValueError('unbound inline PTX output')
    text='''.version 8.0
.target sm_90a
.address_size 64
.visible .entry r09_clock_stamp_ptx_fixture(.param .u64 output_ptr) {
.reg .u64 output_addr,before_out,ns_out,after_out;
ld.param.u64 output_addr,[output_ptr];
'''
    for site in range(sites):
        text+=body+'\n'
        for offset,register in enumerate(('before_out','ns_out','after_out')):
            text+=f'st.global.u64 [output_addr+{site*24+offset*8}],{register};\n'
    return text+'ret;\n}\n'


def audit_fixture(sass,sites=1):
    if sites not in (1,2) or sass.count('Function :')!=1 or 'Function : r09_clock_stamp_ptx_fixture' not in sass:
        raise ValueError('isolated named fixture required; no general kernel audit')
    instructions=[]
    for address,predicate,operation,operands in re.findall(
            r'/\*([0-9a-fA-F]+)\*/\s*(?:(@!?P\d+)\s+)?([A-Z][A-Z0-9.]*)\s*([^;]*);',sass):
        instructions.append(dict(pc=int(address,16),predicate=predicate,operation=operation,
                                 operands=[v.strip() for v in operands.split(',')]))
    reads=[i for i,ins in enumerate(instructions) if ins['operation']=='CS2R' and ins['operands'][-1] in ('SR_CLOCKLO','SR_GLOBALTIMERLO')]
    names=[instructions[i]['operands'][-1] for i in reads]
    if names!=['SR_CLOCKLO','SR_GLOBALTIMERLO','SR_CLOCKLO']*sites:
        raise ValueError('exact three-read order/sites required')
    locations={ins['pc']:i for i,ins in enumerate(instructions)}
    branches=[]
    for site in range(sites):
        before_i,gt_i,after_i=reads[site*3:site*3+3]
        invalid_i=gt_i+7
        zero_block=instructions[invalid_i:invalid_i+4]
        expected_registers=[instructions[i]['operands'][0] for i in (before_i,gt_i,after_i)]
        if (len(zero_block)!=4 or any(ins['operation']!='CS2R' or ins['predicate'] or ins['operands']!=[register,'SRZ']
                for ins,register in zip(zero_block[:3],expected_registers))
                or zero_block[3]['operation']!='BRA' or zero_block[3]['predicate']
                or int(zero_block[3]['operands'][0],16)!=instructions[after_i+1]['pc']):
            raise ValueError('invalid output must zero all three values and skip the final read')
        for current,next_read in zip(reads[site*3:site*3+2],reads[site*3+1:site*3+3]):
            read=instructions[current]
            register=read['operands'][0]
            if not re.fullmatch(r'R\d+',register):
                raise ValueError('64-bit general register read required')
            high='R'+str(int(register[1:])+1)
            low_compare,high_compare,branch=instructions[current+1:current+4]
            predicate=low_compare['operands'][0]
            if (low_compare['operation']!='ISETP.NE.U32.AND'
                    or low_compare['operands']!=[predicate,'PT',register,'RZ','PT']
                    or high_compare['operation']!='ISETP.NE.U32.AND.EX'
                    or high_compare['operands']!=[predicate,'PT',high,'RZ','PT',predicate]
                    or branch['operation']!='BRA' or branch['predicate']!='@'+predicate):
                raise ValueError('next read lacks a branch dependent on both preceding halves')
            target=locations.get(int(branch['operands'][0],16))
            if target is None or not current+4<=target<=next_read:
                raise ValueError('dependent branch does not enter the following read block')
            if any(ins['operation']!='IMAD.MOV.U32' or ins['operands'][1:]!=['RZ','RZ','RZ']
                   for ins in instructions[target:next_read]):
                raise ValueError('unexpected instructions before guarded next read')
            fallthrough=instructions[current+4:current+7]
            if (len(fallthrough)!=3 or fallthrough[0]['operation']!='VIADD'
                    or fallthrough[0]['operands'][1:]!=[fallthrough[0]['operands'][0],'0x1']
                    or fallthrough[1]['operation']!='ISETP.GE.U32.AND'
                    or fallthrough[1]['operands']!=[predicate,'PT',fallthrough[0]['operands'][0],'0x2','PT']
                    or fallthrough[2]['operation']!='BRA' or fallthrough[2]['predicate']!='@!'+predicate
                    or int(fallthrough[2]['operands'][0],16)!=read['pc']):
                raise ValueError('two-read retry/exit shape changed')
            counter=fallthrough[0]['operands'][0]
            writes=[ins for ins in instructions[:current] if ins['operands'][0]==counter]
            if not writes or writes[-1]['operation']!='IMAD.MOV.U32' or writes[-1]['operands'][1:]!=['RZ','RZ','RZ']:
                raise ValueError('retry counter must start at zero')
            if current==before_i:
                exit_branch=instructions[current+7]
                if (target!=current+8 or exit_branch['operation']!='BRA' or exit_branch['predicate']
                        or int(exit_branch['operands'][0],16)!=instructions[invalid_i]['pc']):
                    raise ValueError('before-read exhaustion must enter the zero-output block')
            elif target!=invalid_i+4:
                raise ValueError('GT exhaustion must fall through the zero-output block')
            branches.append(dict(site=site,read_pc=hex(read['pc']),next_read_pc=hex(instructions[next_read]['pc']),
                                 predicate=predicate,branch_pc=hex(branch['pc']),guard_target_pc=hex(instructions[target]['pc'])))
    return dict(scope='Isolated compile fixture only; original CUTLASS binary and hardware read enclosure unqualified.',
                sites=sites,special_reads=len(reads),dependent_exit_branches=branches,
                predicate_selection_alone_accepted=False,runtime_qualified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--header',type=Path)
    parser.add_argument('--ptx-output',type=Path)
    parser.add_argument('--sass',type=Path)
    parser.add_argument('--sites',type=int,default=1)
    args=parser.parse_args()
    if args.header:
        if args.ptx_output is None:
            parser.error('--header requires --ptx-output')
        with args.ptx_output.open('x') as stream:
            stream.write(inline_fixture(args.header.read_text(),args.sites))
    if args.sass:
        result=audit_fixture(args.sass.read_text(),args.sites)
        result['sass_sha256']=hashlib.sha256(args.sass.read_bytes()).hexdigest()
        print(json.dumps(result,indent=2))
    if not args.header and not args.sass:
        parser.error('choose --header or --sass')


if __name__=='__main__':
    main()
