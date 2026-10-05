#!/usr/bin/env python3
"""Isolated short diagnostics. Full-family B3 promotion is deliberately unavailable here."""
import argparse
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from common.suite_io import read_json,relative,require
from common.suite_snapshot import check_snapshot
from runners import validation_diagnostic as diagnostic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run')
    for option in ('contract','profiles','adapter-manifest','suite','output'):run.add_argument('--'+option,type=Path,required=True)
    run.add_argument('--case',required=True);run.add_argument('--profile',required=True);run.add_argument('--seed',type=int,default=3)
    for name in ('resume','audit','status'):sub.add_parser(name).add_argument('run_dir',type=Path)
    args=parser.parse_args()
    if args.command=='run':
        root=args.output.resolve();repo=Path(__file__).resolve().parents[2]
        script=diagnostic.initialize(repo,args.suite,root,args.contract,args.profiles,args.adapter_manifest,args.case,args.profile,args.seed)
        os.execv(sys.executable,[sys.executable,'-B',str(script),'resume',str(root)])
    root=args.run_dir.resolve();spec=read_json(root/'validation_spec.json');check_snapshot(root,spec)
    script=relative(root,'snapshot/repo/microbench/gh200_resource_campaign/validate_suite.py')
    if script!=Path(__file__).resolve():os.execv(sys.executable,[sys.executable,'-B',str(script),args.command,str(root)])
    if args.command=='resume':return diagnostic.execute(root)
    if args.command=='audit':
        import json
        result=diagnostic.audit(root)
        print(json.dumps(result,indent=2))
        return 2 if result['status']=='case_diagnostic_failed' else 0
    diagnostic.load(root)
    print((root/'validation_state.json').read_text());return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,KeyError) as error:
        print('VALIDATION_REJECTED: '+str(error),file=sys.stderr);raise SystemExit(2)
