#!/usr/bin/env python3
"""One fixed S08 long-accumulation observation with complete output retention."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from common.suite_io import read_json,relative,require
from common.suite_snapshot import check_snapshot
from runners import accumulation_diagnostic as diagnostic


def dispatch(root,command):
    spec=read_json(root/'run_spec.json')
    require(spec.get('kind')==diagnostic.KIND,'not a long accumulation diagnostic')
    check_snapshot(root,spec)
    script=relative(root,'snapshot/repo/'+diagnostic.ENTRY)
    if script!=Path(__file__).resolve():
        os.execv(sys.executable,[sys.executable,'-B',str(script),command,str(root)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run')
    run.add_argument('--suite',type=Path,required=True);run.add_argument('--output',type=Path,required=True)
    for name in ('audit','status','_execute'):
        sub.add_parser(name,help=argparse.SUPPRESS if name=='_execute' else None).add_argument('run_dir',type=Path)
    args=parser.parse_args()
    if args.command=='run':
        root=args.output.resolve();suite=args.suite.resolve()
        require(root.parent.parent==suite and root.parent.name=='low_precision','suite/family/run layout')
        if not root.exists():
            repo=Path(__file__).resolve().parents[2]
            script=diagnostic.initialize(repo,suite,root)
            os.execv(sys.executable,[sys.executable,'-B',str(script),'_execute',str(root)])
        dispatch(root,'_execute')
        return diagnostic.execute(root)
    root=args.run_dir.resolve();dispatch(root,args.command)
    if args.command=='_execute':return diagnostic.execute(root)
    if args.command=='audit':
        result=diagnostic.audit(root);print(json.dumps(result,indent=2))
        return 2 if result['status']=='diagnostic-failed' else 0
    diagnostic.load(root);print((root/'diagnostic_state.json').read_text());return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,KeyError) as error:
        print('ACCUMULATION_DIAGNOSTIC_REJECTED: '+str(error),file=sys.stderr);raise SystemExit(2)
