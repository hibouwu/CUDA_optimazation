#!/usr/bin/env python3
"""Replace one owned STATUS.md section under a short lock."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[3]
DEFAULT=ROOT/'Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/STATUS.md'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section',required=True,choices=['OVERVIEW','R00','R01','R02','R03','R04','R05','R06','V01'])
    parser.add_argument('--body',required=True,type=Path,help='Markdown section body, including its heading')
    parser.add_argument('--status-file',type=Path,default=DEFAULT)
    args=parser.parse_args();body=args.body.read_text().strip()
    if '<!-- BEGIN:' in body or '<!-- END:' in body:parser.error('body must not contain section markers')
    start=f'<!-- BEGIN:{args.section} -->';end=f'<!-- END:{args.section} -->'
    path=args.status_file.resolve()
    lock_path=Path(tempfile.gettempdir())/('cudaopt-status-'+hashlib.sha256(str(path).encode()).hexdigest()[:16]+'.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        text=path.read_text()
        if text.count(start)!=1 or text.count(end)!=1:raise ValueError('section markers missing or duplicated')
        first=text.index(start)+len(start);last=text.index(end,first)
        updated=text[:first]+'\n'+body+'\n'+text[last:]
        mode=path.stat().st_mode
        with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as temp:
            temp.write(updated);name=temp.name
        os.chmod(name,mode);os.replace(name,path)
    print('updated',args.section)


if __name__=='__main__':main()
