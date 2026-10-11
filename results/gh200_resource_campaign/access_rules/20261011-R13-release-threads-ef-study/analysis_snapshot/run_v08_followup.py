#!/usr/bin/env python3
"""V08 follow-up: isolate the three mechanisms behind the V08 held-out failures (diagnostic).

  pitch : A/B/D row pitch not a multiple of 128 B vs K/N tails (M = 2304, whole tile rows)
  pad   : swizzle 8 padding along M only / N only / both, K 1024 and 4096
  bw    : large footprint, long K (cfg_b showed 633 vs 513 cycle/Ktile at 3584^2 x 20480)
Same binaries (SASS-identical rebuild of the V08 sources), variants and sampling as run_v08.
  run_v08_followup.py prepare --output RUN --v08 V08RUN ; then run_v08.py setup / sample --set ctrl.
"""
import json,shutil,sys
from pathlib import Path
import run_v08
import v06_run as common

ROOT=Path(__file__).resolve().parent


def matrix():
    rows=[]
    def add(case,cfg,m,n,k,lda=None,ldb=None,ldd=None,sw=1,group=''):
        r=run_v08.row(f'{cfg}_{case}',cfg,'ctrl',m,n,k,sw,kind=run_v08.boundary_kind(cfg,m,n))
        r.update(lda=lda or k,ldb=ldb or n,ldd=ldd or n,group=group);rows.append(r)
    for cfg in run_v08.CONFIGS:
        m=2304
        for k in (1024,4096):
            add(f'p_ref_k{k}',cfg,m,3072,k,group='pitch')
            add(f'p_apitch_k{k}',cfg,m,3072,k,lda=k+8,group='pitch')        # 2064 / 8208 B rows
            add(f'p_bpitch_k{k}',cfg,m,3072,k,ldb=3080,group='pitch')       # 6160 B rows
        add('p_dpitch_k1024',cfg,m,3072,1024,ldd=3080,group='pitch')        # 12320 B rows
        add('p_ktail',cfg,m,3072,1000,lda=1024,group='pitch')
        add('p_ktail_apitch',cfg,m,3072,1000,group='pitch')                 # = h03 A side
        add('p_ntail',cfg,m,3000,1024,ldb=3072,ldd=3072,group='pitch')
        add('p_ntail_bdpitch',cfg,m,3000,1024,group='pitch')                # = h03 B/D side
        for label,mm,nn in (('padM',2304,4096),('padN',4096,2304),('padMN',3072,3072)):
            for k in (1024,4096):
                add(f'{label}_k{k}_sw8',cfg,mm,nn,k,sw=8,group='pad')
                add(f'{label}_k{k}_sw1',cfg,mm,nn,k,sw=1,group='pad')
        for label,mm,nn,k in (('bw3584_k4096',3584,3584,4096),('bw3584_k8192',3584,3584,8192),('bw3584_k12288',3584,3584,12288),
                              ('bw2560_k16384',2560,2560,16384),('bw5120_k8192',5120,5120,8192)):
            add(label,cfg,mm,nn,k,group='bw')
    return rows


def prepare(root,v08):
    if root.exists():raise ValueError('new directory required')
    root.mkdir(parents=True)
    shutil.copytree(v08/'source',root/'source');shutil.copytree(v08/'build',root/'build')
    shutil.copy2(Path(__file__),root/'source'/'run_v08_followup.py')
    for name in ['v08_model.py','analyze_v08.py','v08_fit.py']:shutil.copy2(ROOT/name,root/'source'/name)
    common.write_json(root/'cases.json',matrix())
    common.write_json(root/'run_config.json',dict(family='v08-followup',cases_sha256=common.sha(root/'cases.json'),v08=str(v08)))
    common.write_json(root/'source_hashes.json',{str(p.relative_to(root)):common.sha(p) for p in (root/'source').rglob('*') if p.is_file()})
    common.write_json(root/'build/origin.json',dict(copied_from=str(v08/'build'),binary_hashes_sha256=common.sha(root/'build/binary_hashes.json')))
    print('prepared',len(matrix()),'cases')


if __name__=='__main__':
    # prepare here; setup/sample use run_v08 unchanged (rows are in the 'ctrl' set: no freeze involved).
    if sys.argv[1]!='prepare':raise SystemExit('use run_v08.py setup|sample --set ctrl on the node')
    prepare(Path(sys.argv[sys.argv.index('--output')+1]).resolve(),Path(sys.argv[sys.argv.index('--v08')+1]).resolve())
