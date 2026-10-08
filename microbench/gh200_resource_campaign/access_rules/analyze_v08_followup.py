#!/usr/bin/env python3
"""V08 follow-up analysis: pitch/tail factorial, swizzle padding, large-footprint long K."""
import argparse,json,statistics
from pathlib import Path
import v06_run as common
import v08_model as model

med=statistics.median


def per_ktile(c,key):
    v=c['intervals'].get(key);return v/c['kt'] if v else None


def analyze(root,output):
    common.verify(root)
    rows=json.loads((root/'cases.json').read_text());setups={s['case']:s['setup'] for s in json.loads((root/'static_setup.json').read_text())}
    S={r['id']:dict(model.summarize_case(root,r,setups[r['id']]),group=r['group'],lda=r['lda'],ldb=r['ldb'],ldd=r['ldd'],m=r['m'],n=r['n'],k=r['k'],swizzle=r['swizzle']) for r in rows}
    out=dict(pitch=[],pad=[],bw=[])
    for cfg in ['cfg_a','cfg_b','cfg_c']:
        for k in (1024,4096):
            ref=S[f'{cfg}_p_ref_k{k}']
            names=[f'p_ref_k{k}',f'p_apitch_k{k}',f'p_bpitch_k{k}']+(['p_dpitch_k1024','p_ktail','p_ktail_apitch','p_ntail','p_ntail_bdpitch'] if k==1024 else [])
            for name in names:
                c=S[f'{cfg}_{name}']
                out['pitch'].append(dict(case=c['id'],config=cfg,k=c['k'],lda=c['lda'],ldb=c['ldb'],ldd=c['ldd'],n=c['n'],plain_us=c['plain_us'],
                    time_vs_ref=c['plain_us']/ref['plain_us']-1,L0_vs_ref=c['intervals']['L0']/ref['intervals']['L0']-1,
                    L_vs_ref=(c['intervals']['L']/ref['intervals']['L']-1) if 'L' in c['intervals'] and 'L' in ref['intervals'] else None,
                    Elast=c['E_last'],Elast_ref=ref['E_last'],S=c['intervals']['S'],S_ref=ref['intervals']['S']))
        for label in ('padM','padN','padMN'):
            for k in (1024,4096):
                a,b=S[f'{cfg}_{label}_k{k}_sw8'],S[f'{cfg}_{label}_k{k}_sw1']
                tl=a['tile_L'];base=tl.get('in',{}).get('median')
                out['pad'].append(dict(case=a['id'],config=cfg,k=k,kt=a['kt'],T_sw8=a['T'],T_sw1=b['T'],plain_sw8=a['plain_us'],plain_sw1=b['plain_us'],
                    time_ratio=a['plain_us']/b['plain_us']-1,tile_L={x:dict(v,ratio=v['median']/base-1 if base else None,extra_per_ktile=(v['median']-base)/a['kt'] if base else None) for x,v in tl.items()}))
        for label in ('bw3584_k4096','bw3584_k8192','bw3584_k12288','bw2560_k16384','bw5120_k8192'):
            c=S[f'{cfg}_{label}']
            out['bw'].append(dict(case=c['id'],config=cfg,m=c['m'],n=c['n'],k=c['k'],fp=c['fp'],T=c['T'],plain_us=c['plain_us'],ghz=c['ghz_ends'],
                L0_per_kt=per_ktile(c,'L0'),L_per_kt=per_ktile(c,'L'),c_max=c['c_max_stamped']))
    output.mkdir(parents=True,exist_ok=False);common.write_json(output/'followup.json',out);common.write_json(output/'summary.json',S)
    for r in out['pitch']:print(f"{r['case']:28s} lda{r['lda']:5d} ldb{r['ldb']:5d} ldd{r['ldd']:5d} time {r['time_vs_ref']*100:+6.1f}% L0 {r['L0_vs_ref']*100:+6.1f}% L {'' if r['L_vs_ref'] is None else '%+6.1f%%'%(r['L_vs_ref']*100)} Elast {r['Elast']:.0f}/{r['Elast_ref']:.0f}")
    for r in out['pad']:print(f"{r['case']:24s} T {r['T_sw1']}->{r['T_sw8']} time {r['time_ratio']*100:+6.1f}% ",{x:(v['n'],round(v['ratio'],3) if v['ratio'] is not None else None,round(v['extra_per_ktile'] or 0)) for x,v in r['tile_L'].items()})
    for r in out['bw']:print(f"{r['case']:22s} fp {r['fp']:6.0f} T{r['T']} L0/kt {r['L0_per_kt']:.1f} L/kt {r['L_per_kt'] or 0:.1f} ghz {r['ghz']:.3f} us {r['plain_us']:.1f}")
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();analyze(a.run.resolve(),a.output.resolve())
