"""读取页对的冷/热加载窗口，检验共享存储域的同向与反向响应。"""
import argparse
import json
from pathlib import Path

import numpy as np


def analyze(path):
    sample=json.loads(path.read_text())
    data=np.asarray(sample['records']).reshape(len(sample['actors']),sample['repeats'],sample['samples'],2,3)
    cold=np.median(data[:,:,:,:,0],axis=1)
    difference=cold[:,:,1]-cold[:,:,0]
    smids=[a['smid'] for a in sorted(sample['actors'],key=lambda a:a['index'])]
    groups=difference.reshape(8,2,sample['samples']).mean(axis=1)
    same_sign=[float(np.mean((difference[i]>0)==(difference[i+1]>0))) for i in range(0,16,2)]
    report=dict(gpu_uuid=sample['gpu_uuid'],allocation_address=sample['allocation_address'],
        samples=sample['samples'],repeats=sample['repeats'],smids=smids,
        cold_quantiles=np.quantile(data[:,:,:,:,0],[0,.1,.25,.5,.75,.9,.99]).tolist(),
        hot_quantiles=np.quantile(data[:,:,:,:,1],[0,.1,.25,.5,.75,.9,.99]).tolist(),
        cold_median_per_sm=np.median(data[:,:,:,:,0],axis=(1,2,3)).tolist(),
        hot_median_per_sm=np.median(data[:,:,:,:,1],axis=(1,2,3)).tolist(),
        pair_sign_agreement=same_sign,sm_correlation=np.corrcoef(difference).tolist(),
        group_correlation=np.corrcoef(groups).tolist(),group_page_differences=groups.tolist(),
        frequency_ghz=[a['cycles']/a['ns'] for a in sample['actors']],pairs=sample['pairs'])
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=analyze(a.input);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print('cold/hot median cycles',r['cold_quantiles'][3],r['hot_quantiles'][3])
    print('GPC correlation with group 0',r['group_correlation'][0])
    print('same GPC sign agreement',r['pair_sign_agreement'])
