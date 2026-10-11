from pathlib import Path
import re


def validate(text):
    fields=dict(re.findall(r'(\w+)=(\S+)',text))
    assert fields['NodeList']=='romeo-a058' and fields['NumNodes']=='1'
    assert fields['NumCPUs']=='288' and fields['OverSubscribe']=='NO'
    tres=dict(item.split('=',1) for item in fields['AllocTRES'].split(','))
    assert tres['cpu']=='288' and tres['gres/gpu']=='4' and tres['node']=='1'


if __name__=='__main__':
    validate(Path('allocation.txt').read_text())
