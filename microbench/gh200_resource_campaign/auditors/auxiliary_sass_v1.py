"""Full-code comparison to frozen actual CPU compilation; no source-only admission."""
from pathlib import Path
import hashlib
import json
import re
from common.suite_io import require

BASELINE_PATH = Path(__file__).resolve().parents[1] / 'contracts/auxiliary_sass_baseline_v1.draft.json'

def audit_sass(text, contract):
    baseline = json.loads(BASELINE_PATH.read_text())['targets']
    require(len(baseline) == 19, 'finite nineteen CUDA target baseline')
    parts = re.split(r'Function\s*:\s*', text)[1:]
    names = [part.splitlines()[0].strip() for part in parts]
    require(len(names) == len(set(names)) == 19 and set(names) == set(baseline), 'complete nineteen CUDA targets')
    out = []
    for name, part in zip(names, parts):
        lines = part.splitlines()
        rows = []
        for i, line in enumerate(lines):
            match = re.search(r'/\*([0-9a-f]+)\*/\s*(.*?)\s*/\* 0x([0-9a-f]{16}) \*/', line)
            if match:
                require(i + 1 < len(lines), 'missing second instruction word')
                high = re.search(r'/\* 0x([0-9a-f]{16}) \*/', lines[i + 1])
                require(high is not None, 'complete 128-bit instruction')
                rows.append({'pc': match[1], 'instruction': match[2].strip(), 'low': match[3], 'high': high[1]})
        require(rows and [int(row['pc'], 16) for row in rows] == list(range(0, 16 * len(rows), 16)), 'complete contiguous target')
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        require(digest == baseline[name]['full128_sha256'] and len(rows) == baseline[name]['instruction_count'],
                'actual target differs from frozen current baseline; independent requalification required: ' + name)
        out.append({'kernel_symbol': name, 'instruction_count': len(rows), 'full128_sha256': digest,
                    'comparison_only': True, 'GPU_admission': False})
    return out
