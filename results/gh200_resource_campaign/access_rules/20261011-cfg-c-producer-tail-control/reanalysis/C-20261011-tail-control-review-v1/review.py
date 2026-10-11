#!/usr/bin/env python3
"""C-only offline review. Never invoke the public acquisition or analysis runner."""
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True
out = Path(__file__).resolve().parent
run = out.parents[1]
data = run/'returned'
repo = run.parents[3]
commit = 'b11fae8064489fc71d1b688aa59daa396e8751c2'
inputs = {}

def sha(path):
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs[str(path)] = value
    return value

def read(path):
    sha(path)
    return json.loads(path.read_text())

manifest = read(run/'manifest.json')
freeze = read(run/'freeze.json')
assert manifest['status'] == 'frozen'
assert manifest['gpu_uuid'] == 'GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114'
assert manifest['gates'] == dict(per_common_series_median_absolute_ns=64,
    per_common_series_maximum_absolute_ns=256, event_median_absolute_ratio=.02,
    event_maximum_absolute_ratio=.05, off_event_population_cv=.05)
for name, digest in freeze['files'].items():
    assert sha(run/name) == sha(data/name) == digest, name
assert sha(data/'freeze.json') == sha(run/'freeze.json')
for name in ('freeze.json', 'input.tar.gz'):
    path = run/name
    blob = subprocess.check_output(['git', 'show', commit+':'+str(path.relative_to(repo))], cwd=repo)
    assert hashlib.sha256(blob).hexdigest() == sha(path)

def check_archive(path, directory, expected):
    sha(path)
    files = {}
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = member.name.removeprefix('./')
            assert name not in files and not name.startswith('/') and '..' not in Path(name).parts
            digest = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            assert sha(directory/name) == digest, name
            files[name] = digest
    assert set(expected) <= set(files)
    return len(files)

input_files = check_archive(run/'input.tar.gz', run, [*freeze['files'], 'freeze.json'])
receipt = read(run/'receipt.json')
assert receipt['returncode'] == 0 and sha(run/'output.tar.gz') == receipt['sha256']
output_files = check_archive(run/'output.tar.gz', data, [*freeze['files'], 'freeze.json', 'complete.json'])
assert (data/'exit_code.txt').read_text().strip() == '0'
sha(data/'exit_code.txt')
build = Path(manifest['build_run'])
source_hashes = read(run/'origin-source_hashes.json')
assert sha(build/'source_hashes.json') == manifest['origin_source_hashes_sha256']
assert sha(build/'build-audit.json') == manifest['origin_build_audit_sha256']
for name, digest in source_hashes.items():
    assert sha(build/name) == digest
assert sha(build/'returned/build/cfg_c_s4_dual') == manifest['binary_sha256']

snapshot = out/'analysis_snapshot'
snapshot.mkdir(exist_ok=True)
for path in (run/'analysis_snapshot').glob('*.py'):
    shutil.copyfile(path, snapshot/path.name)
    assert sha(snapshot/path.name) == sha(path)
shutil.copyfile(run/'validate_allocation.py', snapshot/'validate_allocation.py')
assert sha(snapshot/'validate_allocation.py') == sha(run/'validate_allocation.py')
packet_tests = repo/'microbench/gh200_resource_campaign/access_rules/test_cfg_c_producer_tail_packet.py'
shutil.copyfile(packet_tests, snapshot/packet_tests.name)
assert sha(snapshot/packet_tests.name) == sha(packet_tests)
test = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(snapshot),
    '-p', 'test_cfg_c*tail*.py'], capture_output=True, text=True,
    env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
(out/'tests.txt').write_text(test.stdout+test.stderr)
assert test.returncode == 0 and 'Ran 16 tests' in test.stderr
sys.path.insert(0, str(snapshot))
from analyze_cfg_c_tail_control import process, evaluate
from validate_allocation import validate

allocation = (data/'allocation.txt').read_text()
sha(data/'allocation.txt')
validate(allocation)
fields = dict(re.findall(r'(\w+)=(\S+)', allocation))
assert fields['JobId'] == '740091' and fields['Restarts'] == '0'
context = read(data/'assigned-context.json')
assert context['SLURM_JOB_ID'] == '740091' and context['SLURM_STEP_GPUS'] == '0,1,2,3'
assert context['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'
setup = json.loads((data/'setup.stdout').read_text())
sha(data/'setup.stdout')
assert setup['event'] == 'setup' and setup['gpu_uuid'] == manifest['gpu_uuid']

records = read(data/'records.json')
assert len(records) == 20
assert [{k:r[k] for k in ('id','pair','flag','command')} for r in records] == manifest['samples']
assert read(data/'complete.json') == dict(manifest_sha256=sha(run/'manifest.json'), records=20)
smoke = read(run.parent/'20261011-cfg-c-producer-tail-correctness/manifest.json')
smoke_commands = {int(s['id'][-1]):s['command'] for s in smoke['samples'] if s['id'].startswith('flag')}
assert smoke['binary_sha256'] == manifest['binary_sha256'] and smoke['build_run'] == manifest['build_run']
for i, sample in enumerate(manifest['samples']):
    pair = i//2
    assert type(sample['pair']) is int and sample['pair'] == pair
    assert type(sample['flag']) is int and sample['flag'] == (i%2 if pair%2 == 0 else 1-i%2)
    assert sample['command'] == smoke_commands[sample['flag']]

# Independent integer reference, without importing the campaign reference.
reference = []
for row in range(17):
    line = []
    for col in range(17):
        products = [((7*row+13*k)%17-8)*((5*k+11*col)%17-8) for k in range(17)]
        line.append((sum(products)*240+sum(products[:16]))/1024)
    reference.append(line)

expected_counts = dict(producer=132, GLOBAL_ENTRY=132, GLOBAL_PRODUCER=132,
    FINAL=264, GLOBAL_FINAL=264, TAIL=264,
    **{name:432 for name in ('FIRST','MAIN','PERMIT','DONE','L','W','E','GAP',
                            'GLOBAL_FIRST','GLOBAL_MAIN','GLOBAL_PERMIT','GLOBAL_DONE')})
processes = []
telescoping = global_identities = numeric_values = 0
for sample in records:
    raw = data/sample['raw']
    assert sample['returncode'] == 0 and sha(raw) == sample['sha256']
    with gzip.open(raw, 'rt') as stream:
        events = [json.loads(line) for line in stream]
    packets = {p['event']:p for p in events}
    assert set(packets) == {'setup','call','check'} and len(events) == 3
    packet = process(events, sample, manifest['gpu_uuid'])
    trace = packets['call']['trace']
    origin = min(trace[1296*cta+1] for cta in range(132))
    independent = {}
    for index, value in zip(packets['check']['checked_indices'], packets['check']['checked_values']):
        assert value == reference[(index//3072)%17][(index%3072)%17]
        numeric_values += 1
    for cta in range(132):
        head = trace[1296*cta:1296*(cta+1)]
        entry = [head[1],head[0]]
        producer = [head[13],head[3]]
        independent[f'{cta}/GLOBAL_ENTRY'] = [entry[0]-origin,None]
        independent[f'{cta}/GLOBAL_PRODUCER'] = [producer[0]-origin,None]
        independent[f'{cta}/producer'] = [producer[d]-entry[d] for d in range(2)]
        for wg in range(2):
            final = [head[5+3*wg],head[4+3*wg]]
            independent[f'{cta}/{wg}/GLOBAL_FINAL'] = [final[0]-origin,None]
            independent[f'{cta}/{wg}/FINAL'] = [final[d]-entry[d] for d in range(2)]
            previous = producer
            accumulated = independent[f'{cta}/producer'][:]
            for j in range(head[6+3*wg]):
                tile = head[16+640*wg+10*j:26+640*wg+10*j]
                coord = f'{tile[4]-1},{tile[5]-1}'
                prefix = f'{cta}/{wg}/{j}/{coord}/'
                stamps = [[tile[6+k],tile[k]] for k in range(4)]
                for name, stamp in zip(('FIRST','MAIN','PERMIT','DONE'), stamps):
                    independent[prefix+name] = [stamp[d]-entry[d] for d in range(2)]
                    independent[prefix+'GLOBAL_'+name] = [stamp[0]-origin,None]
                    assert independent[prefix+'GLOBAL_'+name][0] == independent[prefix+name][0]+entry[0]-origin
                    global_identities += 1
                gaps = [stamps[0][d]-previous[d] for d in range(2)]
                independent[prefix+'GAP'] = gaps
                for name, a, b in (('L',0,1),('W',1,2),('E',2,3)):
                    independent[prefix+name] = [stamps[b][d]-stamps[a][d] for d in range(2)]
                for d in range(2):
                    accumulated[d] += gaps[d]+sum(independent[prefix+name][d] for name in ('L','W','E'))
                previous = stamps[3]
            tail = [final[d]-previous[d] for d in range(2)]
            independent[f'{cta}/{wg}/TAIL'] = tail
            for d in range(2):
                assert accumulated[d]+tail[d] == final[d]-entry[d]
                telescoping += 1
    assert independent == packet['series']
    assert len(independent) == 6372
    assert dict(collections.Counter(k.split('/')[-1] for k in independent)) == expected_counts
    processes.append(packet)

# Recompute each prescribed gate directly from the raw-derived integer series.
local = []
event_signed = []
off_times = []
pairs = []
for pair in range(10):
    rows = processes[2*pair:2*pair+2]
    off, on = sorted(rows,key=lambda p:p['flag'])
    assert off['flag'] == 0 and on['flag'] == 1
    pairs.append((off,on))
    off_times.append(off['elapsed_us'])
    event_signed.append(on['elapsed_us']/off['elapsed_us']-1)
for key in sorted(processes[0]['series']):
    signed = [on['series'][key][0]-off['series'][key][0] for off,on in pairs]
    absolute = sorted(map(abs,signed))
    median = (absolute[4]+absolute[5])/2
    maximum = absolute[-1]
    local.append(dict(series=key,signed=signed,median_absolute=median,maximum_absolute=maximum,
        passed=median<=64 and maximum<=256,
        native_cycle_deltas=[None if off['series'][key][1] is None else on['series'][key][1]-off['series'][key][1] for off,on in pairs],
        sm_changed=[off['sms'][key.split('/')[0]] != on['sms'][key.split('/')[0]] for off,on in pairs]))
absolute = sorted(map(abs,event_signed))
event = dict(signed=event_signed,median_absolute=(absolute[4]+absolute[5])/2,maximum_absolute=absolute[-1],
    passed=(absolute[4]+absolute[5])/2<=.02 and absolute[-1]<=.05)
cv = statistics.pstdev(off_times)/statistics.fmean(off_times)
main = read(run/'analysis.json')
assert main['local'] == local and main['event'] == event and main['off_event_population_cv'] == cv
frozen_result = evaluate(processes,manifest)
for key,value in frozen_result.items():
    assert main[key] == value
assert main['manifest_sha256'] == sha(run/'manifest.json')
assert main['processes_verified'] == 20 and main['series_per_process'] == 6372
assert numeric_values == 81920
assert sum(not x['passed'] for x in local) == 4771
assert not any(any(x['sm_changed']) for x in local)
assert event['passed'] and cv <= .05
for key in ('incremental_control_passed','observation_qualified','tail_precision_qualified','baseline_dual_qualified'):
    assert main[key] is False
groups = {}
for name in expected_counts:
    rows = [x for x in local if x['series'].split('/')[-1] == name]
    groups[name] = dict(series=len(rows),failed=sum(not x['passed'] for x in rows),
        median_abs_range_ns=[min(x['median_absolute'] for x in rows),max(x['median_absolute'] for x in rows)],
        maximum_abs_range_ns=[min(x['maximum_absolute'] for x in rows),max(x['maximum_absolute'] for x in rows)])
assert groups['L']['failed'] == 432 and groups['L']['maximum_abs_range_ns'] == [320,1280]
summary = read(run/'summary.json')
assert summary['total_series'] == 6372 and summary['failed_series'] == 4771
assert summary['event'] == event and summary['off_cv'] == cv
for name, group in groups.items():
    for key,value in group.items():
        assert summary['groups'][name][key] == value

report = dict(freeze_commit=commit,job_id=740091,input_sha256=inputs,
    frozen_files_verified=len(freeze['files']),input_archive_files_verified=input_files,
    output_archive_files_verified=output_files,source_files_verified=len(source_hashes),
    cpu_tests_passed=16,commands_equal_to_correctness_smoke=20,
    processes_verified=20,ctas_per_process=132,roles_per_process=264,series_per_process=6372,
    independent_numeric_values_verified=numeric_values,integer_telescoping_identities=telescoping,
    shared_origin_endpoint_identities=global_identities,failed_series=4771,groups=groups,
    event=event,off_event_population_cv=cv,paired_sm_changes=0,
    independent_raw_series_exactly_match_frozen_process=True,
    independent_gate_recalculation_exactly_matches_main=True,
    specification_findings=['W/E/GAP/TAIL directly gated; opposed endpoint movement cannot hide in endpoint-only gates.',
        'Min-entry common ns origin retains relative whole-CTA phase; global native cycle fields remain null.',
        'All 6372 prescribed series are present in every process; gates are applied separately without pooling or centering.',
        'TAIL means consumer FINAL minus last DONE; it is not the new producer-tail timestamp.',
        'Passing this control could not qualify tail precision or baseline dual; this control actually failed.'],
    incremental_control_passed=False,observation_qualified=False,tail_precision_qualified=False,
    baseline_dual_qualified=False,new_gpu_runs_by_C=0,public_code_modified=False,
    causal_attribution='No attribution to added tail instruction overhead or a physical cause. Fixed paired observations only.')
(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','groups','specification_findings')},indent=2))
