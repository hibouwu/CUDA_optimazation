"""Bind host availability and output timing of the existing measurement program."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent;base=root.parents[2]
build=base/'20261011-cfg-c-producer-tail-check-build'
path=build/'source/probes/r18.cu'
text=path.read_text();digest=hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((build/'source_hashes.json').read_text())
assert manifest['source/probes/r18.cu']==digest
def line(fragment,start=0):
    pos=text.index(fragment,start)
    return pos,text[:pos].count('\n')+1
begin,_=line('auto timed =')
report_start,_=line('auto report =',begin)
sync=line('cudaEventSynchronize(call_end)',begin)
elapsed=line('cudaEventElapsedTime(&ms, call_begin, call_end)',begin)
returned=line('return ms;',begin)
assert begin<sync[0]<elapsed[0]<returned[0]<report_start
common=line('if (common_history) {',report_start)
warm=line('warm.push_back(timed(warmup_launch_params));',common[0])
target_flag=line('set_target_epilogue();',warm[0])
target=line('float ms = timed(launch_params);',target_flag[0])
reported=line('report("warm", ms, warm);',target[0])
assert warm[0]<target_flag[0]<target[0]<reported[0]
printed=line('print_f("warmup_us", warm_us);',report_start)
assert printed[0]<common[0]
block=text[common[0]:reported[0]]
assert 'std::cout' not in block and 'std::cin' not in block
result=dict(source_sha256=digest,source_path=str(path),
    source_manifest_sha256=hashlib.sha256((build/'source_hashes.json').read_bytes()).hexdigest(),
    source_lines=dict(event_synchronize=sync[1],event_elapsed=elapsed[1],timed_return=returned[1],
        common_prefix_push=warm[1],set_target=target_flag[1],target_timed=target[1],
        report_call=reported[1],prefix_print_inside_report=printed[1]),
    prefix_values_in_host_before_target=True,
    prefix_values_exported_before_target=False,
    existing_binary_generates_numeric_prediction=False,
    supported_validation='Freeze deterministic rule and allowed inputs before new data; evaluate it using only prefix fields afterward, explicitly recording later evaluation.',
    unsupported_claim='Numeric predictions were generated or externally committed before the target launch.',
    observation_qualification_unchanged=True,
    limits='Source flow evidence bound to the previously verified build; not a fresh binary execution, observation qualification or new holdout.')
(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
