"""Actual S14 full-byte/full-word and closed public transaction implementation proof.

Requires independently reviewed resolver/transaction B1. The output remains
unsigned: another implementer must review the real pack/bridge/core revision.
"""
import argparse,time,resource,json,shutil
from pathlib import Path
from unittest.mock import patch
from common.suite_io import require,read_json,sha
from common.packed_evidence import PackedEvidence
from common.s14_packed_observer import S14EvidenceObserver
from common.s14_packed_policy import validate_original84
from audit_s14_evidence_pack import prepare_original84
from pack_s14_evidence import CONTRACT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--pack',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();repo=args.repo.resolve();pack=args.pack.resolve();out=args.output.resolve();require(not out.exists(),'preserve prior transaction proofs');out.mkdir(parents=True)
    suite=repo/'results/gh200_resource_campaign/20261001-resource-suite-v2';gate=suite/'reviews/S14-packed-transaction-B1-review.json';require(sha(gate)=='6acb6b5f0523367daef20e221bd20d25e9b066b8ce395e35973faa905478e3f3' and read_json(gate)['status']=='pass','independent transaction B1 required')
    names=['common/packed_evidence.py','common/packed_transaction.py','common/s14_packed_values.py','common/s14_packed_observer.py','common/s14_packed_policy.py','common/family_b3.py','common/suite_io.py','auditors/tma_bulk.py','auditors/tma_bulk_sass_baseline.py','auditors/tma_bulk_validation.py','auditors/memory_baseline.py','audit_s14_evidence_pack.py','pack_s14_evidence.py','replay_s14_packed_transaction.py']
    sources={};campaign=repo/'microbench/gh200_resource_campaign';start_wall_ns=time.time_ns()
    for name in names:
        source=campaign/name;dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);sources[name]=sha(source);require(sha(dest)==sources[name],'source snapshot exact')
    closure=read_json(pack/'closure.json');construction=read_json(pack/'construction.json');descriptors,small=prepare_original84(repo,closure);coverage=read_json(repo/closure['roots']['coverage']);sass={r['run_path']+'/build/sass.stdout':closure['members'][r['run_path']+'/build/sass.stdout']['bytes'] for r in coverage['records']};observer=S14EvidenceObserver(descriptors,sass,read_json(repo/CONTRACT))
    counts={};real_sha=sha
    def observed_sha(path):
        path=Path(path).resolve()
        if path in (pack/'original84.tar.xz',pack/'index.json'):
            entry=counts.setdefault(path.name,{'calls':0,'bytes_read':0});entry['calls']+=1;entry['bytes_read']+=path.stat().st_size
        return real_sha(path)
    start=time.monotonic()
    with patch('common.packed_evidence.sha',side_effect=observed_sha):
        bundle=PackedEvidence(pack/'original84.tar.xz',pack/'index.json',expected_index_sha256=construction['index_sha256'],expected_archive_sha256=construction['archive_sha256'],expected_closure=closure['members'])
        verified=bundle.verify_all(small_objects=small,observer=observer);proof=observer.finish();before={name:dict(value) for name,value in counts.items()}
        with bundle.verified_transaction(small) as view:
            normalized=validate_original84(view,closure,proof)
        transaction=bundle.transaction_receipt
        require(transaction['closed'] and transaction['qualification_granted'] is False,'only closed unqualified transaction receipt')
        require(counts['original84.tar.xz']['calls']-before['original84.tar.xz']['calls']==2,'exact entry/exit archive identity checks')
        try:view.inventory()
        except ValueError:closed_view_rejected=True
        else:raise ValueError('closed transaction view still active')
    elapsed=time.monotonic()-start;stop_wall_ns=time.time_ns()
    for name,digest in sources.items():require(sha(campaign/name)==digest,'implementation source changed during proof')
    with (out/'normalized-original84.json').open('x') as f:json.dump(normalized,f,sort_keys=True,indent=2);f.write('\n')
    with (out/'complete-observer-proof.json').open('x') as f:json.dump(proof,f,sort_keys=True,indent=2);f.write('\n')
    receipt={'status':'actual_original84_public_transaction_passed_pending_independent_bridge_B','archive_verification':verified,'transaction':transaction,'hash_calls':counts,'hash_calls_before_transaction':before,'closed_view_rejected':closed_view_rejected,'historical_contexts':normalized['historical_contexts'],'profile_processes':84,'complete_payload_guard_words':proof['complete_values']['payload_guard_words'],'complete_lifecycle_words':proof['complete_values']['lifecycle_words'],'start_wall_ns':start_wall_ns,'stop_wall_ns':stop_wall_ns,'elapsed_seconds':elapsed,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'source_sha256':sources,'normalized_sha256':sha(out/'normalized-original84.json'),'observer_proof_sha256':sha(out/'complete-observer-proof.json'),'bridge_qualified':False,'GPU_execution':False}
    with (out/'receipt.json').open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({k:receipt[k] for k in ['status','hash_calls','historical_contexts','complete_payload_guard_words','complete_lifecycle_words','elapsed_seconds','peak_rss_kib','bridge_qualified','GPU_execution']},indent=2))

if __name__=='__main__':main()
