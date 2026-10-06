"""Finite transport of independently qualified S14 capture=false B3 evidence.

This creates unsigned derived facts, never family admission or execution.
Numeric predicates were checked by the pinned independent B3 review; this
bridge proves the exact bytes and namespaces to which that review applies.
"""
from pathlib import Path
from common.packed_evidence import PackedEvidence
from common.suite_io import gate_mapping, read_json, relative, require, sha, validate_gate

SUITE = 'results/gh200_resource_campaign/20261001-resource-suite-v2'
REVIEW = SUITE + '/reviews/S14-formal-short-B3-review.json'
REVIEW_SHA = '62482777dcfd4e9651a4a009c4e2745eadde6a4b924ed03e61de8fcb442f60ff'
COVERAGE = SUITE + '/reviews/evidence/S14-formal-short-B3/coverage.json'
COVERAGE_SHA = 'e8c553bec3b2ba8c749c59e2287d97ed9bb5e1a3d20fabbf50c38e3965847fc2'
ARCHIVE = SUITE + '/implementation/short-collection-v2/s14-formal-short-collected/gh200-s14-formal-short-files-v1.tar.xz'
ARCHIVE_SHA = '7687a296f7f4938ebe3c2ede670292e24be871dace82e962bb6ad6951b93ca21'
INDEX = SUITE + '/implementation/s14-formal-short-packed-index-a/index.json'
INDEX_SHA = '5841639f53f0f16d5fff5db52999398fe19d7790188847da17877ccbb48f197a'
CLOSURE = SUITE + '/implementation/s14-formal-short-packed-index-a/closure.json'
CLOSURE_SHA = 'c9edd1e0e68fb68abe2a2dd6d6c86d66e6b5df26fa4f97e2164009fa1fd2a5c0'


def construct(repo):
    repo = Path(repo).resolve()
    for name, digest in ((REVIEW, REVIEW_SHA), (COVERAGE, COVERAGE_SHA),
                         (INDEX, INDEX_SHA), (CLOSURE, CLOSURE_SHA)):
        require(sha(relative(repo, name)) == digest, 'pinned S14 bridge input changed: ' + name)
    review = validate_gate(relative(repo, REVIEW), repo, 'S14', 'early-validation-B3')
    require(review['authorization']['family_B3_qualified'] is True
            and review['authorization']['capture_false_final_path_qualified'] is True
            and review['authorization']['formal_sampling'] is False,
            'capture=false independent B3 purpose')
    require(review['coverage_record'] == {'path': COVERAGE, 'sha256': COVERAGE_SHA},
            'direct signed coverage reference')
    coverage = read_json(relative(repo, COVERAGE))
    require(coverage['status'] == 'independent_capture_false_all72_final_values_and_code_pass'
            and coverage['capture_enabled'] is False and coverage['processes'] == 24
            and coverage['target_launches'] == 72 and len(coverage['records']) == 24,
            'finite new-branch coverage')
    require(coverage['payload_guard_words'] == 1461766464
            and coverage['lifecycle_and_stamp_words'] == 365040
            and coverage['CTA_launch_records'] == 18252, 'pinned complete B3 counts')
    closure = read_json(relative(repo, CLOSURE))
    bundle = PackedEvidence(relative(repo, ARCHIVE), relative(repo, INDEX),
                            expected_index_sha256=INDEX_SHA,
                            expected_archive_sha256=ARCHIVE_SHA,
                            expected_closure=closure['members'])
    # Explicitly map only signed canonical run members into the physical repo/
    # namespace. Other review artifacts remain separately bound; no basename
    # lookup, executable import from metadata, or permissive fallback exists.
    packed_bindings, external_bindings = {}, {}
    roots = {record['run_path'] for record in coverage['records']}
    require(len(roots) == 24 and all(root.startswith(SUITE + '/tma_bulk/formal-short-v1-')
                                   for root in roots), 'canonical new run roots')
    for name, digest in gate_mapping(review).items():
        owners = [root for root in roots if name.startswith(root + '/')]
        if owners:
            require(len(owners) == 1, 'ambiguous canonical run owner')
            physical = 'repo/' + name
            require(physical in bundle.members and bundle.members[physical]['sha256'] == digest,
                    'signed run byte absent from exact physical namespace')
            packed_bindings[name] = {'physical_member': physical, 'sha256': digest}
        else:
            require(sha(relative(repo, name)) == digest, 'external signed review artifact changed')
            external_bindings[name] = digest
    metadata = set()
    for root in roots:
        for local in ('validation_manifest.json', 'validation_state.json', 'validation_spec.json',
                      'attempts/attempt_00/raw.jsonl', 'attempts/attempt_00/receipt.json'):
            metadata.add('repo/' + root + '/' + local)
    verified = bundle.verify_all(small_objects=sorted(metadata))
    records = []
    with bundle.verified_transaction(sorted(metadata)) as view:
        seen = set()
        for record in coverage['records']:
            root = record['run_path']; prefix = 'repo/' + root + '/'
            manifest = view.read_json(prefix + 'validation_manifest.json')
            require(view.file_sha256(prefix + 'validation_manifest.json') == record['manifest_sha256'],
                    'signed original manifest identity')
            for local, digest in manifest.items():
                require(view.file_sha256(prefix + local) == digest, 'complete original manifest member')
            require(view.read_json(prefix + 'validation_state.json')['status'] == 'case_diagnostic_passed',
                    'original diagnostic terminal state')
            spec = view.read_json(prefix + 'validation_spec.json')
            require(spec['case_id'] == record['case_id'] and spec['seed'] == record['seed'] == 3
                    and spec['profile_id'] == record['profile_id'] == 'formal_final_1_2_33_v1',
                    'signed fixed case/profile/top seed')
            require(record['case_id'] not in seen and record['status'] == 'pass'
                    and record['target_launches'] == 3, 'unique qualified finite record')
            seen.add(record['case_id'])
            require(view.file_sha256(prefix + 'attempts/attempt_00/receipt.json') == record['receipt_sha256'],
                    'signed process receipt identity')
            records.append(record)
    return {'status': 'unsigned_exact_qualified_parent_transport_constructed',
            'parent_review': {'path': REVIEW, 'sha256': REVIEW_SHA},
            'parent_coverage': {'path': COVERAGE, 'sha256': COVERAGE_SHA},
            'archive': {'path': ARCHIVE, 'sha256': ARCHIVE_SHA},
            'index': {'path': INDEX, 'sha256': INDEX_SHA},
            'closure': {'path': CLOSURE, 'sha256': CLOSURE_SHA},
            'packed_bindings': packed_bindings, 'external_bindings': external_bindings,
            'records': records, 'verification': verified,
            'transaction': bundle.transaction_receipt,
            'device': coverage['device'], 'environment': coverage['environment'],
            'target_facts': coverage['target_facts'],
            'numeric_predicates_reexecuted_here': False,
            'numeric_qualification_source': 'pinned independent B3 with unchanged full raw bytes',
            'GPU_execution': False, 'bridge_qualified': False,
            'formal_sampling': False, 'family_B3_qualified': False}
