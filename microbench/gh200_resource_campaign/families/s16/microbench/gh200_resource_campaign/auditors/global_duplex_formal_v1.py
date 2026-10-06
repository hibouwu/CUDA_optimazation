"""S13 formal admission wrapper; numerical/source-only auditor stays frozen."""
import json
from auditors import global_duplex
from common.family_b3 import RESOURCE_FIELDS, enabled
from common.suite_io import require

ADAPTER_ID = 'global_duplex_formal_v1'
validate_device = global_duplex.validate_device
def audit_sass(text, contract):
    # The core persists this evidence as JSON and compares it on offline load.
    # Normalize tuple containers without changing any original audit decision.
    return json.loads(json.dumps(global_duplex.audit_sass(text, contract)))


def validate_contract(contract):
    require(contract.get('adapter_id') == ADAPTER_ID and enabled(contract),
            'S13 formal requires its finite B3 revision')
    global_duplex.validate_contract({**contract, 'adapter_id': global_duplex.ADAPTER_ID})
    return contract


def validate_trial(row, case, device, seed, protocol):
    require('b3_resource_identity' in case and 'b3_blocks' in case,
            'S13 B3 admission resources missing')
    bound = case['b3_resource_identity']
    require(all(row.get(key) == bound[key] for key in RESOURCE_FIELDS),
            'S13 formal resources differ from B3')
    require(row.get('kernel_symbol') == bound['kernel_symbol']
            and row.get('blocks') == case['b3_blocks'],
            'S13 formal target/grid differs from B3')
    require(all(row.get(key) == value for key, value in bound['extensions'].items()),
            'S13 formal allocation/ratio differs from B3')
    return global_duplex.validate_trial(row, case, device, seed, protocol)
