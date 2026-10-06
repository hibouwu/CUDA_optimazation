"""S12 integrated formal ABI; original source-only adapter stays unchanged."""
from auditors import async_copy_formal
from common.family_b3 import ADAPTER, RESOURCE_FIELDS, enabled
from common.suite_io import require

ADAPTER_ID = ADAPTER
validate_device = async_copy_formal.validate_device
audit_sass = async_copy_formal.audit_sass
work = async_copy_formal.work


def validate_contract(contract):
    require(enabled(contract), 'S12 formal requires B3 revision')
    legacy = {**contract, 'adapter_id': async_copy_formal.ADAPTER_ID}
    async_copy_formal.validate_contract(legacy)
    return contract


def validate_trial(row, case, device, seed, protocol):
    require('b3_resource_identity' in case and 'b3_blocks' in case, 'B3 admission resources missing')
    resource = case['b3_resource_identity']
    require(all(row.get(key) == resource[key] for key in RESOURCE_FIELDS), 'formal trial resources differ from B3')
    require(row.get('blocks') == case['b3_blocks'], 'formal grid differs from B3')
    return async_copy_formal.validate_trial(row, case, device, seed, protocol)
