"""PTX ISA 9.0 encoders for dense ``tcgen05.mma`` instruction descriptors.

This module deliberately keeps instruction-descriptor encoding independent from
benchmark generation.  A generated benchmark records both the symbolic fields
and the resulting 32-bit value, so a result cannot silently change meaning when
the generator evolves.

Primary source:
https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/
index.html#tcgen05-instruction-descriptor
Tables 42--44, PTX ISA 9.0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


class DescriptorError(ValueError):
    """Raised when fields do not describe a legal v1 dense descriptor."""


F8F6F4_TYPES = {
    "e4m3": 0,
    "e5m2": 1,
    "e2m3": 3,
    "e3m2": 4,
    "e2m1": 5,
}


@dataclass(frozen=True)
class DescriptorRecord:
    kind: str
    m: int
    n: int
    a_type: str
    b_type: str
    d_type: str
    value_u32: int
    scale_type: str | None = None
    a_scale_data_id: int | None = None
    b_scale_data_id: int | None = None
    saturate: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_mn(m: int, n: int, *, block_scaled: bool) -> None:
    if not 8 <= n <= 256 or n % 8:
        raise DescriptorError(f"N={n} must be a multiple of 8 in [8, 256]")
    if block_scaled:
        if m != 128:
            raise DescriptorError(
                f"block-scaled CTA-group-1 M={m} must be 128 on SM110"
            )
    elif m not in {64, 128}:
        raise DescriptorError(f"unscaled CTA-group-1 M={m} must be 64 or 128")


def encode_unscaled(
    kind: str,
    *,
    m: int,
    n: int,
    a_type: str,
    b_type: str | None = None,
    d_type: str | None = None,
    saturate: bool = False,
) -> DescriptorRecord:
    """Encode PTX ISA 9.0 Table 42 for a dense MMA operation.

    ``kind`` is one of ``f16``, ``tf32``, ``f8f6f4`` or ``i8``.  The returned
    integer is the 32-bit instruction descriptor passed to inline PTX; it is
    not the historical helper's 64-bit value shifted into the high word.
    """

    _validate_mn(m, n, block_scaled=False)
    b_type = a_type if b_type is None else b_type
    value = 0

    if kind == "f16":
        type_codes = {"f16": 0, "bf16": 1}
        if a_type not in type_codes or b_type not in type_codes:
            raise DescriptorError("kind=f16 accepts only f16 or bf16 A/B types")
        d_type = "f32" if d_type is None else d_type
        if d_type not in {"f16", "f32"}:
            raise DescriptorError("kind=f16 D type must be f16 or f32")
        value |= {"f16": 0, "f32": 1}[d_type] << 4
        value |= type_codes[a_type] << 7
        value |= type_codes[b_type] << 10
        if saturate:
            raise DescriptorError("saturate is only defined for kind=i8")
    elif kind == "tf32":
        if a_type != "tf32" or b_type != "tf32":
            raise DescriptorError("kind=tf32 requires tf32 A/B types")
        d_type = "f32" if d_type is None else d_type
        if d_type != "f32":
            raise DescriptorError("kind=tf32 requires f32 D type")
        value |= 1 << 4
        value |= 2 << 7
        value |= 2 << 10
        if saturate:
            raise DescriptorError("saturate is only defined for kind=i8")
    elif kind == "f8f6f4":
        if a_type not in F8F6F4_TYPES or b_type not in F8F6F4_TYPES:
            raise DescriptorError(f"unsupported f8f6f4 A/B types: {a_type}, {b_type}")
        d_type = "f32" if d_type is None else d_type
        if d_type not in {"f16", "f32"}:
            raise DescriptorError("kind=f8f6f4 D type must be f16 or f32")
        value |= {"f16": 0, "f32": 1}[d_type] << 4
        value |= F8F6F4_TYPES[a_type] << 7
        value |= F8F6F4_TYPES[b_type] << 10
        if saturate:
            raise DescriptorError("saturate is only defined for kind=i8")
    elif kind == "i8":
        if n not in {8, 16, 24, 32} and not (
            48 <= n <= 256 and n % 16 == 0
        ):
            raise DescriptorError(
                f"kind=i8 N={n} violates the nonuniform Table-39 N rule"
            )
        type_codes = {"u8": 0, "s8": 1}
        if a_type not in type_codes or b_type not in type_codes:
            raise DescriptorError("kind=i8 accepts only s8 or u8 A/B types")
        d_type = "s32" if d_type is None else d_type
        if d_type != "s32":
            raise DescriptorError("kind=i8 requires s32 D type")
        value |= int(saturate) << 3
        value |= 2 << 4
        value |= type_codes[a_type] << 7
        value |= type_codes[b_type] << 10
    else:
        raise DescriptorError(f"unsupported unscaled kind: {kind}")

    value |= (n >> 3) << 17
    value |= (m >> 4) << 24
    return DescriptorRecord(
        kind=kind,
        m=m,
        n=n,
        a_type=a_type,
        b_type=b_type,
        d_type=d_type,
        value_u32=value,
        saturate=saturate,
    )


def encode_block_scaled_fp4(
    kind: str,
    *,
    m: int,
    n: int,
    scale_type: str,
    a_scale_data_id: int = 0,
    b_scale_data_id: int = 0,
    k: int = 64,
) -> DescriptorRecord:
    """Encode PTX ISA 9.0 Table 44 for MXFP4 or NVFP4.

    ``kind=mxf4`` uses UE8M0 scales.  ``kind=mxf4nvf4`` accepts UE8M0 or
    UE4M3; the project's NVFP4 contract uses UE4M3.  Table 44 encodes E2M1 as
    value 1, which is intentionally different from value 5 in Table 42.
    """

    _validate_mn(m, n, block_scaled=True)
    if kind not in {"mxf4", "mxf4nvf4"}:
        raise DescriptorError("block-scaled FP4 kind must be mxf4 or mxf4nvf4")
    if kind == "mxf4" and scale_type != "ue8m0":
        raise DescriptorError("kind=mxf4 requires ue8m0 scales")
    if kind == "mxf4nvf4" and scale_type not in {"ue8m0", "ue4m3"}:
        raise DescriptorError("kind=mxf4nvf4 scale type must be ue8m0 or ue4m3")
    if a_scale_data_id not in {0, 2} or b_scale_data_id not in {0, 2}:
        raise DescriptorError("Table 44 scale data IDs must be 0 or 2")
    if k != 64:
        raise DescriptorError(
            "SM110 block-scaled FP4 supports dense K=64; K=96 is sm_103a-only"
        )

    value = 0
    value |= b_scale_data_id << 4
    value |= 1 << 7  # Table 44 E2M1 code, not Table 42 code 5.
    value |= 1 << 10
    value |= (n >> 3) << 17
    value |= int(scale_type == "ue8m0") << 23
    value |= (m >> 7) << 27
    value |= a_scale_data_id << 29
    return DescriptorRecord(
        kind=kind,
        m=m,
        n=n,
        a_type="e2m1",
        b_type="e2m1",
        d_type="f32",
        value_u32=value,
        scale_type=scale_type,
        a_scale_data_id=a_scale_data_id,
        b_scale_data_id=b_scale_data_id,
    )


def decode_fields(value: int, *, kind: str) -> dict[str, int]:
    """Return raw fields for tests, audits, and provenance manifests."""

    if not 0 <= value <= 0xFFFFFFFF:
        raise DescriptorError("instruction descriptor must fit in 32 bits")
    if kind in {"mxf4", "mxf4nvf4"}:
        return {
            "b_scale_data_id": (value >> 4) & 0x3,
            "atype": (value >> 7) & 0x7,
            "btype": (value >> 10) & 0x3,
            "n": ((value >> 17) & 0x3F) << 3,
            "scale_type": (value >> 23) & 0x1,
            "m": ((value >> 27) & 0x3) << 7,
            "a_scale_data_id": (value >> 29) & 0x3,
            "k_is_96": (value >> 31) & 0x1,
        }
    return {
        "saturate": (value >> 3) & 0x1,
        "dtype": (value >> 4) & 0x3,
        "atype": (value >> 7) & 0x7,
        "btype": (value >> 10) & 0x7,
        "n": ((value >> 17) & 0x3F) << 3,
        "m": ((value >> 24) & 0x1F) << 4,
    }
