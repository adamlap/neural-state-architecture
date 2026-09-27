from nsa.residency.packed_kernels import (
    int4_linear_reference,
    unpack_signed,
    unpack_ternary,
    unpack_unsigned,
)


def test_unpack_unsigned_int4():
    assert unpack_unsigned(bytes([0x21, 0x43]), 4, 4) == [1, 2, 3, 4]


def test_unpack_signed_int4():
    assert unpack_signed(bytes([0xF8]), 4, 2) == [-8, -1]


def test_unpack_ternary_matches_five_trits_per_byte():
    assert unpack_ternary(bytes([121]), 5) == [0, -1, 0, 1, 0]


def test_int4_reference_linear():
    assert int4_linear_reference([1, 2, 3], [2.0, 1.0, 4.0], 0.5) == 8.0
