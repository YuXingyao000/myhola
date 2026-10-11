import pytest

from tools.legacy_failure_census_manifest import partition_population


def test_uses_native_validity_even_when_sampling_failed():
    raw = {('legacy', '00000001'): {}, ('legacy', '00000002'): {}}
    post = {('legacy', '00000001'): {'native_valid_solid': True, 'status': 'geometry_error'},
            ('legacy', '00000002'): {'native_valid_solid': False, 'status': 'ok'}}
    fit = {('legacy', '00000001'): {'native_valid_solid': False},
           ('legacy', '00000002'): {'native_valid_solid': True}}
    ids, failed, counts = partition_population(raw, post, fit)
    assert ids == ['00000001', '00000002']
    assert failed == ['00000002']
    assert counts == {'fit_failed_optimized_valid': 1, 'fit_valid_optimized_failed': 1}


def test_unknown_validity_is_not_silently_a_failure():
    raw = {('legacy', '00000001'): {}}
    with pytest.raises(ValueError, match='Unknown native validity'):
        partition_population(raw, {('legacy', '00000001'): {'native_valid_solid': None}},
                             {('legacy', '00000001'): {'native_valid_solid': True}})


def test_missing_identity_is_not_dropped():
    with pytest.raises(ValueError, match='identical'):
        partition_population({('legacy', '00000001'): {}}, {}, {})
