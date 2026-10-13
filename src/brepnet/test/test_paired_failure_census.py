import copy
import json
from pathlib import Path

import pytest

from tools.failure_census_contract import METHODS as HISTORICAL_METHODS, case_method, case_record
from tools.paired_failure_manifest import METHODS, partition_population, read_table
from tools.paired_failure_report import (
    COHORTS, PAGE_HTML, PAGE_JS, THRESHOLDS, audit_source_preservation,
    diagnostic_audit, optimization_counterfactual, paired_transitions, validate_cohorts,
)


def tables():
    ids = [f'{i:08}' for i in range(1, 5)]
    raw, post, fit = {}, {}, {}
    for i, sid in enumerate(ids):
        for method in METHODS:
            key = method, sid
            raw[key] = {'model_id': sid, 'method': method, 'status': 'ok', 'mode': 'posterior_mean',
                        'shared_input_status': 'verified', 'input_latent_sha256': 'a'*64}
            valid = (i in (1, 3)) if method == 'intersection' else (i in (2, 3))
            post[key] = {'sample_id': sid, 'method': method, 'status': 'ok', 'mode': 'optimized',
                         'native_valid_solid': valid, 'representation': 'legacy_half_edges'}
            fit[key] = {**post[key], 'mode': 'fit_only', 'native_valid_solid': i == 1}
    return ids, raw, post, fit


def test_method_extension_preserves_historical_pair_and_disallows_aliases():
    assert HISTORICAL_METHODS == ('legacy', 'intersection')
    assert case_method({'method': 'local'}) == 'local'
    for method in ['refinement', 'intersection_local', None]:
        with pytest.raises(ValueError):
            case_method({'method': method})
    with pytest.raises(ValueError):
        case_record({'id': '1', 'method': 'local', 'raw_legacy': {}}, 'raw')


def test_dynamic_partition_retains_all_four_cohorts_and_optimization_regressions():
    ids, raw, post, fit = tables()
    result = partition_population(raw, post, fit, ids)
    assert result['cohorts'] == {'both_failed': [ids[0]], 'new_failure': [ids[1]], 'rescued': [ids[2]], 'both_valid': [ids[3]]}
    assert result['optimization_regression_ids'] == [ids[1]]
    assert sum(result['fit_only_vs_optimized_all_accepted']['local'].values()) == 4


@pytest.mark.parametrize('defect', ['missing', 'unknown', 'hash', 'extra', 'nonterminal'])
def test_partition_rejects_mixed_or_incomplete_inputs(defect):
    ids, raw, post, fit = tables()
    key = 'local', ids[0]
    if defect == 'missing': raw.pop(key)
    elif defect == 'unknown': post[key]['native_valid_solid'] = 0
    elif defect == 'hash': raw[key]['input_latent_sha256'] = 'b'*64
    elif defect == 'extra': raw['local', '00000005'] = raw[key]
    elif defect == 'nonterminal': post[key]['status'] = 'running'
    with pytest.raises(ValueError):
        partition_population(raw, post, fit, ids)


def test_csv_preserves_ids_and_hashes_and_rejects_old_legacy(tmp_path):
    path = tmp_path / 'rows.csv'
    path.write_text('model_id,method,status,input_latent_sha256\n00000001,local,ok,'+'1'*64+'\n')
    assert read_table(path, 'model_id')['local', '00000001']['input_latent_sha256'] == '1'*64
    path.write_text(path.read_text().replace(',local,', ',legacy,'))
    with pytest.raises(ValueError, match='method'):
        read_table(path, 'model_id')


def case(sid, method, valid):
    return {'id': sid, 'method': method, 'post_record': {'method': method, 'sample_id': sid, 'native_valid_solid': valid}}


def test_dynamic_cohorts_validate_actual_identity_and_empty_groups():
    new = [case('1', 'local', False), case('2', 'local', True)]
    old = [case('1', 'intersection', False), case('2', 'intersection', False)]
    frozen = {'both_failed': ['1'], 'new_failure': [], 'rescued': ['2']}
    assert validate_cohorts(new, old, frozen) == {'persistent': ['1'], 'regressed': [], 'rescued': ['2']}
    with pytest.raises(ValueError):
        validate_cohorts(new, old, {**frozen, 'rescued': ['3']})
    old[1]['post_record']['native_valid_solid'] = True
    with pytest.raises(ValueError, match='Both-success'):
        validate_cohorts(new, old, frozen)


def test_paired_unknown_and_empty_counts():
    def pair(sid, a, b):
        return {'id': sid, 'baseline': {'flags': {'0.006': {'x': a}}}, 'candidate': {'flags': {'0.006': {'x': b}}}}
    result = paired_transitions([pair('1', True, None), pair('2', None, False), pair('3', True, False)], 'x', '0.006')
    assert result['counts']['unknown'] == 2
    assert result['counts']['present_to_absent'] == 1
    assert paired_transitions([], 'x', '0.006')['denominator'] == 0


def test_empty_control_and_counterfactual_do_not_require_diagnostic_files(tmp_path):
    assert diagnostic_audit({'cases': []}, tmp_path, tmp_path/'absent.json')['checked_count'] == 0
    result = optimization_counterfactual({'optimization_regression_ids': []}, tmp_path, [])
    assert result['denominator'] == 0 and result['cases'] == []


def test_source_preservation_detects_modified_and_newly_missing_assets(tmp_path):
    import hashlib
    source = tmp_path / 'source.bin'
    source.write_bytes(b'input')
    (tmp_path / 'source_asset_audit.json').write_text(json.dumps({'source_assets': [
        {'path': str(source), 'exists': True, 'sha256': hashlib.sha256(b'input').hexdigest()}]}))
    assert audit_source_preservation(tmp_path)['mismatch_count'] == 0
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        audit_source_preservation(tmp_path)
    source.unlink()
    with pytest.raises(ValueError, match='changed'):
        audit_source_preservation(tmp_path)


def test_page_is_dynamic_and_baseline_not_labelled_original_hola():
    for historic_count in ('304', '262', '369', '2048', '2113'):
        assert historic_count not in PAGE_HTML + PAGE_JS
    assert 'method_labels[method]' in PAGE_JS
    assert 'PAIRED_CENSUS_DATA' in PAGE_JS
    assert 'den?100*values[s].length/den:0' in PAGE_JS
    assert 'population.baseline_native_valid' in PAGE_JS
    assert '原版 HoLa' not in PAGE_JS
