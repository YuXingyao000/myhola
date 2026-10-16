import copy
import json
from pathlib import Path

import pytest

from tools.failure_census_contract import METHODS as HISTORICAL_METHODS, case_method, case_record
from tools.paired_failure_manifest import METHODS, audit_reuse, paired_methods, partition_population, read_table
from tools.paired_failure_report import (
    COHORTS, PAGE_HTML, PAGE_JS, THRESHOLDS, audit_source_preservation,
    diagnostic_audit, optimization_counterfactual, page_payload, paired_transitions, validate_cohorts,
)


def tables(candidate_method="local", baseline_method="intersection"):
    ids = [f'{i:08}' for i in range(1, 5)]
    raw, post, fit = {}, {}, {}
    for i, sid in enumerate(ids):
        for method in paired_methods(candidate_method, baseline_method):
            key = method, sid
            raw[key] = {'model_id': sid, 'method': method, 'status': 'ok', 'mode': 'posterior_mean',
                        'shared_input_status': 'verified', 'input_latent_sha256': 'a'*64}
            valid = (i in (1, 3)) if method == baseline_method else (i in (2, 3))
            post[key] = {'sample_id': sid, 'method': method, 'status': 'ok', 'mode': 'optimized',
                         'native_valid_solid': valid, 'representation': 'legacy_half_edges'}
            fit[key] = {**post[key], 'mode': 'fit_only', 'native_valid_solid': i == 1}
    return ids, raw, post, fit


def test_method_extension_preserves_historical_pair_and_disallows_aliases():
    assert HISTORICAL_METHODS == ('legacy', 'intersection')
    assert case_method({'method': 'local'}) == 'local'
    assert case_method({'method': 'conditioned'}) == 'conditioned'
    assert case_method({'method': 'point_feedback'}) == 'point_feedback'
    assert case_method({'method': 'encoder_edge_value'}) == 'encoder_edge_value'
    assert case_method({'method': 'encoder_gat_control'}) == 'encoder_gat_control'
    for method in ['refinement', 'intersection_local', None]:
        with pytest.raises(ValueError):
            case_method({'method': method})
    with pytest.raises(ValueError):
        case_record({'id': '1', 'method': 'local', 'raw_legacy': {}}, 'raw')


@pytest.mark.parametrize('candidate_method', ['local', 'conditioned'])
def test_dynamic_partition_retains_all_four_cohorts_and_optimization_regressions(candidate_method):
    ids, raw, post, fit = tables(candidate_method)
    result = partition_population(raw, post, fit, ids, candidate_method)
    assert result['cohorts'] == {'both_failed': [ids[0]], 'new_failure': [ids[1]], 'rescued': [ids[2]], 'both_valid': [ids[3]]}
    assert result['optimization_regression_ids'] == [ids[1]]
    assert sum(result['fit_only_vs_optimized_all_accepted'][candidate_method].values()) == 4


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


@pytest.mark.parametrize('candidate_method', ['local', 'conditioned'])
def test_csv_preserves_ids_and_hashes_and_rejects_old_legacy(tmp_path, candidate_method):
    path = tmp_path / 'rows.csv'
    path.write_text(f'model_id,method,status,input_latent_sha256\n00000001,{candidate_method},ok,'+'1'*64+'\n')
    assert read_table(path, 'model_id', candidate_method)[candidate_method, '00000001']['input_latent_sha256'] == '1'*64
    path.write_text(path.read_text().replace(f',{candidate_method},', ',legacy,'))
    with pytest.raises(ValueError, match='method'):
        read_table(path, 'model_id', candidate_method)


def test_conditioned_is_explicit_and_cannot_relabel_local_rows():
    ids, raw, post, fit = tables('conditioned')
    with pytest.raises(ValueError, match='intended IDs'):
        partition_population(raw, post, fit, ids)
    with pytest.raises(ValueError, match='Unsupported paired'):
        paired_methods('intersection')
    with pytest.raises(ValueError, match='Missing/invalid'):
        case_record({'id': '1', 'method': 'conditioned', 'raw_legacy': {}}, 'raw')


def case(sid, method, valid):
    return {'id': sid, 'method': method, 'post_record': {'method': method, 'sample_id': sid, 'native_valid_solid': valid}}


@pytest.mark.parametrize('candidate_method', ['local', 'conditioned'])
def test_dynamic_cohorts_validate_actual_identity_and_empty_groups(candidate_method):
    new = [case('1', candidate_method, False), case('2', candidate_method, True)]
    old = [case('1', 'intersection', False), case('2', 'intersection', False)]
    frozen = {'both_failed': ['1'], 'new_failure': [], 'rescued': ['2']}
    assert validate_cohorts(new, old, frozen, candidate_method) == {'persistent': ['1'], 'regressed': [], 'rescued': ['2']}
    with pytest.raises(ValueError):
        validate_cohorts(new, old, {**frozen, 'rescued': ['3']}, candidate_method)
    old[1]['post_record']['native_valid_solid'] = True
    with pytest.raises(ValueError, match='Both-success'):
        validate_cohorts(new, old, frozen, candidate_method)


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


def test_conditioned_page_retains_explicit_methods_and_labels_for_empty_scope():
    summary = {'population': {}, 'main': {'denominator': 0},
               'method_labels': {'baseline': '上一版 Intersection refinement', 'candidate': '曲线条件调制'},
               'methods': {'baseline': 'intersection', 'candidate': 'conditioned'},
               'highlights': [], 'cohorts': {key: {'count': 0} for key in COHORTS}, 'limitations': []}
    payload = page_payload([], summary, [], set())
    assert payload['methods']['candidate'] == 'conditioned'
    assert payload['method_labels']['candidate'] == '曲线条件调制'
    assert payload['pairs'] == []


def test_point_feedback_uses_explicit_conditioned_baseline_without_relabelling(tmp_path):
    ids, raw, post, fit = tables('point_feedback', 'conditioned')
    result = partition_population(raw, post, fit, ids, 'point_feedback', 'conditioned')
    assert result['cohorts'] == {'both_failed': [ids[0]], 'new_failure': [ids[1]],
                                 'rescued': [ids[2]], 'both_valid': [ids[3]]}
    assert result['optimization_regression_ids'] == [ids[1]]
    assert set(result['fit_only_vs_optimized_all_accepted']) == {'conditioned', 'point_feedback'}
    with pytest.raises(ValueError, match='intended IDs'):
        partition_population(raw, post, fit, ids, 'point_feedback')
    for baseline in ['point_feedback', 'legacy', None]:
        with pytest.raises(ValueError, match='Unsupported paired baseline'):
            paired_methods('point_feedback', baseline)
    path = tmp_path / 'rows.csv'
    path.write_text('model_id,method,status\n00000001,conditioned,ok\n00000001,point_feedback,ok\n')
    assert set(read_table(path, 'model_id', 'point_feedback', 'conditioned')) == {
        ('conditioned', '00000001'), ('point_feedback', '00000001')}
    path.write_text(path.read_text().replace(',conditioned,', ',intersection,'))
    with pytest.raises(ValueError, match='identity/method'):
        read_table(path, 'model_id', 'point_feedback', 'conditioned')


def test_point_feedback_cohorts_reject_wrong_baseline_identity():
    new = [case('1', 'point_feedback', False), case('2', 'point_feedback', True)]
    old = [case('1', 'conditioned', False), case('2', 'conditioned', False)]
    frozen = {'both_failed': ['1'], 'new_failure': [], 'rescued': ['2']}
    assert validate_cohorts(new, old, frozen, 'point_feedback', 'conditioned') == {
        'persistent': ['1'], 'regressed': [], 'rescued': ['2']}
    with pytest.raises(ValueError, match='wrong method'):
        validate_cohorts(new, old, frozen, 'point_feedback')
    with pytest.raises(ValueError, match='exact historical conditioned'):
        audit_reuse(old[0], case('1', 'intersection', False), None, None, None, 'conditioned')


def test_point_feedback_page_keeps_distinct_method_labels():
    summary = {'population': {}, 'main': {'denominator': 0},
               'method_labels': {'baseline': '曲线条件调制', 'candidate': '逐点曲线反馈'},
               'methods': {'baseline': 'conditioned', 'candidate': 'point_feedback'},
               'highlights': [], 'cohorts': {key: {'count': 0} for key in COHORTS}, 'limitations': []}
    payload = page_payload([], summary, [], set())
    assert payload['methods'] == {'baseline': 'conditioned', 'candidate': 'point_feedback'}
    assert payload['method_labels'] == {'baseline': '曲线条件调制', 'candidate': '逐点曲线反馈'}


@pytest.mark.parametrize('candidate_method', ['encoder_edge_value', 'encoder_gat_control'])
def test_current_encoders_require_explicit_same_geometry_contract(candidate_method):
    ids, raw, post, fit = tables(candidate_method, 'conditioned')
    for (method, sid), row in raw.items():
        row.update(input_contract='same_geometry_source', input_source_sha256='c'*64,
                   actual_gt_source_sha256='c'*64, latent_digest_status='verified',
                   input_latent_sha256=('a' if method == 'conditioned' else 'b')*64)
    result = partition_population(raw, post, fit, ids, candidate_method, 'conditioned', 'same_geometry_source')
    assert result['cohorts'] == {'both_failed': [ids[0]], 'new_failure': [ids[1]],
                                 'rescued': [ids[2]], 'both_valid': [ids[3]]}
    # This is not a new meaning silently assigned to the historical default.
    with pytest.raises(ValueError, match='input contract'):
        partition_population(raw, post, fit, ids, candidate_method, 'conditioned')


@pytest.mark.parametrize('defect', ['source', 'actual_gt', 'missing', 'latent', 'own_latent', 'contract', 'unknown_contract'])
def test_same_geometry_contract_rejects_misattribution(defect):
    candidate = 'encoder_edge_value'
    ids, raw, post, fit = tables(candidate, 'conditioned')
    for (method, sid), row in raw.items():
        row.update(input_contract='same_geometry_source', input_source_sha256='c'*64,
                   actual_gt_source_sha256='c'*64, latent_digest_status='verified',
                   input_latent_sha256=('a' if method == 'conditioned' else 'b')*64)
    row = raw[candidate, ids[0]]
    if defect == 'source': row['input_source_sha256'] = 'd'*64
    elif defect == 'actual_gt': row['actual_gt_source_sha256'] = 'd'*64
    elif defect == 'missing': row.pop('input_source_sha256')
    elif defect == 'latent': row['input_latent_sha256'] = 'not-a-digest'
    elif defect == 'own_latent': row['latent_digest_status'] = 'mismatch'
    elif defect == 'contract': row.pop('input_contract')
    contract = 'unknown' if defect == 'unknown_contract' else 'same_geometry_source'
    with pytest.raises(ValueError):
        partition_population(raw, post, fit, ids, candidate, 'conditioned', contract)


def test_current_encoder_page_carries_input_contract():
    summary = {'population': {}, 'main': {'denominator': 0},
               'method_labels': {'baseline': '曲线条件调制', 'candidate': 'Encoder 边内容消息'},
               'methods': {'baseline': 'conditioned', 'candidate': 'encoder_edge_value'},
               'input_contract': 'same_geometry_source',
               'highlights': [], 'cohorts': {key: {'count': 0} for key in COHORTS}, 'limitations': []}
    payload = page_payload([], summary, [], set())
    assert payload['input_contract'] == 'same_geometry_source'
    assert payload['methods']['candidate'] == 'encoder_edge_value'
