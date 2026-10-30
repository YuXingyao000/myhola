"""Two-policy dispatch, identity/provenance isolation and report labels; CPU only."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from src.eval import preference_compare as compare
from src.eval import preference_report as report
from src.eval import diffusion_audit_report as audit_report
from tests import test_preference_compare_reuse as reuse_fixture
from tests import test_preference_report as report_fixture
from tests.test_diffusion_audit import audit_row
from tools.vae_compare_post import atomic_json, fingerprint, read_json, sha256_file


def test_control_slot_loads_its_own_policy_and_both_require_fixed_step():
    config = {'original_checkpoint': 'original', 'preference_checkpoint': 'geometry',
              'baseline_preference_checkpoint': 'control'}
    with patch('src.models.diffusion.legacy.load_preference_svr') as loader, \
            patch('src.models.diffusion.legacy.load_legacy_svr') as original:
        loader.return_value = ('model', {'preference': {'global_step': 5000}})
        compare.load_comparison_model(config, 'baseline', device='cpu')
        loader.assert_called_once_with('original', 'control', device='cpu')
        compare.load_comparison_model(config, 'dpo_5000', device='cpu')
        loader.assert_called_with('original', 'geometry', device='cpu')
        original.assert_not_called()
        loader.return_value = ('model', {'preference': {'global_step': 4000}})
        with pytest.raises(ValueError, match='5,000-step'):
            compare.load_comparison_model(config, 'baseline')
        config.pop('baseline_preference_checkpoint')
        compare.load_comparison_model(config, 'baseline', device='cpu')
        original.assert_called_once_with('original', device='cpu')


def test_pair_cannot_reuse_original_baseline_or_a_different_control():
    fixture = reuse_fixture.PreferenceCompareReuseTests()
    fixture.setUp()
    try:
        config = {**fixture.protocol['config'],
                  'baseline_preference_checkpoint': str(fixture.root / 'old.ckpt')}
        pair = compare.build_protocol(config, fixture.identities)
        assert pair['version'] == 'svr_dpo_geometry_pair_validation_v1'
        assert pair['method_labels']['baseline'] == 'DPO control'
        assert 'src/models/diffusion/preference_geometry.py' in pair['source_sha256']
        with pytest.raises(ValueError, match='version'):
            compare.validate_reuse_protocol(pair)
        source = deepcopy(pair)
        source['config'] = {**fixture.old_protocol['config'],
                            'baseline_preference_checkpoint': config['baseline_preference_checkpoint']}
        source['fingerprint'] = fingerprint({k:v for k,v in source.items() if k != 'fingerprint'})
        atomic_json(fixture.source / 'metadata/protocol.json', source)
        compare.validate_reuse_protocol(pair)
        pair['checkpoint_sha256']['baseline_preference_checkpoint'] = 'different-control'
        with pytest.raises(ValueError, match='baseline_preference_checkpoint'):
            compare.validate_reuse_protocol(pair)
    finally:
        fixture.doCleanups()


def test_geometry_report_labels_follow_protocol_without_changing_metrics():
    fixture = report_fixture.PreferenceReportTests()
    fixture.setUp()
    try:
        protocol = {'identities': fixture.ids, 'version': 'svr_dpo_geometry_pair_validation_v1',
                    'method_labels': {'baseline': 'DPO control', 'dpo_5000': 'DPO + geometry'},
                    'method_labels_zh': {'baseline': '继续 DPO', 'dpo_5000': 'DPO + 几何项'},
                    'comparison': 'geometry minus control', 'checkpoint_sha256': {'control': 'a', 'geometry': 'b'}}
        path = fixture.root / 'metadata/protocol.json'
        atomic_json(path, protocol)
        ids, rows = fixture.load()
        expected = report.summarize(ids, rows)
        with patch.object(report, 'plot_comparison'):
            actual = report.write_report(fixture.root, expected_count=4)
        assert actual['method_labels'] == protocol['method_labels']
        assert actual['protocol']['comparison'] == 'geometry minus control'
        assert actual['metrics'] == expected['metrics']
        audit = fixture.root / 'audit'
        atomic_json(audit / 'protocol.json', {'config': {'comparison_run': str(fixture.root)},
                                             'comparison_protocol_sha256': sha256_file(path)})
        with patch.object(audit_report, 'plot_figures'):
            actual = audit_report.write_report([audit_row('baseline', 'one'), audit_row('dpo_5000', 'one')], audit)
        assert actual['method_labels'] == protocol['method_labels_zh']
        assert '继续 DPO' in (audit / 'report/README.md').read_text()
        protocol['comparison'] = 'tampered'
        atomic_json(path, protocol)
        with pytest.raises(ValueError, match='protocol changed'):
            audit_report.write_report([audit_row('baseline', 'one')], audit)
    finally:
        fixture.doCleanups()
