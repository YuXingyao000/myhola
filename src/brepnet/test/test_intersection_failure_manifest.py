import csv
import hashlib
import json

import pytest

from tools.intersection_failure_manifest import (
    AssetAudit, measurement_settings_delta, partition_population, read_table, step_data_hash, typed_value, verify_reuse,
)
from tools.vae_compare_post import fingerprint


def tables():
    ids = [f"{i:08}" for i in range(1,5)]
    raw, post, fit = {}, {}, {}
    for i,sid in enumerate(ids):
        for method in ("legacy","intersection"):
            raw[method,sid] = {"status":"ok","mode":"posterior_mean","shared_input_status":"verified","input_latent_sha256":"a"*64}
            valid = (False,True,False,True)[i] if method == "legacy" else (False,False,True,True)[i]
            post[method,sid] = {"status":"timeout" if i==2 else "ok","mode":"optimized","representation":"legacy_half_edges","native_valid_solid":valid}
            fit[method,sid] = {"status":"ok","mode":"fit_only","representation":"legacy_half_edges","native_valid_solid":True}
    return ids, raw, post, fit


def test_groups_use_known_native_boolean_including_geometry_timeout():
    ids,raw,post,fit = tables()
    result = partition_population(raw,post,fit,ids)
    assert result['cohorts'] == {'both_failed':[ids[0]],'new_failure':[ids[1]],'rescued':[ids[2]],'both_valid':[ids[3]]}
    assert result['optimization_regression_ids'] == ids[:2]


@pytest.mark.parametrize('invalid',[None,0,'False'])
def test_unknown_or_non_boolean_native_validity_rejected(invalid):
    ids,raw,post,fit = tables()
    post['intersection',ids[0]]['native_valid_solid'] = invalid
    with pytest.raises(ValueError,match='Unknown native validity'):
        partition_population(raw,post,fit,ids)


def test_missing_method_id_nonterminal_and_different_latent_are_rejected():
    ids,raw,post,fit = tables()
    del raw['intersection',ids[0]]
    with pytest.raises(ValueError,match='identical intended IDs'):
        partition_population(raw,post,fit,ids)
    ids,raw,post,fit = tables()
    post['intersection',ids[0]]['status'] = 'running'
    with pytest.raises(ValueError,match='Nonterminal'):
        partition_population(raw,post,fit,ids)
    post['intersection',ids[0]]['status'] = 'ok'
    raw['intersection',ids[0]]['input_latent_sha256'] = 'b'*64
    with pytest.raises(ValueError,match='latent hashes'):
        partition_population(raw,post,fit,ids)


def test_csv_accepts_intersection_and_preserves_hashes_but_rejects_duplicates(tmp_path):
    path = tmp_path/'table.csv'
    path.write_text('sample_id,method,native_valid_solid,source_record_sha256\n00000001,intersection,False,'+'1'*64+'\n')
    row=read_table(path,'sample_id')['intersection','00000001']
    assert row['native_valid_solid'] is False and row['source_record_sha256']=='1'*64
    path.write_text(path.read_text()+path.read_text().splitlines()[1]+'\n')
    with pytest.raises(ValueError,match='Duplicate'):
        read_table(path,'sample_id')
    with pytest.raises(ValueError,match='Nonfinite'):
        typed_value('face_cd','nan')


def test_step_identity_ignores_header_but_preserves_data_bytes(tmp_path):
    path=tmp_path/'model.step'
    data=b'\r\n#1=EXAMPLE();\r\n'
    path.write_bytes(b'HEADER;old;ENDSEC;DATA;'+data+b'ENDSEC;')
    digest=step_data_hash(path)
    assert digest==hashlib.sha256(data).hexdigest()
    path.write_bytes(b'HEADER;new;ENDSEC;DATA;'+data+b'ENDSEC;')
    assert step_data_hash(path)==digest
    path.write_bytes(b'DATA;'+data.replace(b'EXAMPLE',b'OTHER')+b'ENDSEC;')
    assert step_data_hash(path)!=digest


def test_asset_audit_records_missing_and_rejects_changed_source(tmp_path):
    audit=AssetAudit();path=tmp_path/'raw.npz'
    assert audit.add(path)['exists'] is False
    path=tmp_path/'present.npz';path.write_bytes(b'content')
    with pytest.raises(ValueError,match='SHA256 changed'):
        audit.add(path,'a'*64)


def reusable_case(tmp_path):
    sid='00000001';paths={}
    for name in ['raw_npz','post_step','fit_step','gt_npz','gt_step']:
        p=tmp_path/name;p.write_bytes(b'DATA;#1=EXAMPLE();ENDSEC;' if 'step' in name else name.encode());paths[name]=str(p)
    assets={key:hashlib.sha256(open(path,'rb').read()).hexdigest() for key,path in paths.items()}
    case={'id':sid,'method':'legacy','cohort':'rescued','paths':paths,'asset_sha256':assets,'path_exists':{key:True for key in paths},
          'raw_record':{'input_latent_sha256':'a'*64},'post_record':{'raw_sha256':assets['raw_npz'],'protocol_fingerprint':'old','native_valid_solid':False},
          'fit_record':{'raw_sha256':assets['raw_npz'],'protocol_fingerprint':'fit','native_valid_solid':False}}
    protocols={name:{'old_protocol':name} for name in ['trace','upstream','saved_step']}
    trace={'sample_id':sid,'status':'ok','protocol_fingerprint':fingerprint(protocols['trace']),'instrumentation_errors':[],
           'comparison':{'same_recorded_outcome':True},'step_data_comparison':{'equal':True,'original':step_data_hash(tmp_path/'post_step'),'replay':step_data_hash(tmp_path/'post_step')},'raw_sha256':assets['raw_npz']}
    tp=tmp_path/'traces'/f'{sid}.json';tp.parent.mkdir();tp.write_text(json.dumps(trace))
    geometry=tmp_path/'replay'/sid/'trace_geometry.npz';geometry.parent.mkdir(parents=True);geometry.write_bytes(b'geometry')
    upstream={'sample_id':sid,'status':'ok','protocol_fingerprint':fingerprint(protocols['upstream']),
              'sources':{'raw':{'sha256':assets['raw_npz']},'gt':{'sha256':assets['gt_npz']},
                         'trace':{'sha256':hashlib.sha256(tp.read_bytes()).hexdigest()},
                         'stage_geometry':{'sha256':hashlib.sha256(geometry.read_bytes()).hexdigest()}}}
    # The real saved-step schema uses id, unlike trace/upstream sample_id.
    saved={'id':sid,'status':'ok','protocol_fingerprint':fingerprint(protocols['saved_step']),'step_sha256':assets['post_step']}
    for name,record in [('upstream',upstream),('saved_step',saved)]:
        p=tmp_path/name/f'{sid}.json';p.parent.mkdir();p.write_text(json.dumps(record))
    return case,protocols


def test_reuse_preserves_historical_fingerprints_and_checks_step_data(tmp_path):
    case,protocols=reusable_case(tmp_path)
    result=verify_reuse(case,case,tmp_path,protocols,AssetAudit())
    assert result['eligible'] is True
    assert result['original_protocol_fingerprints']=={k:fingerprint(v) for k,v in protocols.items()}
    (tmp_path/'post_step').write_text('DATA;#1=CHANGED();ENDSEC;')
    result=verify_reuse(case,case,tmp_path,protocols,AssetAudit())
    assert result['eligible'] is False and 'original STEP DATA differs from frozen trace' in result['reasons']


def test_missing_diagnostic_remains_ineligible_not_negative_evidence(tmp_path):
    case,protocols=reusable_case(tmp_path)
    (tmp_path/'saved_step/00000001.json').unlink()
    result=verify_reuse(case,case,tmp_path,protocols,AssetAudit())
    assert result['eligible'] is False and 'missing saved_step' in result['reasons']


def test_saved_step_requires_its_own_id_field_not_trace_identity_alias(tmp_path):
    case,protocols=reusable_case(tmp_path)
    path=tmp_path/'saved_step/00000001.json'
    data=json.loads(path.read_text());data['sample_id']=data.pop('id');path.write_text(json.dumps(data))
    result=verify_reuse(case,case,tmp_path,protocols,AssetAudit())
    assert result['eligible'] is False and 'unverified saved_step identity/status' in result['reasons']


def test_only_dedup_description_change_is_allowed():
    old={'thresholds':[.002,.01],'dedup':'not_applied: old description'}
    new={'thresholds':(.002,.01),'dedup':'not_applied: fixed identity description'}
    assert set(measurement_settings_delta(old,new))=={'dedup'}
    for changed in [dict(new,thresholds=(.002,.02)),dict(new,dedup='applied'),dict(new,new_metric=True)]:
        with pytest.raises(ValueError,match='settings differ'):
            measurement_settings_delta(old,changed)
