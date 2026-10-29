"""Independent paired summary of frozen raw and native STEP records."""
from __future__ import annotations
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import numpy as np

CODE = Path('/mnt/d/python_uv_train/checkpoints/conditional_intersection/feature_on_evaluation_20261029')
HERE = Path('/root/hola_eval/conditional_feature_on_20261029')
ROOT = HERE / 'repository'
REFERENCE = Path('/root/hola_eval/conditional_feature_on_20261029/baseline')
METHODS = ('shared_curve_uv', 'conditional_best415', 'conditional_last5395')


def stats(values):
    valid = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return dict(count=len(valid), mean=float(np.mean(valid)) if valid else None,
        median=float(np.median(valid)) if valid else None, p90=float(np.quantile(valid,.9)) if valid else None)


def load_raw():
    rows = list(csv.DictReader((REFERENCE/'raw_metrics/posterior_mean/per_object.csv').open()))
    base = {r['model_id']:r for r in rows if r['method']=='shared_curve_uv'}
    for row in base.values():
        for key in ('face_cd_sq','edge_cd_sq','face_mae','topology_tp','topology_fp','topology_fn'):
            row[key] = float(row[key]) if row[key] else None
    result = {'shared_curve_uv':base}
    for method in METHODS[1:]:
        data = [json.loads(line) for line in (HERE/'metadata/export'/method/'records.jsonl').read_text().splitlines()]
        assert len(data) == len({r['id'] for r in data})
        result[method] = {r['id']:r for r in data}
    return result


def load_post():
    result = {}
    for method in METHODS:
        root = REFERENCE if method == METHODS[0] else HERE/method
        data = [json.loads(p.read_text()) for p in (root/'metadata/post/optimized/samples'/method).glob('*.json')]
        result[method] = {r['sample_id']:r for r in data}
    return result


def main():
    ids = (ROOT/'checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt').read_text().split()
    raw, post = load_raw(), load_post()
    assert len(ids)==len(set(ids))==3495
    for method in METHODS:
        assert set(raw[method]) == set(post[method]) == set(ids), method
    result = dict(intended=3495,original_split=3505,methods={},pairs={})
    compact = []
    for method in METHODS:
        rr, pp = raw[method], post[method]
        assert all(p['status'] in {'ok','geometry_error','timeout','post_error','no_step','step_read_error','crash','missing_raw','invalid_raw','interrupted'} for p in pp.values())
        valid = [sid for sid in ids if pp[sid].get('native_valid_solid')]
        tp,fp,fn = [sum(r[k] for r in rr.values()) for k in ('topology_tp','topology_fp','topology_fn')]
        result['methods'][method] = dict(native_valid=len(valid),valid_rate=len(valid)/3495,
            valid_rate_original=len(valid)/3505,terminal_records=len(pp),
            statuses=dict(Counter(p['status'] for p in pp.values())),
            raw={k:stats([r.get(k) for r in rr.values()]) for k in ('face_cd_sq','edge_cd_sq','face_mae')},
            topology=dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=tp/(tp+fp),recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),
                exact_objects=sum(r['topology_fp']==0 and r['topology_fn']==0 for r in rr.values())),
            raw_internal_status={key:dict(Counter(r.get(key,'MISSING') for r in rr.values())) for key in ('prediction_status','gt_status','face_status','edge_status','face_mae_status','internal_topology_status','topology_status')},
            geometry_computable=sum(p.get('geometry_status')=='ok' for p in pp.values()),
            valid_geometry_computable=sum(pp[sid].get('geometry_status')=='ok' for sid in valid),
            step_exists=sum(bool(p.get('step_exists')) for p in pp.values()),
            step_readable=sum(bool(p.get('step_readable')) for p in pp.values()),
            success_markers=sum(bool(p.get('success_marker')) for p in pp.values()))
        for sid in ids:
            compact.append(dict(id=sid,method=method,native_valid=bool(pp[sid].get('native_valid_solid')),
                status=pp[sid]['status'],geometry_status=pp[sid].get('geometry_status'),
                face_cd_sq=rr[sid].get('face_cd_sq'),edge_cd_sq=rr[sid].get('edge_cd_sq'),
                face_mae=rr[sid].get('face_mae'),topology_tp=rr[sid]['topology_tp'],
                topology_fp=rr[sid]['topology_fp'],topology_fn=rr[sid]['topology_fn'],
                step_face_cd_sq=pp[sid].get('face_cd'),step_edge_cd_sq=pp[sid].get('edge_cd')))
    assert result['methods']['shared_curve_uv']['native_valid']==3180
    for candidate in METHODS[1:]:
        old,new = post[METHODS[0]],post[candidate]
        transitions = Counter()
        common = []
        for sid in ids:
            a,b = bool(old[sid].get('native_valid_solid')),bool(new[sid].get('native_valid_solid'))
            transitions['both_valid' if a and b else 'rescued' if b else 'regressed' if a else 'both_failed'] += 1
            if a and b and old[sid].get('geometry_status')==new[sid].get('geometry_status')=='ok':
                common.append(sid)
        assert sum(transitions.values())==3495
        assert transitions['rescued']-transitions['regressed']==result['methods'][candidate]['native_valid']-3180
        paired=dict(transitions=dict(transitions),common_valid_computable=len(common),step_geometry={},raw={})
        for key in ('face_cd','edge_cd'):
            paired['step_geometry'][key] = {method:stats([post[method][sid].get(key) for sid in common]) for method in (METHODS[0],candidate)}
        for key in ('face_cd_sq','edge_cd_sq','face_mae'):
            pairs=[(raw[METHODS[0]][sid].get(key),raw[candidate][sid].get(key)) for sid in ids]
            pairs=[(a,b) for a,b in pairs if a is not None and b is not None]
            paired['raw'][key]=dict(count=len(pairs),improved=sum(b<a for a,b in pairs),regressed=sum(b>a for a,b in pairs),
                unchanged=sum(b==a for a,b in pairs),difference=stats([b-a for a,b in pairs]))
        result['pairs'][candidate]=paired
    result['independent_checks']=dict(full_unique_ids=True,all_terminal=True,baseline_valid_3180=True,paired_partition_and_net=True)
    (HERE/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with gzip.open(HERE/'per_cad.csv.gz','wt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(compact[0]))
        writer.writeheader();writer.writerows(compact)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
