"""Independently reread final STEP files without fitting, repair, or geometry metrics."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

CODE = Path('/mnt/d/python_uv_train/checkpoints/conditional_intersection/feature_on_evaluation_20261029')
HERE = Path('/root/hola_eval/conditional_feature_on_20261029')


def initialize():
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name]='1'
    from OCC.Core.Interface import Interface_Static
    Interface_Static.SetIVal('read.precision.mode',0)


def check(job):
    sid,path,recorded=job
    result=dict(id=sid,recorded_native_valid=bool(recorded),readable=False,native_valid=False)
    path=Path(path)
    if not path.is_file():
        result['status']='missing_step'
        return result
    result['step_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        from OCC.Extend.DataExchange import read_step_file
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        from OCC.Core.TopAbs import TopAbs_SOLID
        from OCC.Core.Interface import Interface_Static
        Interface_Static.SetIVal('read.precision.mode',0)
        shape=read_step_file(str(path),as_compound=False,verbosity=False)
        if shape is None or shape.IsNull():
            raise ValueError('Null shape')
        result.update(status='ok',readable=True,is_solid=shape.ShapeType()==TopAbs_SOLID,
                      brepcheck_valid=bool(BRepCheck_Analyzer(shape).IsValid()))
        result['native_valid']=result['is_solid'] and result['brepcheck_valid']
    except Exception as error:
        result.update(status='read_error',error=repr(error))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',required=True)
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args()
    root=HERE/args.method
    records=[json.loads(p.read_text()) for p in (root/'metadata/post/optimized/samples'/args.method).glob('*.json')]
    assert len(records)==len({r['sample_id'] for r in records})==3495
    jobs=[(r['sample_id'],str(root/'post/optimized'/args.method/r['sample_id']/'recon_brep.step'),r.get('native_valid_solid')) for r in records]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
        results=list(pool.map(check,jobs,chunksize=8))
    mismatches=[r for r in results if r['recorded_native_valid']!=r['native_valid']]
    report=dict(method=args.method,count=len(results),native_valid=sum(r['native_valid'] for r in results),
        readable=sum(r['readable'] for r in results),status_counts=dict(Counter(r['status'] for r in results)),
        mismatch_count=len(mismatches),mismatches=mismatches,passed=not mismatches,
        precision_mode=0,criterion='as-read SOLID and BRepCheck before mutation')
    out=HERE/'validation'
    out.mkdir(parents=True,exist_ok=True)
    (out/f'{args.method}_step_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    import gzip
    with gzip.open(out/f'{args.method}_step_manifest.jsonl.gz','wt') as stream:
        for row in results:
            stream.write(json.dumps(row)+'\n')
    print(json.dumps(report,indent=2))
    if mismatches:
        raise SystemExit(2)


if __name__=='__main__':
    main()
