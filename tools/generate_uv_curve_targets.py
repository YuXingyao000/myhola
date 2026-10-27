"""Generate current-chart GT UV caches, with fixed-list GPU sharding.

Example: python tools/generate_uv_curve_targets.py --raw-root ... --list ...
  --output-root ... --device cuda:0 --shards 7 --shard-index 0 --batch-cads 8
No training, model checkpoint, or OCC/native-UV labels are used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.brepnet.data.uv_curve_targets import (
    FixedSurfaceProjector, REASON_FLAGS, REPRESENTATION, SURFACE_SIGNATURE,
    UVProjectionConfig, load_uv_curve_targets, parameters_json, read_projection_source,
    sha256_file, write_uv_curve_targets,
)


def _metrics(ident, arrays, seconds, status="ok"):
    valid = arrays["valid"]
    residual = arrays["residual"]
    good = residual[valid]
    return dict(id=ident, status=status, seconds=seconds, source_sha256=str(arrays["source_sha256"]),
                edges=len(valid), points=valid.size, valid_points=int(valid.sum()),
                complete_owner_paths=int(valid.all(-1).sum()), owner_paths=2*len(valid),
                residual_p50_p90_p99_max=np.percentile(residual, [50,90,99,100]).tolist() if residual.size else None,
                valid_residual_p50_p90_p99_max=np.percentile(good, [50,90,99,100]).tolist() if good.size else None,
                reasons={name:int(((arrays["reason_flags"] & flag) != 0).sum()) for name,flag in REASON_FLAGS.items()})


def synthetic_checks(projector):
    """Known chart solutions, invalid targets, closure, reversal, and rigid rotation."""
    x=np.linspace(0.,1.,16);v,u=np.meshgrid(x,x,indexing="ij")
    plane=np.stack((u,v,np.zeros_like(u)),-1)
    angle=2*np.pi*x
    boundary=np.stack((x,np.zeros_like(x),np.zeros_like(x)),-1)
    circle=np.stack((.5+.3*np.cos(angle),.5+.3*np.sin(angle),np.zeros_like(x)),-1)
    collapsed=np.tile([.5,.5,0.],(16,1))
    unreachable=boundary+np.array([0.,0.,.1])
    paths=np.stack((boundary,boundary[::-1],circle,collapsed,unreachable))
    source=dict(sample_id="synthetic",source_sha256="0"*64,faces=np.stack((plane,plane)),
                boundary=dict(curves=paths,face_pairs=np.tile([0,1],(len(paths),1)),
                              source_edge_ids=np.arange(len(paths)),is_closed=np.array([False,False,True,True,False])))
    # A synthetic source can repeat owners to exercise multiple known paths;
    # this bypasses the real-data grouping, not cache alignment validation.
    labels=projector.project_batch([source])[0]
    expected=paths[:3,:,:2]
    error=float(np.max(np.abs(labels["uv"][:3]-expected[:,None])))
    if not labels["valid"][:3].all() or error>2e-5:
        raise AssertionError(f"known plane paths failed, max UV error {error}")
    if labels["valid"][3:].any():
        raise AssertionError("collapsed/unreachable paths must be invalid")
    rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    transformed={**source,"faces":source["faces"]@rot.T,"boundary":{**source["boundary"],"curves":paths@rot.T}}
    rotated=projector.project_batch([transformed])[0]
    rotate_error=float(np.max(np.abs(labels["uv"]-rotated["uv"])))
    if rotate_error>2e-5 or not np.array_equal(labels["valid"],rotated["valid"]):
        raise AssertionError("rigid rotation changed UV targets or mask")
    singular={**source,"faces":np.zeros_like(source["faces"])}
    singular_labels=projector.project_batch([singular])[0]
    if singular_labels["valid"].any() or not np.any(singular_labels["reason_flags"]&2):
        raise AssertionError("degenerate Jacobian was not masked")
    # Exact boundary and derivative behavior must match the current Torch module.
    from src.brepnet.models.vae.uv_intersection import FixedBSplineSurface
    reference=FixedBSplineSurface().to(projector.surface.device)
    with torch.inference_mode():
        gt=projector.surface.tensor(source["faces"])
        uv=projector.surface.tensor(labels["uv"][0])
        ours=projector.surface.evaluate(projector.surface.fit(gt),uv)
        theirs=reference.evaluate(reference.fit(gt),uv)
        reference_error=float((ours-theirs).abs().max().cpu())
    if reference_error>1e-10:
        raise AssertionError("projection surface differs from model surface")
    return dict(status="passed",max_known_uv_error=error,rotation_uv_error=rotate_error,
                model_surface_max_abs_difference=reference_error,
                valid_known_points=int(labels["valid"][:3].sum()),masked_invalid_points=int((~labels["valid"][3:]).sum()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root",type=Path)
    ap.add_argument("--list",type=Path,action="append",default=[])
    ap.add_argument("--output-root",type=Path,required=True)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--batch-cads",type=int,default=8)
    ap.add_argument("--shards",type=int,default=1)
    ap.add_argument("--shard-index",type=int,default=0)
    ap.add_argument("--limit",type=int)
    ap.add_argument("--config-json",type=Path)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--synthetic-check",action="store_true")
    args=ap.parse_args()
    if args.batch_cads<1 or args.shards<1 or not 0<=args.shard_index<args.shards:
        ap.error("invalid batch/shard settings")
    if not args.synthetic_check and (args.raw_root is None or not args.list):
        ap.error("--raw-root and at least one --list are required")
    config=UVProjectionConfig(**json.loads(args.config_json.read_text())) if args.config_json else UVProjectionConfig()
    config.validate();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    projector=FixedSurfaceProjector(args.device,config)
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")+f"-shard{args.shard_index:03d}"
    report_root=args.output_root/"_reports";report_root.mkdir(parents=True,exist_ok=True)
    log_path=report_root/f"{run_id}.jsonl"
    params=parameters_json(config);param_sha=hashlib.sha256(params.encode()).hexdigest()
    metadata=dict(run_id=run_id,representation=REPRESENTATION,surface_signature=SURFACE_SIGNATURE,
                  device=args.device,torch_version=torch.__version__,parameters=asdict(config),parameters_sha256=param_sha,
                  script_sha256=sha256_file(__file__),module_sha256=sha256_file(Path(__file__).resolve().parents[1]/"src/brepnet/data/uv_curve_targets.py"),
                  raw_root=str(args.raw_root),lists=[dict(path=str(p),sha256=sha256_file(p)) for p in args.list],
                  shards=args.shards,shard_index=args.shard_index,batch_cads=args.batch_cads,limit=args.limit)
    if args.synthetic_check:
        started=time.monotonic();metadata["synthetic_checks"]=synthetic_checks(projector)
        metadata["seconds"]=time.monotonic()-started
        (report_root/f"{run_id}.json").write_text(json.dumps(metadata,indent=2)+"\n")
        print(json.dumps(metadata,indent=2),flush=True)
        return
    ids=[line.strip() for path in args.list for line in path.read_text().splitlines() if line.strip()]
    if len(set(ids))!=len(ids):raise ValueError("Input lists overlap or contain duplicate IDs")
    if args.limit is not None:
        if args.limit<1:raise ValueError("limit must be positive")
        ids=ids[:args.limit]
    ids=ids[args.shard_index::args.shards]
    metadata["planned_ids"]=ids;metadata["planned"]=len(ids)
    (report_root/f"{run_id}.manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")
    rows=[];started=time.monotonic()
    def record(row):
        rows.append(row)
        with log_path.open("a") as stream:stream.write(json.dumps(row,allow_nan=False)+"\n")
    for offset in range(0,len(ids),args.batch_cads):
        sources=[]
        for ident in ids[offset:offset+args.batch_cads]:
            try:
                source=read_projection_source(args.raw_root/ident/"data.npz",config,ident)
                output=args.output_root/ident/"uv_curve_targets.npz"
                if output.exists():
                    if not args.resume:raise FileExistsError(output)
                    cached=load_uv_curve_targets(output,source_sha256=source["source_sha256"],
                            face_pairs=source["boundary"]["face_pairs"],source_edge_ids=source["boundary"]["source_edge_ids"],expected_parameters_sha256=param_sha)
                    record(_metrics(ident,cached,0.,"reused"))
                else:sources.append(source)
            except Exception as error:record(dict(id=ident,status="failed",error=f"{type(error).__name__}: {error}"))
        if sources:
            begin=time.monotonic()
            try:
                outputs=projector.project_batch(sources)
                duration=time.monotonic()-begin
            except Exception as error:
                for source in sources:record(dict(id=source["sample_id"],status="failed",error=f"batch projection: {type(error).__name__}: {error}"))
            else:
                for source,arrays in zip(sources,outputs):
                    ident=source["sample_id"]
                    try:
                        write_uv_curve_targets(args.output_root/ident/"uv_curve_targets.npz",arrays)
                        record(_metrics(ident,arrays,duration/len(sources)))
                    except Exception as error:
                        record(dict(id=ident,status="failed",error=f"cache write: {type(error).__name__}: {error}"))
        print(json.dumps(dict(processed=len(rows),planned=len(ids),elapsed_seconds=time.monotonic()-started,
                              failures=sum(r["status"]=="failed" for r in rows))),flush=True)
    good=[r for r in rows if r["status"] in ("ok","reused")]
    summary={**metadata,"seconds":time.monotonic()-started,"ok":len(good),"failed":len(rows)-len(good),
             "totals":{key:sum(r[key] for r in good) for key in ("edges","points","valid_points","complete_owner_paths","owner_paths")},
             "reason_counts":{name:sum(r["reasons"][name] for r in good) for name in REASON_FLAGS}}
    (report_root/f"{run_id}.summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="planned_ids"},indent=2),flush=True)
    if summary["failed"]:raise SystemExit(1)


if __name__=="__main__":main()
