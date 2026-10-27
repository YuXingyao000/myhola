"""Fixed-list CPU diagnosis; no model, fitting optimization, or raw mutation."""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.interpolate import BSpline


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(delta):
    delta = np.asarray(delta, dtype=np.float64).reshape(-1, 3)
    if not len(delta):
        return dict(points=0, mae=None, rms=None, max=None, abs_sum=0., square_sum=0.)
    return dict(points=len(delta), mae=float(np.abs(delta).mean()),
                rms=float(np.sqrt(np.square(delta).sum(1).mean())),
                max=float(np.linalg.norm(delta, axis=1).max()),
                abs_sum=float(np.abs(delta).sum()), square_sum=float(np.square(delta).sum()))


def curve_stats(curves, gt):
    if not len(curves):
        return dict(edges=0, mse=None, mse_sum=0., gap=stats(np.empty((0,3))), gap_mean=None)
    f = np.square(curves-gt[:,None]).mean((1,2,3))
    r = np.square(curves-gt[:,None,::-1]).mean((1,2,3))
    gap = curves[:,0]-curves[:,1]
    values = np.minimum(f,r)
    return dict(edges=len(curves), mse=float(values.mean()), mse_sum=float(values.sum()),
                gap=stats(gap), gap_mean=float(np.linalg.norm(gap,axis=-1).mean()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',required=True);ap.add_argument('--list',required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    ids=Path(args.list).read_text().splitlines()[:32]
    if len(ids)!=32 or len(set(ids))!=32:raise ValueError('expected 32 unique fixed IDs')
    (out/'ids.txt').write_text('\n'.join(ids)+'\n')
    x=np.linspace(0.,1.,16);breaks=np.linspace(0.,1.,14)
    knots=np.r_[np.repeat(breaks[0],3),breaks,np.repeat(breaks[-1],3)]
    basis=BSpline(knots,np.eye(16),3,extrapolate=False)
    B=basis(x);A=np.linalg.solve(B,np.eye(16));t=np.linspace(0.,1.,65);H=basis(t)@A
    dd=np.linspace(0.,1.,4001);HH=basis(dd)@A;l2=np.linalg.norm(HH,axis=1)
    uvv,uuu=np.meshgrid(t,t,indexing='ij');boundary=np.minimum.reduce([uvv,uuu,1-uvv,1-uuu]);band=(boundary>0)&(boundary<.05)
    bins=np.array([0.,.001,.01,.05,.1,.25,.5])
    def fit(points):return np.einsum('bi,fijc,aj->fbac',A,points,A,optimize=True)
    def evaluate(poles,uv):return np.einsum('...ta,...bac,...tb->...tc',basis(uv[...,0]),poles,basis(uv[...,1]),optimize=True)
    rows=[]
    for ident in ids:
        row=dict(id=ident,status='failed')
        try:
            file=Path(args.raw)/ident/'data.npz';row['input_sha256']=sha(file)
            with np.load(file,allow_pickle=False) as z:a={k:z[k] for k in z.files}
            pred=np.asarray(a['pred_face'],np.float64);gt=np.asarray(a['gt_face'],np.float64)
            uv=np.asarray(a['pred_edge_uv'],np.float64);curves=np.asarray(a['pred_edge_xyz_by_face'],np.float64)
            pairs=np.asarray(a['pred_edge_owner_pairs']);gpairs=np.asarray(a['gt_edge_owner_pairs'])
            gedges=np.asarray(a['gt_edge'],np.float64)
            if pred.shape!=gt.shape or pred.shape[1:]!=(16,16,3):raise ValueError('face shape')
            if uv.shape!=(len(pairs),2,16,2) or curves.shape!=(len(pairs),2,16,3):raise ValueError('curve shape')
            if any(not np.isfinite(v).all() for v in [pred,gt,uv,curves,gedges]):raise ValueError('nonfinite')
            if np.any((uv<0)|(uv>1)):raise ValueError('UV outside domain')
            if len({tuple(p) for p in pairs})!=len(pairs) or len({tuple(p) for p in gpairs})!=len(gpairs):raise ValueError('duplicate pair')
            difference=pred-gt;dense=np.einsum('vi,fijc,uj->fvuc',H,difference,H,optimize=True)
            ppol=fit(pred);gpol=fit(gt);recalc=evaluate(ppol[pairs],uv)
            lookup={tuple(p):i for i,p in enumerate(gpairs)}
            selected=[i for i,p in enumerate(pairs) if tuple(p) in lookup]
            gi=[lookup[tuple(pairs[i])] for i in selected]
            oracle=evaluate(gpol[pairs[selected]],uv[selected])
            bd=np.minimum(uv,1-uv);point_bd=bd.min(axis=-1)
            row.update(status='ok',faces=len(pred),active_edges=len(pairs),gt_edges=len(gpairs),matched_edges=len(selected),
                       nodes=stats(difference),dense=stats(dense),near_boundary=stats(dense[:,band]),
                       original=curve_stats(curves[selected],gedges[gi]),oracle_gt_surface=curve_stats(oracle,gedges[gi]),
                       uv_coordinate_bin_counts=np.histogram(bd,bins)[0].tolist(),uv_coordinate_count=bd.size,
                       uv_point_bin_counts=np.histogram(point_bd,bins)[0].tolist(),uv_point_count=point_bd.size,
                       uv_recalculation_max_abs=float(np.max(np.abs(recalc-curves))) if curves.size else None,
                       grid_reconstruction_max_abs=float(np.max(np.abs(np.einsum('ib,fbac,ja->fijc',B,ppol,B,optimize=True)-pred))))
        except Exception as error:row['error']=f'{type(error).__name__}: {error}'
        rows.append(row)
    good=[r for r in rows if r['status']=='ok']
    def pooled(values):
        n=sum(v['points'] for v in values);ab=sum(v['abs_sum'] for v in values);sq=sum(v['square_sum'] for v in values)
        return dict(points=n,mae=ab/(3*n) if n else None,rms=np.sqrt(sq/n).item() if n else None,max=max([v['max'] for v in values if v['max'] is not None],default=None))
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),list_sha256=sha(args.list),
                 raw=args.raw,requested=32,ok=len(good),failed=32-len(good),ids=ids,
                 interpretation='Same-parameter pred-spline minus GT-spline, not error to true CAD surface. Oracle replaces surface only at unchanged predicted UV; no projection or optimization.',
                 matrix=dict(cond_B=float(np.linalg.cond(B)),dense_probe_count=4001,max_H_l2=float(l2.max()),max_H_l2_u=float(dd[l2.argmax()]),max_H_l1=float(np.abs(HH).sum(1).max()),max_2d_white_noise_std_gain=float(l2.max()**2),partition_unity_error=float(np.abs(HH.sum(1)-1).max())),
                 bins=bins.tolist(),coverage={key:sum(r[key] for r in good) for key in ['faces','active_edges','gt_edges','matched_edges']})
    for key in ['nodes','dense','near_boundary']:
        summary[key]=dict(pooled=pooled([r[key] for r in good]),cad_macro={metric:float(np.mean([r[key][metric] for r in good])) if good else None for metric in ['mae','rms','max']})
    for key in ['original','oracle_gt_surface']:
        vals=[r[key] for r in good if r[key]['edges']];n=sum(v['edges'] for v in vals)
        summary[key]=dict(edges=n,pooled_mse=sum(v['mse_sum'] for v in vals)/n if n else None,cad_macro_mse=float(np.mean([v['mse'] for v in vals])) if vals else None,gap=pooled([v['gap'] for v in vals]),pooled_gap_mean=sum(v['gap_mean']*v['gap']['points'] for v in vals)/sum(v['gap']['points'] for v in vals) if vals else None)
    for key in ['uv_coordinate','uv_point']:
        counts=np.sum([r[key+'_bin_counts'] for r in good],axis=0) if good else np.zeros(6,int);n=sum(r[key+'_count'] for r in good)
        summary[key]=dict(count=n,bin_counts=counts.tolist(),bin_fractions=(counts/n).tolist() if n else None)
    summary['sanity']={key:max([r[key] for r in good if r[key] is not None],default=None) for key in ['uv_recalculation_max_abs','grid_reconstruction_max_abs']}
    (out/'per_cad.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
