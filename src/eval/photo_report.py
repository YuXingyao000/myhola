"""Read the frozen three-arm photo evaluation and draw scientific result tables."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from src.eval.photo_baseline import METHODS
from src.eval.preference_report import EVALUATION_TERMINAL, finite_number, statistics
from src.common.io import atomic_json, read_json


LABELS = ("原始 SVR → 照片", "照片 4,553 步\n最低验证 loss", "照片 310,000 步\n最后完整权重")
METRICS = ("face_cd_sq", "edge_cd_sq", "face_f1")


def load_rows(run: Path):
    protocol = read_json(run / "metadata/protocol.json")
    ids = protocol["identities"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Require a nonempty, unique frozen validation population")
    rows = {method: [] for method in METHODS}
    for mid in ids:
        seeds = []
        for method in METHODS:
            infer = read_json(run / "raw_predictions" / method / mid / "inference.json")
            if infer.get("status") not in {"ok", "generation_error", "interrupted"}:
                raise ValueError(f"Unfinished inference: {method}/{mid}")
            if infer["input_sha256"] != protocol["input_sha256"][mid]:
                raise ValueError("Unpaired inputs")
            expected = (protocol["reuse_fingerprint"] if mid in protocol.get("reused_ids", [])
                        else protocol["fingerprint"])
            if (infer.get("protocol_fingerprint") != expected or
                    infer.get("method") != method or infer.get("image_id") != mid):
                raise ValueError(f"Inference provenance differs: {method}/{mid}")
            seeds.append(infer["seed"])
            result = read_json(run / "evaluations" / method / mid / "metrics.json") if infer["status"] == "ok" else {}
            if infer["status"] == "ok" and result.get("status") not in EVALUATION_TERMINAL:
                raise ValueError(f"Unfinished evaluation: {method}/{mid}")
            raw, post = result.get("raw", {}), result.get("post", [])
            complete = (result.get("validity") == "valid" and len(post) == 3
                        and {p["repeat"] for p in post} == {0, 1, 2}
                        and all(finite_number(p.get(k)) for p in post for k in METRICS))
            row = {"image_id": mid, "method": method, "seed": infer["seed"],
                   "inference_status": infer["status"], "evaluation_status": result.get("status", "not_run"),
                   "validity": result.get("validity", "unknown"), "post_complete": complete,
                   "step_readable": bool(result.get("native", {}).get("step_readable")),
                   "step_exists": (run / "evaluations" / method / mid / "post/c00/recon_brep.step").is_file(),
                   "inference_seconds": infer["seconds"], "evaluation_seconds": result.get("wall_seconds"),
                   "error": result.get("error") or infer.get("error", "")}
            for k in METRICS:
                row["raw_"+k] = raw.get(k) if finite_number(raw.get(k)) else None
                row["post_"+k] = float(np.mean([p[k] for p in post])) if complete else None
            counts = [raw.get(k) for k in ("pred_face_count", "gt_face_count")]
            row["raw_face_count_abs_error"] = abs(counts[0]-counts[1]) if all(map(finite_number, counts)) else None
            rows[method].append(row)
        if len(set(seeds)) != 1:
            raise ValueError("Unpaired random seeds")
    return protocol, rows


def paired_delta(left, right, key):
    if len(left) != len(right):
        raise ValueError("Paired populations differ in size")
    if any(a.get("image_id") != b.get("image_id") for a, b in zip(left, right)):
        raise ValueError("Paired CAD order differs")
    paired = [(a[key], b[key]) for a,b in zip(left, right)
              if finite_number(a.get(key)) and finite_number(b.get(key))]
    if not paired:
        return {"n": 0, "mean_delta": None, "ci95": None}
    difference = np.diff(np.array(paired), axis=1)[:,0]
    rng = np.random.default_rng(42)
    # Bound memory while preserving the same seeded bootstrap draw sequence.
    means = np.empty(10000)
    for start in range(0, 10000, 100):
        indices = rng.integers(len(difference), size=(100, len(difference)))
        means[start:start+100] = difference[indices].mean(1)
    return {"n": len(paired), "mean_delta": float(difference.mean()),
            "ci95": np.quantile(means, [.025,.975]).tolist(), "bootstrap_repeats": 10000}


def summarize(rows):
    ids = [r["image_id"] for r in rows[METHODS[0]]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Require nonempty unique CAD IDs")
    if any([r["image_id"] for r in rows[m]] != ids for m in METHODS):
        raise ValueError("Methods must cover the same ordered CAD population")
    result = {"intended_per_method": len(ids), "methods": {}, "pairwise": {},
              "scope": "development validation; step4553 selected by loss on the original validation100 only",
              "cd": "sum of two directed mean squared nearest distances; no sqrt or registration"}
    keys = [f"{stage}_{k}" for stage in ("raw", "post") for k in METRICS] + ["raw_face_count_abs_error"]
    common_post = set.intersection(*[{r["image_id"] for r in rows[m] if r["post_complete"]} for m in METHODS])
    result["all_three_common_post_ids"] = sorted(common_post)
    for method in METHODS:
        records = rows[method]
        result["methods"][method] = {
            "generated": sum(r["inference_status"] == "ok" for r in records),
            "step_outputs": sum(r["step_exists"] for r in records),
            "readable_steps": sum(r["step_readable"] for r in records),
            "valid_solids": sum(r["validity"] == "valid" for r in records),
            "invalid": sum(r["validity"] == "invalid" for r in records),
            "unknown": sum(r["validity"] == "unknown" for r in records),
            "timeouts": sum(r["evaluation_status"] == "timeout" for r in records),
            "status_counts": dict(Counter(r["evaluation_status"] for r in records)),
            "metrics": {k: statistics([r[k] for r in records if finite_number(r[k])]) for k in keys},
            "post_on_all_three_common": {k: statistics([r["post_"+k] for r in records if r["image_id"] in common_post]) for k in METRICS},
            "inference_seconds": statistics([r["inference_seconds"] for r in records]),
        }
    for a,b in ((METHODS[0],METHODS[1]),(METHODS[0],METHODS[2]),(METHODS[1],METHODS[2])):
        left=[{**r,"valid":int(r['validity']=='valid')} for r in rows[a]]
        right=[{**r,"valid":int(r['validity']=='valid')} for r in rows[b]]
        result["pairwise"][f"{b}_minus_{a}"]={key:paired_delta(left,right,key) for key in ["valid",*keys]}
    return result


def setup_plot():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    for name in ("/tmp/NotoSansCJKsc-Regular.otf", "/tmp/NotoSansCJKsc-Bold.otf"):
        if Path(name).exists():
            font_manager.fontManager.addfont(name)
    plt.rcParams.update({"font.family":"Noto Sans CJK SC", "axes.unicode_minus":False,
                         "savefig.facecolor":"#f5f7fa", "figure.facecolor":"#f5f7fa"})
    return plt


def result_table(run: Path, summary):
    plt = setup_plot()
    metrics=summary['methods']
    lines=[]
    def add(title, values): lines.append([title,*values])
    total = summary["intended_per_method"]
    for name,key in (("生成成功","generated"),("有效 solid ↑","valid_solids"),
                     ("产出 STEP","step_outputs"),("超时","timeouts")):
        add(name,[f"{metrics[m][key]} / {total}  ({metrics[m][key]/total:.1%})" for m in METHODS])
    for title,key,scale in (("原始面 CD² × 1000 ↓","raw_face_cd_sq",1000),
                            ("原始边 CD² × 1000 ↓","raw_edge_cd_sq",1000),
                            ("原始面 F1 ↑","raw_face_f1",100),
                            ("面数绝对误差 ↓","raw_face_count_abs_error",1),
                            ("有效 solid 面 CD² × 1000 ↓","post_face_cd_sq",1000)):
        values=[]
        for method in METHODS:
            s=metrics[method]['metrics'][key]
            values.append(f"{s['mean']*scale:.3f}{'%' if key.endswith('f1') else ''}  (n={s['n']})" if s['n'] else '—  (n=0)')
        add(title,values)
    add("三组共同有效子集面 CD² × 1000 ↓",[
        f"{metrics[m]['post_on_all_three_common']['face_cd_sq']['mean']*1000:.3f}  (n={len(summary['all_three_common_post_ids'])})"
        if summary['all_three_common_post_ids'] else '—  (n=0)' for m in METHODS])
    fig,ax=plt.subplots(figsize=(15,8.8));ax.axis('off')
    fig.text(.045,.95,f'照片条件重建：固定 {total:,} 个验证对象',fontsize=23,weight='bold',color='#142b46')
    fig.text(.045,.90,'相同照片 · 相同 seed · 每图 1 个候选 · 原版 1000-step DDPM · 同一 VAE / 后处理',fontsize=12,color='#536578')
    table=ax.table(cellText=lines,colLabels=['指标',*LABELS],cellLoc='center',
                   colWidths=[.34,.22,.22,.22],bbox=[0,.12,1,.78])
    table.auto_set_font_size(False);table.set_fontsize(11)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#dbe3ec');cell.set_linewidth(.6)
        if row==0:cell.set_facecolor('#16324f');cell.set_text_props(color='white',weight='bold')
        else:
            cell.set_facecolor('#ffffff' if row%2 else '#edf2f7')
            if col==0:cell.set_text_props(ha='left',color='#24384d')
    fig.text(.045,.075,f'失败与超时保留在 {total:,} 的分母；几何均值旁标注实际测量 n。CD² 为双向平方距离之和。',fontsize=10,color='#536578')
    fig.text(.045,.045,'4,553 步只按原 100 图的验证 loss 选出；这是合成照片 validation，不是独立 test 或实拍评价。',fontsize=10,color='#536578')
    fig.savefig(run/'comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
    return lines


def training_curve(run: Path, training_run: Path):
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2
    path=next((training_run/'wandb').glob('run-*/*.wandb'))
    stream=DataStore();stream.open_for_scan(str(path));train=[];val=[]
    while True:
        data=stream.scan_data()
        if data is None:break
        record=wandb_internal_pb2.Record();record.ParseFromString(data)
        if not record.HasField('history'):continue
        row={'.'.join(x.nested_key) if x.nested_key else x.key:json.loads(x.value_json) for x in record.history.item}
        step=row.get('trainer/global_step')
        if step is None:continue
        if 'train/loss_step_local' in row:train.append((step,row['train/loss_step_local']))
        if 'val/loss' in row:val.append((step,row['val/loss']))
    stream.close()
    train,val=np.array(train),np.array(val)
    if not len(train) or not len(val) or not np.isfinite(train).all() or not np.isfinite(val).all():
        raise ValueError('Missing/nonfinite training history')
    result={'train_points':len(train),'validation_points':len(val),
            'first_100_train_mean':float(train[:100,1].mean()),'last_100_train_mean':float(train[-100:,1].mean()),
            'first_validation':val[0].tolist(),'last_validation':val[-1].tolist(),
            'minimum_validation':val[val[:,1].argmin()].tolist()}
    atomic_json(run/'training_summary.json',result)
    with (run/'training_history.csv').open('w') as handle:
        writer=csv.writer(handle,lineterminator='\n');writer.writerow(['kind','global_step','loss'])
        for kind,values in [('train_local',train),('validation',val)]:
            writer.writerows((kind,int(step),loss) for step,loss in values)
    plt=setup_plot();fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    smooth=np.convolve(train[:,1],np.ones(100)/100,mode='valid')
    axes[0].plot(train[99:,0]/1000,smooth,color='#247a91',lw=1.5)
    axes[0].set_title('训练 loss：100 个日志点移动均值')
    axes[1].plot(val[:,0]/1000,val[:,1],'.-',color='#d06c48',lw=1)
    axes[1].set_title('Validation loss：固定 100 图，随机 t / 噪声')
    for ax in axes:
        ax.set_xlabel('新增照片训练步数 / 千步');ax.set_ylabel('epsilon MSE');ax.grid(alpha=.2)
        ax.axvline(4.553,color='#60748c',ls='--',alpha=.6)
        ax.axvline(310,color='#60748c',ls='--',alpha=.6)
    fig.tight_layout();fig.savefig(run/'training_curve.png',dpi=180);plt.close(fig)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--training-run',required=True)
    args=parser.parse_args();run=Path(args.run_dir)
    protocol,rows=load_rows(run)
    summary=summarize(rows)
    if protocol.get("reused_ids"):
        original = set(protocol["reused_ids"])
        summary["subsets"] = {
            name: summarize({m: [r for r in rows[m] if (r["image_id"] in original) == included]
                             for m in METHODS})
            for name, included in (("original_monitoring100", True), ("additional_validation", False))
        }
    atomic_json(run/'summary.json',summary)
    with (run/'per_object.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[METHODS[0]][0]),lineterminator='\n');writer.writeheader()
        for method in METHODS:writer.writerows(rows[method])
    lines=result_table(run,summary)
    training=training_curve(run,Path(args.training_run))
    markdown=['# 照片 baseline 固定验证结果','',
              f'固定 validation {summary["intended_per_method"]} 件；三组均输入相同合成照片，K=1、相同 seed。原始 SVR 行是原权重直接接收照片，不能与此前灰模成绩直接混合。','',
              '| 指标 | 原始 SVR | 照片4553步（最低val loss） | 照片310000步 |','|---|---:|---:|---:|']
    markdown.extend('| '+' | '.join(row)+' |' for row in lines)
    markdown.extend(['','4,553步只使用原100件的验证 loss 选模；这不是保留 test。CD为双向均值平方距离之和，不开方；face F1阈值0.1。后处理为 recover_v1，每件整体180秒、200次优化；有效STEP几何重复采样3次。',
                     '',f"训练前100个日志点均值 {training['first_100_train_mean']:.6f}，后100个 {training['last_100_train_mean']:.6f}；末次验证 {training['last_validation'][1]:.6f}。loss不能直接替代下游质量。",'',
                     '逐件状态见 per_object.csv；配对差值及95% bootstrap区间见 summary.json，缺失几何未补零。协议及源代码摘要见 metadata/protocol.json。'])
    if "subsets" in summary:
        markdown.extend(['', '## 原100件与新增验证对象', '',
                         '| 子集 | 方法 | 有效 solid | 原始面 CD² × 1000 | 原始面 F1 |',
                         '|---|---|---:|---:|---:|'])
        for name, subset in summary["subsets"].items():
            for method in METHODS:
                values = subset["methods"][method]
                cd = values["metrics"]["raw_face_cd_sq"]["mean"]
                f1 = values["metrics"]["raw_face_f1"]["mean"]
                cd_text = f'{cd*1000:.3f}' if cd is not None else '缺失'
                f1_text = f'{f1:.2%}' if f1 is not None else '缺失'
                markdown.append(f'| {name} | {method} | {values["valid_solids"]}/{subset["intended_per_method"]} | {cd_text} | {f1_text} |')
    (run/'README.md').write_text('\n'.join(markdown)+'\n')
    print(json.dumps({'methods':summary['methods'],'training':training},indent=2))


if __name__=='__main__':main()
