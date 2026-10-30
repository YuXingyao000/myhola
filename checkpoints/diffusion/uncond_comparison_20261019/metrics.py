"""复用仓库的生成集合、图同构与复杂度指标；记录每个可计算分母。"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BASE = ROOT.parent / "uncond_baseline_20261018"
DATA = Path("/mnt/d/data/deepcad_v7")
GROUPS = {"old": BASE, "new_last": ROOT / "new_last", "new_best": ROOT / "new_best"}
sys.path.insert(0, str(REPO))


def write_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def point_job(job):
    import numpy as np
    import trimesh
    from OCC.Core.Interface import Interface_Static
    from src.brepnet.eval.metrics.point_cloud_set import normalize_pc, align_pc
    from src.brepnet.eval.metrics.complexity import evaluate_step_file
    from checkpoints.diffusion.uncond_baseline_20261018.visualize import step_mesh
    group, name, folder, native = job
    destination = ROOT / "features" / group / name
    result = {"sample_id": name, "native_valid_solid": native, "pointcloud_ok": False, "complexity": None}
    seed = int.from_bytes(hashlib.sha256(("uncond-mesh-20261019:"+name).encode()).digest()[:4], "little")
    try:
        if group == "reference":
            mesh_path = folder / "mesh.ply"
            step = folder / "normalized_shape.step"
        else:
            mesh_path = folder / "recon_brep.stl"
            step = folder / "recon_brep.step"
        if mesh_path.is_file():
            mesh = trimesh.load(mesh_path, force="mesh", process=False)
            result["mesh_source"] = str(mesh_path)
        else:
            Interface_Static.SetIVal("read.precision.mode", 0)
            vertices, faces, _ = step_mesh(step)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            result["mesh_source"] = str(step)
        assert len(mesh.faces) and np.isfinite(mesh.vertices).all() and mesh.area > 0
        points, _ = trimesh.sample.sample_surface(mesh, 2000, seed=seed)
        points = align_pc(normalize_pc(points)).astype(np.float32)
        assert points.shape == (2000, 3) and np.isfinite(points).all()
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination.with_suffix(".npy"), points)
        result["pointcloud_ok"] = True
    except Exception as error:
        result["pointcloud_error"] = f"{type(error).__name__}: {error}"
    write_json(destination.with_suffix(".json"), result)
    # 使用现有复杂度定义及其0.1容差gate，独立记录不可计算样本。
    try:
        Interface_Static.SetIVal("read.precision.mode", 1)
        Interface_Static.SetRVal("read.precision.val", 0.1)
        complexity = evaluate_step_file(step)
        if complexity is not None:
            complexity = {key: (float(value) if isinstance(value, np.generic) else value) for key,value in complexity.items()}
        result["complexity"] = complexity
    except Exception as error:
        result["complexity_error"] = f"{type(error).__name__}: {error}"
    if group == "reference":
        with np.load(folder / "data.npz") as raw:
            result["raw_num_faces"] = len(raw["sample_points_faces"])
    write_json(destination.with_suffix(".json"), result)
    return result


def prepare_points(group):
    # 预加载后再fork；每件的OCC三角化拥有独立的180秒硬预算。
    import trimesh
    from src.brepnet.eval.metrics.point_cloud_set import normalize_pc
    from src.brepnet.eval.metrics.complexity import evaluate_step_file
    from checkpoints.diffusion.uncond_baseline_20261018.visualize import step_mesh
    jobs=[]
    if group == "reference":
        for name in (ROOT / "reference_ids.txt").read_text().splitlines():
            jobs.append((group, name, DATA/name, None))
    else:
        root=GROUPS[group]
        for path in sorted((root / "metadata/post/samples").glob("*.json")):
            row=json.loads(path.read_text())
            if row["status"] != "running":
                jobs.append((group, row["sample_id"], root/"post"/row["sample_id"], row.get("native_valid_solid")))
    jobs=[job for job in jobs if not (ROOT/"features"/group/f"{job[1]}.json").exists()]
    print(f"Pointcloud/complexity {group}: {len(jobs)} jobs",flush=True)
    with ProcessPoolExecutor(max_workers=16) as pool:
        for index,result in enumerate(pool.map(timed_point_job,jobs)):
            if (index+1)%50==0:
                print(f"{group}: {index+1}/{len(jobs)}",flush=True)


def timed_point_job(job):
    process=mp.get_context("fork").Process(target=point_job,args=(job,))
    process.start()
    process.join(180)
    group,name,folder,native=job
    destination=ROOT/"features"/group/f"{name}.json"
    if process.is_alive():
        process.kill();process.join()
        result=json.loads(destination.read_text()) if destination.exists() else {
            "sample_id":name,"native_valid_solid":native,"pointcloud_ok":False,"complexity":None}
        result["complexity_error" if result["pointcloud_ok"] else "pointcloud_error"]="MetricTimeout: whole-object 180s"
        write_json(destination,result)
    elif not destination.exists():
        write_json(destination,{"sample_id":name,"native_valid_solid":native,
            "pointcloud_ok":False,"complexity":None,"pointcloud_error":f"MetricWorkerExit: {process.exitcode}"})
    return json.loads(destination.read_text())


def feature_records(group):
    return [json.loads(p.read_text()) for p in sorted((ROOT/"features"/group).glob("*.json"))]


def clouds(group):
    import numpy as np
    records=[row for row in feature_records(group) if row["pointcloud_ok"]]
    return records,np.stack([np.load(ROOT/"features"/group/f"{row['sample_id']}.npy") for row in records])


def cd_shard(args):
    import numpy as np
    import torch
    from src.brepnet.eval.metrics.point_cloud_set import _pairwise_cd
    torch.set_num_threads(1)
    records,points=clouds(args.group)
    reference_records,reference=clouds("reference")
    indices=np.arange(len(points))[args.rank::args.shards]
    with torch.inference_mode():
        matrix=_pairwise_cd(torch.from_numpy(points[indices]).cuda(),torch.from_numpy(reference).cuda(),32).cpu().numpy()
    folder=ROOT/"distances"/args.group
    folder.mkdir(parents=True,exist_ok=True)
    np.savez(folder/f"shard{args.rank:02d}.npz",indices=indices,distances=matrix,
             sample_ids=np.array([row["sample_id"] for row in records]),
             reference_ids=np.array([row["sample_id"] for row in reference_records]))


def set_statistics(matrix,points,reference):
    import numpy as np
    from src.brepnet.eval.metrics.point_cloud_set import jsd_between_point_cloud_sets
    if len(points)==0:
        return {"n":0,"MMD-CD":None,"COV-CD":None,"JSD":None}
    return {"n":len(points),"reference_n":len(reference),
            "MMD-CD":float(matrix.min(axis=0).mean()),
            "COV-CD":float(len(np.unique(matrix.argmin(axis=1)))/matrix.shape[1]),
            "JSD":jsd_between_point_cloud_sets(points,reference,in_unit_sphere=False)}


def summarize_sets():
    import numpy as np
    all_data={}
    _,reference=clouds("reference")
    for group in GROUPS:
        records,points=clouds(group)
        matrix=np.empty((len(points),len(reference)),dtype=np.float32)
        shards=sorted((ROOT/"distances"/group).glob("shard*.npz"))
        coverage=[]
        for path in shards:
            shard=np.load(path)
            matrix[shard["indices"]]=shard["distances"]
            coverage.extend(shard["indices"].tolist())
        assert sorted(coverage)==list(range(len(points)))
        mask=np.array([row["native_valid_solid"] is True for row in records])
        all_data[group]=(records,points,matrix,mask)
    equal_n=min(int(data[3].sum()) for data in all_data.values())
    equal_all_n=min(len(data[1]) for data in all_data.values())
    report={"equal_valid_meshable_n":equal_n,"equal_all_meshable_n":equal_all_n,"reference_n":len(reference),"methods":{}}
    for group,(records,points,matrix,mask) in all_data.items():
        indices=np.where(mask)[0]
        chosen=np.random.default_rng(20261019).permutation(indices)[:equal_n]
        all_chosen=np.random.default_rng(20261019).permutation(len(points))[:equal_all_n]
        report["methods"][group]={
            "all_meshable":set_statistics(matrix,points,reference),
            "native_valid_meshable":set_statistics(matrix[mask],points[mask],reference),
            "equal_valid_meshable":set_statistics(matrix[chosen],points[chosen],reference),
            "equal_all_meshable":set_statistics(matrix[all_chosen],points[all_chosen],reference),
            "equal_all_subset_ids":[records[index]["sample_id"] for index in all_chosen],
            "equal_subset_ids":[records[index]["sample_id"] for index in chosen],
            "pointcloud_failures":[row for row in feature_records(group) if not row["pointcloud_ok"]],
        }
    write_json(ROOT/"report/set_metrics.json",report)


def graph_signature(graph):
    """同构所需的节点几何+degree多重集；仅用于排除不可能匹配。"""
    labels=[]
    for node in graph:
        geometry=graph.nodes[node]["shape_geometry"]
        labels.append((graph.degree[node],hashlib.sha256(geometry.tobytes()).hexdigest()))
    return (graph.number_of_nodes(),graph.number_of_edges(),tuple(sorted(labels)))


def graph_job(path):
    from src.brepnet.eval.metrics.uniqueness import load_data_from_npz,build_graph
    faces,adjacency=load_data_from_npz(path)
    graph=build_graph(faces,adjacency,n_bit=4)
    return str(path),graph_signature(graph)


def graphs(index_only=False):
    import pickle
    from src.brepnet.eval.metrics.uniqueness import load_data_from_npz,build_graph,is_graph_identical
    index_path=ROOT/"graph_index/train.pkl"
    if index_path.exists():
        index=pickle.loads(index_path.read_bytes())
    else:
        names=(REPO/"src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt").read_text().splitlines()
        index=defaultdict(list)
        with ProcessPoolExecutor(max_workers=16) as pool:
            for path,signature in pool.map(graph_job,[DATA/name/"data.npz" for name in names],chunksize=32):
                index[signature].append(path)
        index_path.parent.mkdir(parents=True,exist_ok=True)
        index_path.write_bytes(pickle.dumps(dict(index)))
    if index_only:
        print('Training graph index:',sum(len(v) for v in index.values()),flush=True)
        return
    report={"n_bit":4,"training_reference_n":sum(len(v) for v in index.values()),"methods":{}}
    for group,root in GROUPS.items():
        representatives=defaultdict(list)
        rows=[]
        for number in range(1000):
            name=f"uncond_{number:06d}"
            path=root/"raw_predictions"/name/"data.npz"
            inference_path=root/"metadata/inference/samples"/f"{name}.json"
            inference=json.loads(inference_path.read_text()) if inference_path.exists() else {}
            if not path.exists() or inference.get("status") != "ok":
                rows.append({"sample_id":name,"graph_ok":False});continue
            faces,adjacency=load_data_from_npz(path)
            graph=build_graph(faces,adjacency,n_bit=4)
            signature=graph_signature(graph)
            duplicate=next((old_name for old_name,old in representatives[signature] if is_graph_identical(graph,old)),None)
            if duplicate is None:
                representatives[signature].append((name,graph))
            identical_train=None
            for candidate in index.get(signature,[]):
                other_faces,other_adjacency=load_data_from_npz(candidate)
                if is_graph_identical(graph,build_graph(other_faces,other_adjacency,n_bit=4)):
                    identical_train=Path(candidate).parent.name;break
            post_path=root/"metadata/post/samples"/f"{name}.json"
            native=json.loads(post_path.read_text()).get("native_valid_solid") if post_path.exists() else None
            rows.append({"sample_id":name,"graph_ok":True,"duplicate_of":duplicate,"identical_train":identical_train,"native_valid_solid":native})
        good=[row for row in rows if row["graph_ok"]]
        valid=[row for row in good if row["native_valid_solid"] is True]
        report["methods"][group]={"n":len(good),"unique_count":sum(row["duplicate_of"] is None for row in good),
            "Unique":sum(row["duplicate_of"] is None for row in good)/len(good),
            "Novel":sum(row["identical_train"] is None for row in good)/len(good),
            "native_valid_n":len(valid),
            "native_valid_Unique":len({row["duplicate_of"] or row["sample_id"] for row in valid})/len(valid) if valid else None,
            "native_valid_Novel":sum(row["identical_train"] is None for row in valid)/len(valid) if valid else None,"rows":rows}
        print(group,report["methods"][group]["Unique"],report["methods"][group]["Novel"],flush=True)
    write_json(ROOT/"report/graph_metrics.json",report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=["prepare","cd","sets","graphs","index"],required=True)
    parser.add_argument("--group",choices=["reference",*GROUPS])
    parser.add_argument("--rank",type=int,default=0)
    parser.add_argument("--shards",type=int,default=1)
    args=parser.parse_args()
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name]="1"
    if args.phase=="prepare":prepare_points(args.group)
    elif args.phase=="cd":cd_shard(args)
    elif args.phase=="sets":summarize_sets()
    elif args.phase in {"graphs","index"}:graphs(index_only=args.phase=="index")


if __name__=="__main__":
    main()
