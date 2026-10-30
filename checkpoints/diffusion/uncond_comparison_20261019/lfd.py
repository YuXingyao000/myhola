"""调用现有LFD特征二进制及仓库的距离核，保存完整矩阵。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
sys.path.insert(0,str(REPO))
EXECUTABLE=ROOT/"metadata/lfd_vendor/lfd/Executable"


def encode(job):
    import numpy as np
    import trimesh
    from metrics import write_json
    group,row=job
    name=row["sample_id"]
    folder=ROOT/"features/lfd"/group/name
    folder.mkdir(parents=True,exist_ok=True)
    record={"sample_id":name,"ok":False}
    try:
        path=Path(row["mesh_source"])
        if path.suffix==".step":
            from checkpoints.diffusion.uncond_baseline_20261018.visualize import step_mesh
            from OCC.Core.Interface import Interface_Static
            Interface_Static.SetIVal("read.precision.mode",0)
            vertices,faces,_=step_mesh(path)
            mesh=trimesh.Trimesh(vertices=vertices,faces=faces)
        else:
            mesh=trimesh.load(path,force="mesh")
        scale=np.ptp(mesh.vertices,axis=0).max()
        trimesh.Trimesh(mesh.vertices/(scale+1e-6),mesh.faces).export(folder/"mesh.obj")
        assets=["3DAlignment","align10.txt","q8_table",*[f"12_{i}.obj" for i in range(10)]]
        for name in assets:
            target=folder/name
            if not target.exists():target.symlink_to(EXECUTABLE/name)
        env=dict(os.environ,DISPLAY=":93",LIBGL_ALWAYS_SOFTWARE="1",LP_NUM_THREADS="1")
        process=subprocess.run([str(folder/"3DAlignment"),str(folder/"mesh")],cwd=folder,
            env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
        (folder/"encoder.log").write_bytes(process.stdout+process.stderr)
        if process.returncode:
            raise RuntimeError(f"3DAlignment exit={process.returncode}: {process.stderr.decode(errors='replace')[:150]}")
        for suffix,size in [("art",3500),("fd",1000),("cir",100),("ecc",100)]:
            source=folder/f"all_q8_v1.8.{suffix}"
            if source.exists():source.rename(folder/f"mesh_q8_v1.8.{suffix}")
            assert (folder/f"mesh_q8_v1.8.{suffix}").stat().st_size==size
        record["ok"]=True
    except Exception as error:
        record["error"]=f"{type(error).__name__}: {error}"
    write_json(folder/"status.json",record)
    return record


def prepare(group):
    from metrics import feature_records
    jobs=[(group,row) for row in feature_records(group) if row["pointcloud_ok"] and
          not (ROOT/"features/lfd"/group/row["sample_id"]/"status.json").exists()]
    print(group,'LFD encode',len(jobs),flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        for index,row in enumerate(pool.map(encode,jobs)):
            if (index+1)%50==0:print(group,index+1,len(jobs),flush=True)


def load(group):
    import numpy as np
    import torch
    paths=[p.parent for p in sorted((ROOT/"features/lfd"/group).glob('*/status.json')) if json.loads(p.read_text())["ok"]]
    arrays=[]
    for suffix,shape in [("art",(10,10,35)),("fd",(10,10,10)),("cir",(10,10)),("ecc",(10,10))]:
        arrays.append(torch.from_numpy(np.stack([np.fromfile(p/f"mesh_q8_v1.8.{suffix}",dtype=np.uint8).reshape(shape) for p in paths])).int().cuda())
    return [p.name for p in paths],arrays


def distances(args):
    import numpy as np
    import torch
    from src.brepnet.eval.lfd.evaluation_scripts.lfd_all_compute.lfd import calculate_lfd_distance
    torch.set_num_threads(1)
    names,source=load(args.group)
    refs,target=load('reference')
    q8=torch.from_numpy(np.fromfile(EXECUTABLE/"q8_table",dtype=np.uint8).reshape(256,256)).int().cuda()
    align=torch.from_numpy(np.fromfile(EXECUTABLE/"align10.txt",dtype=np.uint8).reshape(60,20)).int().cuda()
    indices=np.arange(len(names))[args.rank::args.shards]
    matrix=np.empty((len(indices),len(refs)),dtype=np.int64)
    with torch.inference_mode():
        for index,source_index in enumerate(indices):
            for start in range(0,len(refs),32):
                end=min(start+32,len(refs));count=end-start
                src=[a[source_index:source_index+1].expand(count,*a.shape[1:]) for a in source]
                dst=[a[start:end] for a in target]
                matrix[index,start:end]=calculate_lfd_distance(q8,align,*src,*dst).cpu().numpy()
            if (index+1)%25==0:print(args.group,args.rank,index+1,len(indices),flush=True)
    folder=ROOT/'distances/lfd'/args.group
    folder.mkdir(parents=True,exist_ok=True)
    np.savez(folder/f'shard{args.rank:02d}.npz',indices=indices,distances=matrix,sample_ids=np.array(names),reference_ids=np.array(refs))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['prepare','distance'],required=True)
    parser.add_argument('--group',required=True)
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
    if args.phase=='prepare':prepare(args.group)
    else:distances(args)


if __name__=='__main__':main()
