from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os,hashlib,json,time
ROOT=Path('/root/hola_eval/conditional_feature_on_20261029')
SOURCE=Path('/mnt/d/python_curve_eval/checkpoints/shared_curve_uv/evaluation_20261020/best')
IDS=Path('/mnt/d/python_uv_train/checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt').read_text().split()
def read(path):
 fd=os.open(path,os.O_RDONLY|os.O_NOATIME)
 with os.fdopen(fd,'rb') as stream:
  return stream.read()
def listing(path):
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOATIME)
 try:
  return os.listdir(fd)
 finally:
  os.close(fd)
def copy(job):
 source,target=job
 content=read(source)
 target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(content)
 digest=hashlib.sha256(content).hexdigest()
 assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
 return {'source':str(source),'staged':str(target),'sha256':digest,'bytes':len(content)}
jobs=[]
for sid in IDS:
 for relative in (f'raw_predictions/posterior_mean/shared_curve_uv/{sid}/data.npz',f'metadata/post/optimized/samples/shared_curve_uv/{sid}.json',f'post/optimized/shared_curve_uv/{sid}/recon_brep.step'):
  src=SOURCE/relative
  if src.is_file():
   jobs.append((src,ROOT/'baseline'/relative))
 jobs.append((Path('/mnt/d/data/deepcad_v7')/sid/'normalized_shape.step',ROOT/'gt'/sid/'normalized_shape.step'))
for relative in ('metadata/post/optimized/protocol.json','raw_metrics/posterior_mean/per_object.csv','raw_metrics/posterior_mean/summary.json'):
 jobs.append((SOURCE/relative,ROOT/'baseline'/relative))
for name in listing(SOURCE/'metadata/gt_step_samples'):
 if name.endswith('.npz'):
  jobs.append((SOURCE/'metadata/gt_step_samples'/name,ROOT/'baseline/metadata/gt_step_samples'/name))
started=time.monotonic(); records=[]
with ThreadPoolExecutor(max_workers=16) as pool:
 for i,row in enumerate(pool.map(copy,jobs)):
  records.append(row)
  if (i+1)%1000==0: print(i+1,len(jobs),time.monotonic()-started,flush=True)
(ROOT/'validation').mkdir(exist_ok=True)
with (ROOT/'validation/staging_manifest.jsonl').open('w') as stream:
 for row in records:stream.write(json.dumps(row)+'\n')
report=dict(passed=True,count=len(records),bytes=sum(r['bytes'] for r in records),seconds=time.monotonic()-started,
 noatime_source_reads=True,staged_gt_count=len(IDS),logical_data_root='/mnt/d/data/deepcad_v7',physical_data_root=str(ROOT/'gt'),
 baseline_logical_root=str(SOURCE),baseline_physical_root=str(ROOT/'baseline'))
(ROOT/'validation/staging.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
