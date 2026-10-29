from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os,stat,json,hashlib,time
SOURCE=Path('/mnt/d/anaconda3/envs/torch')
TARGET=Path('/root/hola_eval/conditional_feature_on_20261029/environment')
ROOT=TARGET.parent
files=[]
def collect(source,target):
 target.mkdir(parents=True,exist_ok=True)
 fd=os.open(source,os.O_RDONLY|os.O_DIRECTORY|os.O_NOATIME)
 try:
  for item in os.scandir(fd):
   src,dst=source/item.name,target/item.name
   if item.is_symlink():
    link_fd=os.open(src, os.O_PATH | os.O_NOFOLLOW | os.O_NOATIME)
    try: value=os.readlink('',dir_fd=link_fd)
    finally: os.close(link_fd)
    if not dst.is_symlink():dst.symlink_to(value)
   elif item.is_dir(follow_symlinks=False):collect(src,dst)
   elif item.is_file(follow_symlinks=False):files.append((src,dst,item.stat(follow_symlinks=False).st_mode))
 finally:os.close(fd)
def copy(job):
 src,dst,mode=job
 fd=os.open(src,os.O_RDONLY|os.O_NOATIME)
 with os.fdopen(fd,'rb') as stream:content=stream.read()
 dst.write_bytes(content);os.chmod(dst,stat.S_IMODE(mode))
 digest=hashlib.sha256(content).hexdigest()
 assert hashlib.sha256(dst.read_bytes()).hexdigest()==digest
 return dict(path=str(src.relative_to(SOURCE)),sha256=digest,bytes=len(content))
started=time.monotonic();collect(SOURCE,TARGET);print('files',len(files),flush=True)
with (ROOT/'validation/environment_manifest.jsonl').open('w') as log,ThreadPoolExecutor(max_workers=16) as pool:
 count=total=0
 for row in pool.map(copy,files):
  log.write(json.dumps(row)+'\n');count+=1;total+=row['bytes']
  if count%10000==0:print(count,time.monotonic()-started,flush=True)
report=dict(source=str(SOURCE),staged=str(TARGET),count=count,bytes=total,seconds=time.monotonic()-started,all_content_sha_equal=True,noatime_source_reads=True,environment_reinstalled=False)
(ROOT/'validation/environment_staging.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
