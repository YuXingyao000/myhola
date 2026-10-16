"""Task-local scheduler; frozen scientific runners and production algorithms remain untouched."""
from pathlib import Path
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time

import argparse
import fcntl

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--group',choices=('last','resume_best'),required=True)
parser.add_argument('--freeze-sources',action='store_true',help='Freeze source inventory only; does not start diagnostics')
parser.add_argument('--resume',action='store_true')
args=parser.parse_args()
BASE_ROOT=Path(__file__).resolve().parent
REPO=BASE_ROOT.parents[2]
sys.path.insert(0,str(REPO))
ROOT=BASE_ROOT/args.group
METHOD='encoder_edge_value'
CHECKPOINTS={'last':'040c05a45dd7edae825f2e6dd8e2cb3489a418146f9a7b2122cc54d38070618c',
             'resume_best':'e1e579e7bc6bdf64b3c078d6b5414e33630222de989b180ca8f3eb22b59c6ccf'}
META=ROOT/'metadata/orchestration'
META.mkdir(parents=True,exist_ok=True)
lock=(META/'runner.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
PYTHON='/mnt/d/anaconda3/envs/torch/bin/python'
BASELINE=REPO/'checkpoints/intersection_refinement/curve_conditioning/failure_census_20261014'
EVAL=BASE_ROOT.parent/'evaluation_20261016'/args.group
ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
if args.freeze_sources:
    target=ROOT/'metadata/source_freeze.json'
    if target.exists():raise FileExistsError('Existing source freeze must not be overwritten')
    previous=json.loads((REPO/'checkpoints/encoder_edge_value/failure_census_20261015/metadata/source_freeze.json').read_text())
    changed=[name for name,digest in previous['scientific_sources'].items() if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=digest]
    if changed:raise RuntimeError(f'Public scientific source differs from audited prior run: {changed}')
    names=sorted(set(previous['scientific_sources'])|{'tools/paired_failure_report.py','tools/vae_compare_metrics.py'})
    plan=ROOT/'PLAN.md'
    frozen={'schema':'encoder_census_source_freeze_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'baseline_method':'conditioned','candidate_method':METHOD,'input_contract':'same_geometry_source',
            'snapshot_group':args.group,'checkpoint_sha256':CHECKPOINTS[args.group],
            'scientific_sources':{name:hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in names},
            'execution_plan':{'path':str(plan),'sha256':hashlib.sha256(plan.read_bytes()).hexdigest()},
            'validation':{'prior_tests':['src/brepnet/test/test_paired_failure_census.py','src/brepnet/test/test_failure_census_contract.py'],'prior_passed':35,'current_shared_sources_sha_identical':True,'tests_rerun_this_task':False},
            'parent_execution_plan':{'path':str(BASE_ROOT/'PLAN.md'),'sha256':hashlib.sha256((BASE_ROOT/'PLAN.md').read_bytes()).hexdigest()},
            'evaluation_plan':{'path':str(EVAL.parent/'PLAN.md'),'sha256':hashlib.sha256((EVAL.parent/'PLAN.md').read_bytes()).hexdigest()},
            'note':'Task-local orchestration only. All public scientific tools, production algorithms and measurement formulas SHA-identical to prior audited run.'}
    target.write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'source_freeze':str(target),'candidate_method':METHOD,'source_count':len(names)}))
    sys.exit(0)
if (META/'state.json').exists() and not args.resume:
    raise FileExistsError('Existing orchestration state requires --resume')
STATE={'schema':'paired_census_orchestration_v1','baseline_method':'conditioned','candidate_method':METHOD,
       'snapshot_group':args.group,
       'input_contract':'same_geometry_source','started_at_utc':datetime.now(timezone.utc).isoformat(),
       'status':'checking_post','stages':{}}

def verify_frozen_sources():
    frozen=json.loads((ROOT/'metadata/source_freeze.json').read_text())
    changed=[name for name,digest in frozen['scientific_sources'].items()
             if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=digest]
    if changed:raise RuntimeError(f'Frozen scientific sources changed: {changed}')
    for key in ('execution_plan','parent_execution_plan','evaluation_plan'):
        record=frozen[key]
        if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()!=record['sha256']:
            raise RuntimeError(f'Frozen execution plan changed: {key}')
    if frozen['checkpoint_sha256']!=CHECKPOINTS[args.group]:raise RuntimeError('Frozen checkpoint differs')

def emit(event,**data):
    row={'utc':datetime.now(timezone.utc).isoformat(),'event':event,**data}
    with (META/'events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)
    path=META/'state.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(STATE,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def start(name,argv):
    verify_frozen_sources()
    output=META/(name+'.log')
    handle=output.open('ab',buffering=0)
    process=subprocess.Popen(argv,cwd=REPO,env=ENV,stdout=handle,stderr=subprocess.STDOUT)
    STATE['stages'][name]={'status':'running','pid':process.pid,'argv':argv,'log':str(output),'started_at_utc':datetime.now(timezone.utc).isoformat()}
    emit('start',name=name,pid=process.pid,argv=argv,log=str(output))
    return process,handle

def finish(name,pair):
    process,handle=pair;code=process.wait();handle.close()
    STATE['stages'][name].update(status='complete' if code==0 else 'error',exit_code=code,finished_at_utc=datetime.now(timezone.utc).isoformat())
    emit('complete' if code==0 else 'error',name=name,exit_code=code)
    if code:raise RuntimeError(f'{name} failed with exit {code}; see {META/(name+".log")}')

def command(name,argv):finish(name,start(name,argv))

def runner(module,manifest,root,extra=()):
    key='--output-root' if module.endswith('provenance') else '--root'
    args=[PYTHON,'-m',module,'--manifest',str(manifest),key,str(root),'--workers','8']
    if not module.endswith('upstream'):args+=['--timeout','180']
    return args+list(extra)+['--resume']

try:
    verify_frozen_sources()
    emit('checking_post')
    if not all((EVAL/'post'/mode/'summary.json').is_file() and (EVAL/'post'/mode/'samples.csv').is_file() for mode in ('fit_only','optimized')):
        raise RuntimeError('Both post modes must finish before starting census')
    for mode in ('fit_only','optimized'):
        rows=list(csv.DictReader((EVAL/'post'/mode/'samples.csv').open()))
        if len(rows)!=4834 or any(r['status'] in ('running','not_run','queued') for r in rows):
            raise RuntimeError(f'Post table incomplete: {mode}/{len(rows)}')
    raw_rows=list(csv.DictReader((EVAL/'raw_metrics/posterior_mean/per_object.csv').open()))
    candidate_rows=[row for row in raw_rows if row['method']==METHOD]
    if len(candidate_rows)!=2417 or any(row.get('encoder_checkpoint_sha256')!=CHECKPOINTS[args.group] for row in candidate_rows):
        raise RuntimeError('Candidate checkpoint identity/count differs from frozen selection')
    STATE['status']='freezing_manifest';emit('post_tables_complete')
    if not (ROOT/'manifest.json').exists():
        command('manifest',[PYTHON,'-m','tools.paired_failure_manifest','--comparison-root',str(EVAL),
        '--reference-root','/mnt/d/python_curve_eval/checkpoints/intersection_refinement/curve_conditioning/evaluation_20261014',
        '--baseline-census-root',str(BASELINE),'--data-root','/mnt/d/data/deepcad_v7',
        '--split-list',str(REPO/'checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt'),
        '--output-root',str(ROOT),'--candidate-method',METHOD,'--baseline-method','conditioned','--input-contract','same_geometry_source'])
    elif not args.resume:
        raise FileExistsError('Frozen manifest exists; use explicit --resume')
    else:
        from tools.paired_failure_report import audit_source_preservation
        audit_source_preservation(ROOT)
    manifest=json.loads((ROOT/'manifest.json').read_text());controls=json.loads((ROOT/'baseline_control_manifest.json').read_text())
    if manifest.get('candidate_method')!=METHOD or manifest.get('input_contract')!='same_geometry_source' or Path(manifest['comparison_root']).resolve()!=EVAL.resolve():
        raise RuntimeError('Frozen manifest identity/input contract/comparison root differs')
    candidate_ids=[]
    for cohort in ('both_failed','new_failure','rescued'):
        ids=manifest['cohorts'][cohort]
        if ids:candidate_ids.append(ids[0])
    control_ids=[c['id'] for c in controls['cases'][:2]]
    STATE.update(status='real_smoke',population=manifest['population'],cohort_counts={k:len(v) for k,v in manifest['cohorts'].items()},
                 optimization_counterfactual_count=len(manifest['optimization_regression_ids']),candidate_smoke_ids=candidate_ids,baseline_smoke_ids=control_ids)
    emit('population_frozen',population=manifest['population'],cohort_counts=STATE['cohort_counts'],optimization_counterfactual_count=STATE['optimization_counterfactual_count'])
    smoke={}
    if candidate_ids:smoke['candidate_smoke']=start('candidate_smoke',runner('tools.legacy_failure_provenance',ROOT/'manifest.json',ROOT,['--only-ids',*candidate_ids]))
    if control_ids:smoke['baseline_smoke']=start('baseline_smoke',runner('tools.legacy_failure_provenance',ROOT/'baseline_control_manifest.json',ROOT/'baseline_controls',['--only-ids',*control_ids]))
    for name,pair in smoke.items():finish(name,pair)
    checks=[]
    for side,ids,path in [('candidate',candidate_ids,ROOT),('baseline',control_ids,ROOT/'baseline_controls')]:
        for sid in ids:
            trace=json.loads((path/'traces'/f'{sid}.json').read_text())
            check={'side':side,'id':sid,'status':trace.get('status'),'same_outcome':trace.get('comparison',{}).get('same_recorded_outcome'),
                   'step_data_equal':trace.get('step_data_comparison',{}).get('equal'),'instrumentation_errors':trace.get('instrumentation_errors',[])}
            checks.append(check)
            if check['status']!='ok' or check['same_outcome'] is not True or check['step_data_equal'] is not True or check['instrumentation_errors']:
                raise RuntimeError(f'Smoke replay mismatch: {check}')
    (META/'real_smoke.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    emit('real_smoke_passed',checks=checks)
    tasks={}
    if manifest['cases']:
        tasks['candidate_trace']=(runner('tools.legacy_failure_provenance',ROOT/'manifest.json',ROOT),[])
        tasks['candidate_saved_step']=(runner('tools.legacy_failure_saved_step',ROOT/'manifest.json',ROOT),[])
        tasks['candidate_upstream']=(runner('tools.legacy_failure_upstream',ROOT/'manifest.json',ROOT),['candidate_trace'])
    if controls['cases']:
        tasks['baseline_trace']=(runner('tools.legacy_failure_provenance',ROOT/'baseline_control_manifest.json',ROOT/'baseline_controls'),[])
        tasks['baseline_saved_step']=(runner('tools.legacy_failure_saved_step',ROOT/'baseline_control_manifest.json',ROOT/'baseline_controls'),[])
        tasks['baseline_upstream']=(runner('tools.legacy_failure_upstream',ROOT/'baseline_control_manifest.json',ROOT/'baseline_controls'),['baseline_trace'])
    fit_ids=manifest['optimization_regression_ids']
    if fit_ids:
        tasks['fit_trace']=(runner('tools.legacy_failure_provenance',ROOT/'manifest.json',ROOT/'fit_only_counterfactual',['--post-mode','fit_only','--only-ids',*fit_ids]),[])
        tasks['fit_upstream']=(runner('tools.legacy_failure_upstream',ROOT/'manifest.json',ROOT/'fit_only_counterfactual',['--only-ids',*fit_ids]),['fit_trace'])
    priority=['candidate_trace','baseline_trace','candidate_saved_step','baseline_saved_step','fit_trace','candidate_upstream','baseline_upstream','fit_upstream']
    pending=[name for name in priority if name in tasks];running={};done=set();STATE['status']='full_diagnostics';emit('full_diagnostics_start')
    last_progress=0
    while pending or running:
        for name in list(pending):
            argv,deps=tasks[name]
            if len(running)<4 and all(d in done for d in deps):
                running[name]=start(name,argv);pending.remove(name)
        for name,pair in list(running.items()):
            if pair[0].poll() is not None:
                finish(name,pair);done.add(name);del running[name]
        if time.monotonic()-last_progress>60:
            progress={}
            for side,path in [('candidate',ROOT),('baseline',ROOT/'baseline_controls'),('fit',ROOT/'fit_only_counterfactual')]:
                for directory in ('traces','upstream','saved_step'):
                    files=list((path/directory).glob('*.json'))
                    counts={}
                    for p in files:
                        try:status=json.loads(p.read_text()).get('status','unknown')
                        except (OSError,json.JSONDecodeError):status='unreadable_during_write'
                        counts[status]=counts.get(status,0)+1
                    if files:progress[f'{side}_{directory}']=counts
            STATE['progress']=progress;emit('progress',running=list(running),pending=pending,counts=progress);last_progress=time.monotonic()
        time.sleep(5)
    STATE['status']='generating_report';emit('all_diagnostics_terminal')
    command('report',[PYTHON,'-m','tools.paired_failure_report','--census-root',str(ROOT),'--baseline-census-root',str(BASELINE)]+(['--overwrite'] if args.resume and (ROOT/'summary.json').exists() else []))
    STATE['status']='report_complete_browser_pending';emit('report_complete_browser_pending',report=str(ROOT/'REPORT.md'))
except Exception as exc:
    STATE.update(status='error',error=f'{type(exc).__name__}: {exc}');emit('orchestration_error',error=STATE['error'])
    raise
