import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'packages/cad_scene_factory'))
import call_llm as m

import argparse
import tempfile
parser = argparse.ArgumentParser(description='Codex transport contracts; subprocess stubs and help-only checks, no model calls')
parser.add_argument('--output_dir', type=Path)
OUT = parser.parse_args().output_dir or Path(tempfile.mkdtemp(prefix='cad-scene-transport-'))
OUT.mkdir(parents=True, exist_ok=True)
SID = '0199a213-81c0-7800-8aa1-bbab2a035a53'
OTHER = '0199a213-81c0-7800-8aa1-bbab2a035a54'
results = []
request = m.prepare_request(m.DEVELOPER_INSTRUCTIONS, '测试输入', OUT/'template', asset_cache=OUT/'cache')
assert not Path(request['output_dir']).exists()
assert request['model'] == 'gpt-6-astra' and request['reasoning_effort'] == 'ultra'
assert request['prompt'] == '测试输入'
assert set(request['instruction_files']) == {'SCENE_GUIDE.md', 'ACCEPTANCE.md', 'ASSET_TOOLS.md', 'RUNTIME.md'}

def fake_runner(mode='success'):
    def run(command, *, input, cwd, env, text, encoding, stdout, stderr):
        assert command[:2] == ['codex','exec'] and '--last' not in command
        assert 'OPENAI_API_KEY' not in env and 'CODEX_API_KEY' not in env
        assert 'forced_login_method="chatgpt"' in command
        assert '--ignore-user-config' in command and '--ignore-rules' in command
        assert 'project_doc_max_bytes=0' in command and 'features.memories=false' in command
        assert '-o' in command and '--json' in command
        if mode == 'launch_error':
            raise FileNotFoundError('fake codex is unavailable')
        events=[]
        if mode != 'missing_id':
            events.append({'type':'thread.started','thread_id':OTHER if mode=='wrong_id' else SID})
        events.append({'type':'turn.started'})
        if mode == 'exit_error':
            events.append({'type':'turn.failed','error':{'message':'fixture failure'}})
        elif mode != 'incomplete':
            events.append({'type':'turn.completed','usage':{'input_tokens':12,'cached_input_tokens':5,'output_tokens':3}})
        stdout.write('\n'.join(json.dumps(e) for e in events)+'\n')
        if mode == 'malformed_json':
            stdout.write('broken event\n')
        stderr.write('fixture process log\n')
        if mode != 'missing_reply':
            Path(command[command.index('-o')+1]).write_text('fixture reply\n')
        if mode == 'wrapper':
            for filename in ('build_scene.py','build_material.py'):
                (Path(cwd)/filename).write_text('# fixture\n')
        return subprocess.CompletedProcess(command,7 if mode=='exit_error' else 0)
    return run

with patch.dict(os.environ, {'OPENAI_API_KEY':'fixture-never-used','CODEX_API_KEY':'fixture-never-used'}):
    for name, sid in [('new',None),('resume',SID)]:
        with patch.object(m.subprocess,'run',fake_runner()):
            r=m.run_codex_turn(request,'阶段输入',OUT/name,session_id=sid)
        assert r['session_id']==SID and r['reply']=='fixture reply' and r['usage']['input_tokens']==12
        assert r['status']=='completed' and json.loads((OUT/name/'result.json').read_text())==r
        assert (OUT/name/'prompt.txt').read_text()=='阶段输入'
        results.append(name+': pass')
    for mode in ('wrong_id','exit_error','launch_error','missing_reply','malformed_json','missing_id','incomplete'):
        try:
            with patch.object(m.subprocess,'run',fake_runner(mode)):
                m.run_codex_turn(request,'failure fixture',OUT/mode,session_id=SID)
        except Exception:
            result=json.loads((OUT/mode/'result.json').read_text())
            assert result['status']=='failed' and result['error']
            assert (OUT/mode/'prompt.txt').exists() and (OUT/mode/'events.jsonl').exists() and (OUT/mode/'stderr.log').exists()
        else:
            raise AssertionError(mode+' did not fail')
        results.append(mode+': pass')

with patch.object(m.subprocess,'run',side_effect=AssertionError('dry-run invoked process')):
    result=m.call_llm(m.DEVELOPER_INSTRUCTIONS,'dry-run',OUT/'dry_run',dry_run=True)
assert (OUT/'dry_run'/'request.json').exists() and '未启动' in result
results.append('dry_run: pass')

with patch.object(m.subprocess,'run',fake_runner('wrapper')):
    assert m.call_llm(m.DEVELOPER_INSTRUCTIONS,'generate',OUT/'wrapper')=='fixture reply'
assert (OUT/'wrapper'/'reply.txt').read_text().strip()=='fixture reply'
results.append('compatibility_wrapper: pass')

schema=OUT/'schema.json'
schema.write_text('{"type":"object","properties":{},"required":[],"additionalProperties":false}')
for label,sid in [('exec_parse',None),('resume_parse',SID)]:
    command=m._codex_command(request,OUT/'parse',sid,[OUT/'one.png',OUT/'two.png'],schema)
    assert command[-1]=='-' and (sid is None or command[-2]==sid)
    if sid:
        assert command.index('--cd') < command.index('resume') < command.index('--json')
    # --help terminates argument parsing before any auth/network/model execution.
    command=command[:command.index('--')]+['--help']
    p=subprocess.run(command,text=True,capture_output=True)
    (OUT/(label+'.log')).write_text(p.stdout+p.stderr)
    assert p.returncode==0,(p.returncode,p.stderr)
    results.append(label+': pass (help-only, no model)')

(OUT/'results.json').write_text(json.dumps(results,indent=2))
print('\n'.join(results))
