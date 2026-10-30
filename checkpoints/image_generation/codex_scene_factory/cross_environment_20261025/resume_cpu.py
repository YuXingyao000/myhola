"""本批次的 CPU 恢复入口；保留原会话、原返修计数与所有已保存轮次。

现有 resume_room 会把未完成的 repair 再计为一次返修，因此这里先完成被中断的
那一次 repair，再交还原工作流。宿主源码和场景源码均不修改；request 的资源变更
单独备份、记录。提示词见相邻 CPU_RUNTIME.md。
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

HOST = Path('/mnt/d/python_room_generation/packages/cad_scene_factory')
sys.path.insert(0, str(HOST))
import room_workflow as workflow


def resume_cpu(template_dir: Path, cpu_offset: int) -> dict:
    # 限制子进程资源；agent、工具调用与后续正式渲染都会继承。
    available = sorted(os.sched_getaffinity(0))
    cpu_ids = available[cpu_offset:cpu_offset + 8]
    if len(cpu_ids) != 8:
        raise RuntimeError('需要八个可用逻辑 CPU')
    os.sched_setaffinity(0, cpu_ids)
    os.nice(10)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['OPENBLAS_NUM_THREADS'] = '8'

    with workflow._task_lock(template_dir):
        request_path = template_dir / 'request.json'
        request = workflow._read_json(request_path)
        state = workflow._read_json(template_dir / 'workflow_result.json')
        if state['status'] == 'running' or state['stage'] != 'repair':
            raise RuntimeError('此入口仅用于已退出的中断返修')
        workflow._check_host(request)
        repair = state['repairs_used']
        old_turn_dir = template_dir / 'workflow' / f'repair_{repair:02d}'
        old_turn = workflow._read_json(old_turn_dir / 'result.json')
        if old_turn['status'] != 'failed' or old_turn['returncode'] is None:
            raise RuntimeError('旧返修还没有终止记录')

        change_dir = template_dir / 'workflow' / 'cpu_transition_01'
        change_dir.mkdir(exist_ok=False)
        original = request_path.read_bytes()
        (change_dir / 'request.before.json').write_bytes(original)
        workflow._write_json(change_dir / 'state.before.json', state)
        instructions = (Path(__file__).parent / 'CPU_RUNTIME.md').read_text()
        request['runtime']['device'] = 'CPU'
        request['instruction_files']['CPU_RUNTIME.md'] = instructions
        request['developer_instructions'] = '\n\n'.join(request['instruction_files'].values())
        request['developer_instructions'] += '\n\n```json\n' + json.dumps(request['runtime'], ensure_ascii=False, indent=2) + '\n```'
        workflow._write_json(request_path, request)
        workflow._write_json(change_dir / 'request.after.json', request)
        workflow._write_json(change_dir / 'transition.json', {
            'at_utc': datetime.now(timezone.utc).isoformat(),
            'authorization': 'User selected CPU rendering, accepting slower completion.',
            'before_sha256': hashlib.sha256(original).hexdigest(),
            'after_sha256': workflow._sha256(request_path),
            'changes': ['runtime.device', 'instruction_files.CPU_RUNTIME.md', 'developer_instructions'],
            'cpu_ids': cpu_ids, 'nice': os.nice(0), 'cuda_visible_devices': '',
            'session_id': state['session_id'], 'repairs_used': repair,
            'max_repairs': request['max_repairs'], 'host_source_unchanged': True,
        })
        (change_dir / 'resume_cpu.py').write_bytes(Path(__file__).read_bytes())
        (change_dir / 'CPU_RUNTIME.md').write_text(instructions)

        state['codex_turns'].append(workflow._turn_summary(old_turn))
        turn_dir = template_dir / 'workflow' / f'repair_{repair:02d}_resume_cpu_01'
        state.update(status='running', stage='repair', active_turn_dir=str(turn_dir))
        state.pop('error', None)
        workflow._save_result(state)
        feedback = json.dumps(state['pending_feedback'], ensure_ascii=False, indent=2)
        try:
            turn = workflow.run_codex_turn(
                request, instructions + '\n\n' + workflow._prompt('repair') + feedback,
                turn_dir, session_id=state['session_id'],
            )
            state['codex_turns'].append(workflow._turn_summary(turn))
            state.pop('pending_feedback')
            workflow._save_result(state)
            return workflow._execute_locked(request, state, create=False)
        except Exception as error:
            result_file = turn_dir / 'result.json'
            if result_file.exists():
                turn = workflow._read_json(result_file)
                if not any(Path(item['turn_dir']).resolve() == turn_dir.resolve()
                           for item in state['codex_turns']):
                    state['codex_turns'].append(workflow._turn_summary(turn))
            state['error'] = f'{type(error).__name__}: {error}'
            return workflow._finish(state, 'incomplete', [state['error']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template_dir', type=Path, required=True)
    parser.add_argument('--cpu_offset', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(resume_cpu(args.template_dir, args.cpu_offset), ensure_ascii=False, indent=2))
