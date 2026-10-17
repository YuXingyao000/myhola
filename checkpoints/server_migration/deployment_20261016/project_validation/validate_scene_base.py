"""验证主机工作流和 Blender 入口，不调用 Codex、不构建研究场景。"""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

root = Path('/mnt/d/python')
out = Path('/mnt/d/migration/project-validation')
out.mkdir(exist_ok=True, parents=True)
result = {'python': sys.version, 'executed_interpreter': sys.executable, 'commands': []}
for name in ['numpy', 'Pillow', 'rich', 'trimesh']:
    result[name] = importlib.metadata.version(name)
commands = [
    [sys.executable, str(root / 'packages/cad_scene_factory' / (name + '.py')), '--help']
    for name in ('workflow', 'call_llm', 'preview', 'prepare_stl', 'assets')
]
commands += [[str(Path('/mnt/d/natural') / version / 'blender'), '--background', '--factory-startup',
              '--python-exit-code', '1', '--python-expr',
              'import bpy; print("BLENDER_RUNTIME", bpy.app.version_string); print("SCENE_COUNT", len(bpy.data.scenes))']
             for version in ('blender-4.2.0-linux-x64', 'blender-5.2.2-linux-x64')]
for index, command in enumerate(commands):
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (out / f'scene-command-{index}.log').write_text(completed.stdout)
    result['commands'].append({'command': command, 'returncode': completed.returncode})
    assert completed.returncode == 0, completed.stdout[-3000:]
# The older data-generation worktree keeps its standard-library-only modules.
worktree = Path('/root/.codex/worktrees/data-generation/python')
sys.path.insert(0, str(worktree / 'packages/cad_scene_factory/src'))
import cad_scene_factory.geometry
import cad_scene_factory.contracts
import cad_scene_factory.visibility
result['data_generation_worktree_core_imports'] = True
result['status'] = 'passed'
result['scope'] = '只检查帮助、纯 Python 核心导入和 Blender 后台启动；未运行 Codex/模型制作/实际渲染。主机 base 无 PythonOCC，STEP 回退使用 torch 环境。'
(out / 'scene-base.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
