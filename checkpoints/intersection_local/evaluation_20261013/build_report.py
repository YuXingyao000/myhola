"""Oct13 presentation adapter: full-capacity labels and current failure links."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))
from tools import local_evaluate_report as renderer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adjacency_paragraph():
    with (ROOT / 'raw_metrics/posterior_mean/per_object.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    counts = {}
    for method in renderer.METHODS:
        group = [row for row in rows if row['method'] == method]
        assert len(group) == 2417
        counts[method] = sum(float(row['topology_fp']) == 0 and float(row['topology_fn']) == 0 for row in group)
    return (f"整件无向面邻接全部正确（FP=FN=0）：上一版 {counts['intersection']}/2417，本版 {counts['local']}/2417。"
            "这只检查面pair集合，不证明半边方向、环闭合或几何正确。\n\n")


def main():
    original = renderer.build_report
    def current_report(data):
        text = original(data)
        old_link = '../failure_census_20261012/REPORT.md'
        assert text.count(old_link) == 1
        text = text.replace(old_link, '../failure_census_20261013/REPORT.md')
        text = text.replace('本版局部注意力', '本版完整容量局部注意力')
        text = text.replace('# 局部注意力 Intersection 与上一版 refinement 对比', '# 完整容量局部注意力 Intersection 与上一版 refinement 对比')
        text = text.replace('逐件分布、中位数/P90/P95', adjacency_paragraph() + '逐件分布、中位数/P90/P95')
        return text
    renderer.build_report = current_report
    renderer.LABELS['local'] = '本版完整容量局部注意力'
    paths = renderer.generate_report(ROOT, Path('/root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf'))
    sources = [Path(__file__), REPO / 'tools/local_evaluate_report.py', REPO / 'tools/intersection_evaluate_report.py']
    record = {'artifacts': paths, 'artifact_sha256': {k: digest(Path(p)) for k, p in paths.items()},
              'source_sha256': {str(p): digest(p) for p in sources},
              'adaptations': ['Presentation only: full-capacity candidate label', 'Failure report link points to 20261013', 'Count full correct face adjacency from validated per-object CSV'],
              'validation': 'Shared generator enforces all 2417 unique paired IDs, matching metrics, terminal statuses and cohort conservation.'}
    with (ROOT / 'report_generation.json').open('x') as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
