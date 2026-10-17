"""CPU-only contracts for migration entrypoints; no CAD corpus or model weights."""
import json
import sys
import types
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

from src.brepnet.data import extract_brep
from src.brepnet.data.datagen import run_blender, run_flux, run_pack
from src.brepnet.data.datagen.prompts import build_prompt, decode_prompt_config
from src.brepnet.data.datagen.flux_scripts import generate_cube24_dynamic as cube_worker
from src.brepnet.data.datagen.external.generation import generate as single_worker


@pytest.mark.parametrize("prompt_mode", ["fixed", "dynamic"])
@pytest.mark.parametrize("mode", ["single-view", "cube24"])
def test_launcher_commands_match_worker_cli(tmp_path, monkeypatch, prompt_mode, mode):
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="datagen", overrides=[f"datagen/prompt={prompt_mode}"])
    ids = tmp_path / "ids.txt"
    ids.write_text("00000001\n")
    cfg.mode = mode
    cfg.paths.train_list = str(ids)
    cfg.paths.logs_root = str(tmp_path / "logs")
    cfg.paths.flux_single_view_out = str(tmp_path / "single")
    cfg.paths.flux_cube24_out = str(tmp_path / "cube")
    cfg.flux.num_gpus = 1
    commands = []
    monkeypatch.setattr(run_flux, "DATAGEN_ROOT", tmp_path)
    monkeypatch.setattr(run_flux, "run_parallel_ranked", lambda jobs: commands.extend(jobs))
    run_flux.main.__wrapped__(cfg)
    argv = commands[0][1][3:]
    args = (single_worker.build_top_level_parser().parse_args(argv)
            if mode == "single-view" else cube_worker.parse_args(argv))
    prompt_cfg = decode_prompt_config(args.prompt_config_json)
    assert prompt_cfg.mode == prompt_mode
    assert args.pre_encode_text == (prompt_mode == "fixed")
    assert args.neg_prompt == cfg.prompt.negative
    assert "{material}" not in build_prompt(prompt_cfg, "00000001")


@pytest.mark.parametrize("prompt_mode", ["fixed", "dynamic"])
@pytest.mark.parametrize("mode", ["single-view", "cube24"])
def test_workers_use_forwarded_prompts(tmp_path, monkeypatch, prompt_mode, mode):
    policy = {
        "mode": prompt_mode, "pre_encode_text": prompt_mode == "fixed",
        "positive": "Fixed prompt unchanged", "negative": "Negative prompt unchanged",
        "base_positive": "A {material} part", "materials": ["titanium"],
        "finishes": ["matte"], "processes": ["milled"], "conditions": ["new"],
    }
    seen = []

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def __call__(self, image, seed=None):
            seen.append((self.config.pos_prompt, self.config.neg_prompt, self.config.pre_encode_text, seed))
            return Image.fromarray(image)

    fake_module = types.SimpleNamespace(Img2BrepPipeline=FakePipeline)
    monkeypatch.setitem(sys.modules, "generation.pipeline", fake_module)
    monkeypatch.setitem(sys.modules, "src.brepnet.data.datagen.external.generation.pipeline", fake_module)
    ids = tmp_path / "ids.txt"
    ids.write_text("00000001\n")
    relative = Path("00000001/0/0.png" if mode == "single-view" else "00000001/18.png")
    source = tmp_path / "input" / relative
    source.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), "white").save(source)
    argv = [
        "--render-root", str(tmp_path / "input"), "--output-root", str(tmp_path / "output"),
        "--model-list", str(ids), "--prompt-config-json", json.dumps(policy),
    ]
    if mode == "single-view":
        single_worker.main(argv + ["--material-count", "1"])
    else:
        cube_worker.main(argv)
    assert len(seen) == 1
    assert seen[0][:3] == (build_prompt(OmegaConf.create(policy), "00000001"), policy["negative"], prompt_mode == "fixed")
    assert isinstance(seen[0][3], int)
    assert (tmp_path / "output" / relative).is_file()


def test_dynamic_prompt_cannot_use_stale_embeddings():
    with pytest.raises(ValueError, match="pre_encode_text=false"):
        decode_prompt_config(json.dumps({"mode": "dynamic", "pre_encode_text": True}))


@pytest.mark.parametrize("mode", ["cube24", "single-view"])
def test_shared_config_preserves_blender_and_pack_paths(tmp_path, monkeypatch, mode):
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    ids = tmp_path / "ids.txt"
    ids.write_text("00000001\n")
    monkeypatch.setenv("HOLA_TEST_MODEL_ROOT", str(tmp_path / "models"))
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="datagen", overrides=[
            f"mode={mode}", f"paths.train_list={ids}",
            "paths.model_root=${oc.env:HOLA_TEST_MODEL_ROOT}",
            f"paths.cond_target={tmp_path / 'conditions'}",
            f"paths.blender_cube24_out={tmp_path / 'render_cube'}",
            f"paths.blender_single_view_out={tmp_path / 'render_single'}",
            f"paths.flux_cube24_out={tmp_path / 'flux_cube'}",
            f"paths.flux_single_view_out={tmp_path / 'flux_single'}",
            f"paths.logs_root={tmp_path / 'logs'}", "blender.num_gpus=1", "flux.pack_workers=2",
        ])
    renders = []
    packs = []
    monkeypatch.setattr(run_blender, "DATAGEN_ROOT", tmp_path)
    monkeypatch.setattr(run_blender, "run_parallel_ranked", lambda jobs: renders.extend(jobs))
    monkeypatch.setattr(run_pack.subprocess, "run", lambda cmd, **kwargs: packs.append(cmd))
    run_blender.main.__wrapped__(cfg)
    run_pack.main.__wrapped__(cfg)
    render = renders[0][1]
    assert render[render.index("--model-root") + 1] == str(tmp_path / "models")
    pack = packs[0]
    assert pack[pack.index("--target-root") + 1] == str(tmp_path / "conditions")
    assert pack[pack.index("--model-list") + 1] == str(ids)
    assert pack[pack.index("--num-workers") + 1] == "2"
    suffix = "cube" if mode == "cube24" else "single"
    assert pack[pack.index("--flux-root") + 1] == str(tmp_path / f"flux_{suffix}")


def extraction_args(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    ids = tmp_path / "ids.txt"
    ids.write_text("00000002\n00000001\n00000002\n")
    return ["--data-root", str(source), "--output-root", str(tmp_path / "output"), "--model-list", str(ids)]


def test_extraction_dry_run_preserves_source_and_does_not_import_occ(tmp_path):
    argv = extraction_args(tmp_path)
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("00000001\n")
    args = extract_brep.parse_args(argv + ["--exclude-list", str(excluded)])
    ids, manifest = extract_brep.prepare_plan(args)
    assert ids == ["00000002"]
    assert manifest["source_count"] == 2
    assert manifest["excluded_count"] == 1
    extract_brep.main(argv + ["--dry-run"])
    assert not (tmp_path / "output").exists()
    assert "src.brepnet.data.extract_brep_from_step_half_structure" not in sys.modules


def test_extraction_preserves_existing_outputs(tmp_path):
    argv = extraction_args(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    existing = output / "existing.txt"
    existing.write_text("preserve")
    with pytest.raises(FileExistsError, match="preserved"):
        extract_brep.prepare_plan(extract_brep.parse_args(argv))
    assert existing.read_text() == "preserve"


def test_extraction_rejects_overlapping_paths(tmp_path):
    args = extract_brep.parse_args(extraction_args(tmp_path))
    args.output_root = args.data_root / "derived"
    with pytest.raises(ValueError, match="overlap"):
        extract_brep.prepare_plan(args)
