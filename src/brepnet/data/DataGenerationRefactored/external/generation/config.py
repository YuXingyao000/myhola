import torch
import random
from pathlib import Path
from dataclasses import dataclass

SKETCH_POS_PROMPT = (
    "technical mechanical engineering sketch, hand-drawn pencil lineart on paper, subtle line wobble, "
    "pressure-sensitive strokes, varying line weight, slightly darker outer contour, faint construction lines, "
    "clean drafting style, light paper grain, minimal shading, precise geometry, engineer sketch, CAD model sketch"
)

# Default when conditioning on Blender studio renders (DataGeneration step 04).
PHOTOREAL_POS_PROMPT = (
    "Generate a natural-looking photo placing this CAD part on a desk, as if shot on iPhone. Make it look realistic, and ensure the material and surface texture match a real-world aluminum alloy CAD part that has been used, including believable wear and aging."
)

MATERIAL = [
    "aluminum",
    "stainless steel",
    "titanium",
    "carbon steel",
]

FINISH = [
    "brushed finish",
    "matte finish",
    "polished finish",
    "satin finish",
]

PROCESS = [
    "CNC-machined",
    "precision-milled",
    "machined",
]

CONDITION = [
    "with subtle machining marks",
    "with minor surface imperfections",
    "clean and new",
    "with fine tool marks",
]


def build_metal_description(seed: int) -> str:
    rng = random.Random(seed)   # <-- deterministic

    return ", ".join([
        rng.choice(MATERIAL),
        rng.choice(FINISH),
        rng.choice(PROCESS),
        rng.choice(CONDITION),
    ])


def build_prompt(base_prompt: str, seed: int) -> str:
    metal_desc = build_metal_description(seed)
    return base_prompt.replace("metallic materials", metal_desc)

@dataclass
class Img2BrepConfig:
    transformer_path: Path = "/mnt/d/model/Flux1_Kontext_dev_GGUF/flux1-kontext-dev-Q8_0.gguf"
    base_model_path: Path = "/mnt/d/model/Flux1_Kontext_dev"

    pos_prompt: str = PHOTOREAL_POS_PROMPT
    neg_prompt: str = "deformed geometry, distorted shape, incorrect proportions, warped structure, missing parts, extra parts, altered topology"
    
    pre_encode_text: bool = True
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype: torch.dtype = torch.bfloat16
    num_steps: int = 40
    guidance_scale: float = 3.5
    true_cfg_scale: float = 1.0
    seed: int = 0
