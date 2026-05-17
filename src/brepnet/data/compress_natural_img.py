import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import ray


@ray.remote
def process_folder(natural_folder: Path, sketch_root: Path, outdir: Path):

    natural_path = natural_folder / "natural.png"
    if not natural_path.exists():
        print(f"skip {natural_folder.name} (sketch.png missing)")
        return

    try:
        with Image.open(natural_path) as img:
            img = img.convert("RGB")
            img = img.resize((224, 224), Image.BICUBIC)

        natural_data = np.array(img, dtype=np.uint8)

        sketch_path = sketch_root / natural_folder.name / "sketch.png"
        if not sketch_path.exists():
            sketch_path = sketch_root / natural_folder.name / "natural.png"
            
        with Image.open(sketch_path) as img:
            img = img.convert("RGB")
            img = img.resize((224, 224), Image.BICUBIC)
        
        sketch_data = np.array(img, dtype=np.uint8)

        assert natural_data.shape == (224, 224, 3)
        assert sketch_data.shape == (224, 224, 3)

        out_folder = outdir / natural_folder.name
        out_folder.mkdir(parents=True, exist_ok=True)

        out_path = out_folder / "new_imgs.npz"
        
        np.savez_compressed(out_path, natural_img=natural_data, sketch_img=sketch_data)

        print(f"saved {out_path}")

    except Exception as e:
        print(f"error processing {natural_folder.name}: {e}")



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--natural_root", type=Path, required=True)
    parser.add_argument("--sketch_root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)

    args = parser.parse_args()
    

    args.outdir.mkdir(parents=True, exist_ok=True)

    # 启动ray
    ray.init()

    tasks = []

    for folder in sorted(args.natural_root.iterdir()):

        if not folder.is_dir():
            continue
        
        task = process_folder.remote(folder, Path(args.sketch_root), args.outdir)
        tasks.append(task)

    # 等待全部任务完成
    ray.get(tasks)


if __name__ == "__main__":
    main()