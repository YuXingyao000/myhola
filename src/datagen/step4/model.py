"""一次加载 FLUX，缓存场景提示词，再逐图编辑。

由外部设置 CUDA_VISIBLE_DEVICES；这里的 cuda:0 是该进程唯一可用的 GPU。
只负责模型采样，保护区的最终像素合成与验收由外围 workflow 完成。
"""

import gc
from pathlib import Path
from time import perf_counter

from PIL import Image
import torch
from diffusers import (
    FluxInpaintPipeline,
    FluxKontextPipeline,
    FluxKontextInpaintPipeline,
    FluxControlNetInpaintPipeline,
    FluxControlNetModel,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
)
from diffusers.models.model_loading_utils import load_state_dict


class FluxEnhancer:
    """本地 FLUX GGUF；提示词编码后可将采样模型常驻 GPU，逐图复用。"""

    def __init__(self, model_dir: Path, transformer_file: Path, prompt: str, config: dict):
        self.config = dict(config)
        pipeline_class = {
            "kontext": FluxKontextInpaintPipeline,
            "kontext_edit": FluxKontextPipeline,
            "flux_depth": FluxControlNetInpaintPipeline,
            "flux_control": FluxControlNetInpaintPipeline,
            "flux": FluxInpaintPipeline,
        }[self.config["pipeline"]]
        started = perf_counter()
        print(f"[FLUX] 加载 GGUF Transformer：{transformer_file}", flush=True)
        checkpoint = load_state_dict(str(transformer_file))
        transformer = FluxTransformer2DModel.from_single_file(
            checkpoint,
            config=str(model_dir),
            subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        del checkpoint
        print(f"[FLUX] Transformer 就绪，耗时 {perf_counter() - started:.1f}s", flush=True)

        started = perf_counter()
        print("[FLUX] 加载 VAE、文本编码器与分词器", flush=True)
        components = {}
        if self.config["pipeline"] in ("flux_depth", "flux_control"):
            components["controlnet"] = FluxControlNetModel.from_pretrained(
                self.config["controlnet_dir"], torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
        self.pipe = pipeline_class.from_pretrained(
            str(model_dir),
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            **components,
        )
        self.pipe.enable_model_cpu_offload(gpu_id=0)
        # 每张图只打印开始、完成两行，不在批量日志中展开逐步进度条。
        self.pipe.set_progress_bar_config(disable=True)
        print(f"[FLUX] Pipeline 就绪，耗时 {perf_counter() - started:.1f}s", flush=True)

        started = perf_counter()
        print("[FLUX] 编码并缓存本场景提示词", flush=True)
        with torch.inference_mode():
            self.prompt_embeds, self.pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.pipe._execution_device,
                max_sequence_length=self.config["max_sequence_length"],
            )
        torch.cuda.synchronize()
        print(f"[FLUX] 提示词缓存就绪，耗时 {perf_counter() - started:.1f}s", flush=True)

        if self.config.get("resident_model", False):
            # 提示词已经缓存，之后只需 Transformer 和 VAE。去掉 offload hooks，
            # 避免每张图都重新把约 12 GB 的 Transformer 搬入显卡。
            started = perf_counter()
            print("[FLUX] 释放文本编码器，将 Transformer 与 VAE 常驻显卡", flush=True)
            self.pipe.remove_all_hooks()
            self.pipe.register_modules(text_encoder=None, text_encoder_2=None)
            # Accelerate 恢复的 bound forward 会留下引用环；仅设为 None
            # 仍可能保留约 9 GiB 的 T5 权重，需在常驻模型前明确回收一次。
            gc.collect()
            torch.cuda.empty_cache()
            self.pipe.to("cuda:0")
            torch.cuda.synchronize()
            print(f"[FLUX] 常驻模式就绪，耗时 {perf_counter() - started:.1f}s", flush=True)

    @torch.inference_mode()
    def enhance(
        self, image: Image.Image, edit_mask: Image.Image, strength: float, seed: int,
        depth_image: Image.Image | None = None,
        step_callback=None,
        control_image: Image.Image | None = None,
    ) -> tuple[Image.Image, dict]:
        """白色 mask 区域可修改；标准 Kontext 整图编辑忽略 mask/strength。

        strength 会截取采样时间表，因此记录实际执行步数，而非仅记录配置步数。
        Kontext 将原图作为参考条件；普通 FLUX 从原图加噪后去噪。
        flux_depth 额外接收已转换为近白远黑的深度图，不能直接传入米制 Z。
        processing_size 控制模型内部尺寸；最终交给外围合成的图片始终是 512×512。
        step_callback 供独立实验逐步调整编辑 mask；默认不改变原采样流程。
        control_image 可传入边缘或深度条件；depth_image 保留旧调用兼容。
        """
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        print(f"[FLUX] 去噪开始：strength={strength:g}，seed={seed}", flush=True)
        size = self.config.get("processing_size", 512)
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        edit_mask = edit_mask.resize((size, size), Image.Resampling.NEAREST)
        # 标准 Kontext 从噪声生成、持续参考原图，没有 img2img 的 strength。
        # 其余两条管线从原图加噪，并逐步回填 mask 外的 latent。
        editing_options = {} if self.config["pipeline"] == "kontext_edit" else {
            "mask_image": edit_mask, "strength": strength,
        }
        # 只有 Kontext 默认自动放大；普通 FluxInpaintPipeline 没有这两个参数。
        resolution_options = (
            {"max_area": size**2, "_auto_resize": False}
            if self.config["pipeline"] in ("kontext", "kontext_edit") else {}
        )
        if self.config["pipeline"] in ("flux_depth", "flux_control"):
            condition = control_image if control_image is not None else depth_image
            editing_options.update(
                control_image=condition.resize((size, size), Image.Resampling.BILINEAR),
                controlnet_conditioning_scale=self.config["controlnet_conditioning_scale"],
                control_guidance_start=0.0,
                control_guidance_end=self.config.get("control_guidance_end", 1.0),
            )
        else:
            editing_options["true_cfg_scale"] = 1.0
        if step_callback is not None:
            editing_options.update(
                callback_on_step_end=step_callback,
                callback_on_step_end_tensor_inputs=["mask"],
            )
        pixels = self.pipe(
            image=image,
            prompt_embeds=self.prompt_embeds,
            pooled_prompt_embeds=self.pooled_prompt_embeds,
            num_inference_steps=self.config["num_inference_steps"],
            guidance_scale=self.config["guidance_scale"],
            height=size,
            width=size,
            max_sequence_length=self.config["max_sequence_length"],
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="pt",
            **resolution_options,
            **editing_options,
        ).images[0]
        # 先检查浮点输出，防止 NaN 在转成 PNG 时变成黑色而被误计为成功。
        if not torch.isfinite(pixels).all().item():
            raise ValueError("FLUX 输出包含非有限像素")
        pixels = pixels.mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        result = Image.fromarray(pixels)
        if size != 512:
            result = result.resize((512, 512), Image.Resampling.LANCZOS)
        torch.cuda.synchronize()
        metrics = {
            "actual_steps": self.pipe.num_timesteps,
            "elapsed_seconds": perf_counter() - started,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "processing_resolution": [size, size],
            "initial_sigma": float(self.pipe.scheduler.sigmas[self.pipe.scheduler.begin_index or 0]),
            "masked_sampling": self.config["pipeline"] != "kontext_edit",
        }
        print(
            f"[FLUX] 完成：{metrics['actual_steps']} 步，"
            f"{metrics['elapsed_seconds']:.1f}s，"
            f"显存峰值 {metrics['peak_memory_allocated_bytes'] / 1024**3:.2f} GiB",
            flush=True,
        )
        return result, metrics
