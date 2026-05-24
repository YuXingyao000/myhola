---
name: brepnet-diffusion-model-maintenance
description: Maintain and extend the modular HoLa-BRep diffusion model without breaking latent padding, conditioning, alignment, or oracle topology self-attention behavior.
---

# BRepNet Diffusion Model Skill

Use this skill when modifying `src/brepnet/models/diffusion*.py`, changing diffusion configs, adding a new condition modality, changing face padding, or running oracle topology experiments.

The diffusion model is intentionally assembled from small modules. Keep the config protocol stable and avoid adding defensive runtime guesses. If a field is required, put it in the relevant config group and make the module read that field directly.

## Module Composition

Current source files:

- `diffusion.py`: top-level model assembly, training loss, inference loop.
- `diffusion_latents.py`: chooses cached latents or VAE-encoded latents.
- `diffusion_padding.py`: packs variable-length face latents into fixed `[B, F, C]` sequences.
- `diffusion_condition.py`: builds condition encoders from condition config.
- `diffusion_denoiser.py`: condition fusion, alignment loss, timestep embedding, transformer denoising, oracle topology self-attention mask.
- `vae.py`: autoencoder used to encode raw B-Rep faces or decode generated face latents.

High-level assembly:

```text
configs/model/diffusion.yaml
configs/condition/*.yaml
        |
        v
+-----------------------------------------------------------+
| Diffusion                                                 |
|-----------------------------------------------------------|
| DDPMScheduler                                             |
| AutoEncoder                                               |
| FacePadder                                                |
| LatentProvider                                            |
| ConditionEncoder                                          |
| CrossAttentionConditionFuser                              |
| BRepDenoiser                                              |
| OracleTopologySelfAttentionMask                           |
+-----------------------------------------------------------+
```

Expanded module graph:

```text
batch
 |
 |  cached latent path:
 |    cached_latent_stats [B,F,2C], face_mask [B,F]
 |
 |  raw latent path:
 |    face_points, edge_points, face_counts
 v
+-----------------+
| LatentProvider  |
+-----------------+
 |                      +----------------+
 | use_cached_latents   | AutoEncoder    |
 | -------------------> | encode/sample  |  only when use_cached_latents=false
 |                      +----------------+
 v
latent_sequence [B,F,C]
face_mask [B,F] or None
 |
 |                 batch conditions
 |                       |
 |                       v
 |              +------------------+
 |              | ConditionEncoder |
 |              +------------------+
 |                       |
 |                       v
 |              condition_tokens
 |
 | face_adj [B,F,F], timesteps [B]
 |              |
 |              v
 | +---------------------------------+
 | | OracleTopologySelfAttentionMask |
 | +---------------------------------+
 |              |
 |              v
 | self_attention_mask [B,F,F] or None
 |
 v
+-------------------------------------------------------------+
| BRepDenoiser                                                |
|-------------------------------------------------------------|
| noisy latent -> input_projection                            |
| clean latent -> input_projection -> alignment branch         |
| condition_tokens -> CrossAttentionConditionFuser             |
| timestep -> sinusoidal embedding -> time MLP                 |
| self_attention_mask -> TransformerEncoder src_mask           |
| output_projection                                            |
+-------------------------------------------------------------+
 |
 v
prediction [B,F,C], align_loss or None
```

## Training Forward Flow

`Diffusion.forward(batch)` is the training entry point.

```text
1. latent_batch = LatentProvider(batch)
      values:    latent_sequence [B,F,C]
      face_mask: validity mask for zero padding, or all-valid for random padding

2. timesteps ~ Uniform(0, num_train_timesteps)

3. noise = randn_like(latent_sequence)

4. noisy_latent_sequence = DDPMScheduler.add_noise(
      latent_sequence,
      noise,
      timesteps
   )

5. condition = ConditionEncoder(batch)

6. self_attention_mask = OracleTopologySelfAttentionMask(
      batch["face_adj"],
      timesteps
   )

7. prediction, align_loss = BRepDenoiser(
      noisy_latent_sequence,
      timesteps,
      condition,
      clean_latent_sequence=latent_sequence,
      self_attention_mask=self_attention_mask
   )

8. diffusion target:
      epsilon prediction: target = noise
      sample prediction:  target = latent_sequence

9. total_loss =
      diffusion_loss
    + align_loss, if enabled
    + validity_loss, only for zero padding
```

## Denoiser Flow

`BRepDenoiser.forward()` mirrors the historical diffusion model:

```text
noisy_latent_sequence
  -> input_projection
  -> cross-attention query projection
  -> cross-attention with condition tokens
  -> hidden conditioned on image/point/text

clean_latent_sequence
  -> input_projection
  -> same query projection
  -> CAD/image alignment projection
  -> CLIP-style symmetric InfoNCE loss

timesteps
  -> sincos_embedding
  -> time_embedding MLP

conditioned hidden + time embedding
  -> TransformerEncoder self-attention
       optionally with oracle topology self_attention_mask
  -> output_projection
  -> prediction
```

The alignment branch must use clean latent, not noisy latent. This preserves the old experiment semantics:

```text
diffusion branch: noisy CAD latent + image condition -> denoise target
alignment branch: clean CAD latent + image condition -> CLIP-style alignment loss
```

## Oracle Topology Self-Attention

The current config key is still `model.topology_bias` for compatibility, but the implementation is `OracleTopologySelfAttentionMask`.

This module only affects denoiser self-attention. It does not add a new model branch.

Input:

```text
face_adj:  [B,F,F], already aligned with padded face order
timesteps: [B]
```

Modes:

```text
mode: hard_mask
  self and GT-adjacent faces can attend
  non-adjacent faces are masked out
  returns bool tensor [B,F,F]

mode: soft_bias
  GT-adjacent faces receive an additive attention-logit boost
  non-adjacent faces remain attendable
  returns float tensor [B,F,F]
```

Hard mask semantics:

```text
allowed_attention = face_adj OR identity
self_attention_mask = NOT allowed_attention

PyTorch bool src_mask:
  True  means blocked
  False means allowed
```

Soft bias semantics:

```text
self_attention_mask = face_adj.float() * timestep_weight * scale

PyTorch float src_mask:
  values are added to attention logits before softmax
```

Treat `soft_bias` as an ablation with an explicit trick. Treat `hard_mask` as the cleaner oracle topology upper-bound experiment.

## Face Padding And Topology Alignment

The topology matrix must be constructed in the same padded face order as the latent sequence.

Cached latent path, currently used by diffusion training:

```text
Diffusion_dataset.__getitem__()
  latent_stats: [N,2C]
  face_adj:     [N,N]

  if padding=random:
      face_padding_indices: [F]
      cached_latent_stats = latent_stats[face_padding_indices]
      padded_face_adj     = face_adj[face_padding_indices][:, face_padding_indices]

  if padding=zero:
      cached_latent_stats[:N] = latent_stats
      padded_face_adj[:N,:N]  = face_adj
```

For a 7-face shape with `F=30`, random padding creates one length-30 index. That same index must be used for both `[30,2C]` latent stats and `[30,30]` adjacency.

Do not create `face_adj [F,F]` in a different module or with a second random sample.

Raw latent path, `use_cached_latents=false`, is not currently a complete stable protocol for diffusion topology experiments. To support it correctly:

```text
dataset returns raw B-Rep tensors and raw face_adj
AutoEncoder encodes raw faces -> unpadded face_latents
FacePadder creates face_padding_indices
the same face_padding_indices remap both:
  face_latents -> latent_sequence [B,F,C]
  raw face_adj -> padded face_adj [B,F,F]
```

Do not enable oracle topology with `use_cached_latents=false` until `LatentSequence` carries the padding indices or padded adjacency returned by the same packing step.

## Config Protocol

The top-level diffusion config owns model assembly:

```yaml
latent:
  dim: 32
  use_cached_latents: true
  use_mean: true

padding:
  type: random        # zero or random
  max_faces: 30
  valid_loss_weight: 0.01

noise:
  prediction_type: epsilon
  beta_schedule: squaredcos_cap_v2
  num_train_timesteps: 1000

condition_fuser:
  type: cross_attention
  condition_dim: 1024
  hidden_dim: 1024
  num_layers: 4
  alignment:
    enabled: true
    weight: 1.0
    temperature: 0.07
    projection_dim: 256

topology_bias:
  enabled: true
  mode: hard_mask     # hard_mask or soft_bias
  scale: 2.0          # only meaningful for soft_bias
```

The condition config owns condition modality:

```yaml
type: single_img
dataset_names: [single_img]
cached_features: false
output_dim: 1024
```

The dataset config owns data layout and must match model padding:

```yaml
load_topology: true
max_faces: ${model.padding.max_faces}
padding: ${model.padding.type}
condition_names: ${condition.dataset_names}
cached_condition: ${condition.cached_features}
```

## Extension Rules

When adding a new condition modality:

1. Add a condition config under `configs/condition/`.
2. Extend `build_condition_encoder()` in `diffusion_condition.py`.
3. Keep the encoder output shape compatible with `condition_fuser.condition_dim`.
4. Do not add modality-specific checks inside `Diffusion.forward()`.

When adding a new padding mode:

1. Implement it in `diffusion_padding.py`.
2. Make it return a `LatentSequence`.
3. If topology is enabled, ensure adjacency is remapped with the exact same face order.
4. Update dataset config defaults only after the data protocol is stable.

When changing topology experiments:

1. Keep topology injection inside denoiser self-attention.
2. Prefer explicit modes such as `hard_mask` and `soft_bias`.
3. Avoid generic strategy names unless there are real separate implementations.
4. Document whether the mask is bool hard blocking or float additive logits.

When changing latent source:

1. `use_cached_latents=true` expects dataset-provided `cached_latent_stats`.
2. `use_cached_latents=false` requires raw VAE input tensors.
3. Do not mix cached-latent dataset output with raw VAE encoding behavior.

## Minimal Mental Model

The diffusion model is:

```text
latent source
  -> fixed face sequence
  -> add noise
  -> condition fusion through cross-attention
  -> optional clean-latent alignment loss
  -> timestep-conditioned transformer denoiser
  -> optional oracle topology self-attention mask
  -> latent prediction
```

The most important invariant:

```text
latent_sequence order == face_adjacency row/column order
```

If this invariant breaks, oracle topology experiments are invalid even if tensor shapes still look correct.
