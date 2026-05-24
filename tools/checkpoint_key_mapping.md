# Diffusion Checkpoint Key Mapping

Use `tools/convert_diffusion_checkpoint.py` to rewrite legacy checkpoints into
the current modular `Diffusion` layout.

| Legacy prefix | Current prefix |
| --- | --- |
| `model.p_embed.` | `model.denoiser.input_projection.` |
| `model.net1.` | `model.denoiser.backbone.` |
| `model.fc_out.` | `model.denoiser.output_projection.` |
| `model.time_embed.` | `model.denoiser.time_embedding.` |
| `model.cross_attn_pre_proj.` | `model.denoiser.condition_fuser.query_projection.` |
| `model.cross_attn_add_cond.` | `model.denoiser.condition_fuser.decoder.` |
| `model.cross_attn_post_proj.` | `model.denoiser.condition_fuser.output_projection.` |
| `model.cad_align_proj.` | `model.denoiser.condition_fuser.cad_alignment_projection.` |
| `model.img_align_proj.` | `model.denoiser.condition_fuser.condition_alignment_projection.` |
| `model.cad_align_head.` | `model.denoiser.condition_fuser.cad_alignment_head.` |
| `model.img_align_head.` | `model.denoiser.condition_fuser.condition_alignment_head.` |
| `model.classifier.` | `model.face_padder.validity_head.` |
| `model.padding.mask_head.` | `model.face_padder.validity_head.` |
| `model.latent_codec.autoencoder.` | `model.autoencoder.` |
| `model.ae_model.` | `model.autoencoder.` |
| `model.condition_extractor.` | `model.condition_encoder.` |
| `model.img_model.` | `model.condition_encoder.image_encoder.img_model.` |
| `model.img_fc.` | `model.condition_encoder.image_encoder.projection.` |
| `model.camera_embedding.` | `model.condition_encoder.camera_embedding.` |
| `model.point_model.` | `model.condition_encoder.point_encoder.` |
| `model.txt_model.` | `model.condition_encoder.text_encoder.text_model.` |
| `model.txt_fc.` | `model.condition_encoder.text_encoder.projection.` |

Dropped prefixes: `model.strategy.`, `model._feature_domain_mapper.`,
`model.img_adapters.`, `model.cond_attn.`, and learned unconditional/SVR/MVR/
sketch/point/text embeddings.
