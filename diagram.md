┌─────────────────────────────────────────────────────────────────┐
│ 1. BUILD MODULES (src/builders/build.py)                       │
├─────────────────────────────────────────────────────────────────┤
│ • build_unet(args) → UNet                                       │
│ • build_style_encoder(args) → StyleEncoder                      │
│ • build_content_encoder(args) → ContentEncoder                  │
│ [FST mode:]                                                     │
│ • build_mss_encoder(args) → MultiScaleStyleEncoder             │
│ • build_fst(args) → FontStyleTransformationModule              │
│ • build_fst_projection(...) → nn.Linear                        │
│ • build_original_style_projection(...) → nn.Linear             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. LOAD STATE DICTS (inference/training setup)                 │
├─────────────────────────────────────────────────────────────────┤
│ unet.load_state_dict(torch.load("ckpt/unet.pth"))             │
│ style_encoder.load_state_dict(torch.load("ckpt/style_..."))   │
│ content_encoder.load_state_dict(torch.load("ckpt/content..."))│
│ [FST mode:]                                                     │
│ mss_encoder.load_state_dict(torch.load("ckpt/mss_..."))       │
│ fst_module.load_state_dict(torch.load("ckpt/fst_..."))        │
│ fst_projection.load_state_dict(torch.load("ckpt/fst_proj..."))│
│ original_style_projection.load_state_dict(...)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. CREATE MODEL (pass pre-loaded modules)                      │
├─────────────────────────────────────────────────────────────────┤
│ Standard:                                                       │
│   model = FontDiffuserModelDPM(                                │
│       unet=unet,                                               │
│       style_encoder=style_encoder,                             │
│       content_encoder=content_encoder                          │
│   )                                                            │
│                                                                 │
│ FST:                                                           │
│   model = FontDiffuserWithFST(                                 │
│       unet=unet,                                               │
│       style_encoder=style_encoder,                             │
│       content_encoder=content_encoder,                         │
│       mss_encoder=mss_encoder,                                 │
│       fst_module=fst_module,                                   │
│       fst_projection=fst_projection,                           │
│       original_style_projection=original_style_projection      │
│   )                                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. FORWARD PASS (inference or training)                        │
├─────────────────────────────────────────────────────────────────┤
│ INFERENCE (sample_optimized.py):                               │
│   images = pipe.generate(                                      │
│       content_images=content_batch,                            │
│       style_images=style_batch,                                │
│       num_inference_steps=20,                                  │
│       ...                                                      │
│   )                                                            │
│                                                                 │
│ TRAINING (trainer_fst.py):                                     │
│   outputs = model(                                             │
│       noisy_latents=noisy_latents,                            │
│       timestep=timesteps,                                      │
│       content_img=content_images,                              │
│       style_source_img=style_source_images,                    │
│       style_target_img=style_images,                           │
│   )                                                            │
│   loss = F.mse_loss(outputs["noise_pred"], noise)            │
│   accelerator.backward(loss)                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. SAVE CHECKPOINTS (after training)                           │
├─────────────────────────────────────────────────────────────────┤
│ save_dir = f"output/checkpoint_{global_step}/"                │
│                                                                 │
│ Standard:                                                       │
│   torch.save(unet.state_dict(), "unet.safetensors")          │
│   torch.save(style_encoder.state_dict(), "style_encoder...")  │
│   torch.save(content_encoder.state_dict(), "content_...")     │
│   [if phase_2] torch.save(scr.state_dict(), "scr...")        │
│                                                                 │
│ FST:                                                           │
│   torch.save(unet.state_dict(), "unet.safetensors")          │
│   torch.save(style_encoder.state_dict(), "style_encoder...")  │
│   torch.save(content_encoder.state_dict(), "content_...")     │
│   torch.save(mss_encoder.state_dict(), "mss_encoder...")      │
│   torch.save(fst_module.state_dict(), "fst_module...")        │
│   torch.save(fst_projection.state_dict(), "fst_proj...")      │
│   torch.save(original_style_projection.state_dict(), "...")   │
└─────────────────────────────────────────────────────────────────┘