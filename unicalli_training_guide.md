# UniCalli Training Improvements Guide

This document explains the new parameters introduced by the UniCalli improvements and how they affect the FontDiffuser training pipeline (`train_unicalli.py`).

### 1. `--style_noise_fraction` (Default: `0.0`)
* **What it does:** Controls how much noise is added to the style reference image during training. `0.0` keeps the style image completely clean, while `1.0` would apply the full forward diffusion noise. 
* **Why use it:** Adding a small amount of noise (e.g., `0.05` to `0.1`) acts as a powerful data augmentation technique. It prevents the style encoder from overfitting to exact pixel values of the training styles, encouraging it to learn more robust style features.
* **Note:** In this setup, the content image is *always* kept clean to ensure the content encoder has a perfectly sharp signal of the structural glyph.

### 2. `--p_drop_content` (Default: `0.1`)
* **What it does:** The probability (0.0 to 1.0) of dropping the content conditioning signal during a training step. 
* **Why use it:** Forces the model to not become overly reliant on the exact content image, which heavily improves style-content disentanglement. This is especially useful for preventing the model from collapsing on long-tail (rare) font styles.

### 3. `--p_drop_style` (Default: `0.05`)
* **What it does:** The probability (0.0 to 1.0) of dropping the style conditioning signal during a training step.
* **Why use it:** Similar to content dropout, it acts as a form of standard Classifier-Free Guidance (CFG) dropping, but specifically targeted at the style branch. 

### 4. `--use_hard_negative` (Default: `True`)
* **What it does:** When a condition is dropped (triggered by `p_drop_content` or `p_drop_style`), standard diffusion typically replaces the condition with pure zeros or pure noise. If this flag is enabled, it replaces the dropped condition with a *different sample's* condition from the same batch.
* **Why use it:** This is a much tougher challenge for the model. It forces the denoiser to learn invariance to mismatched content/style pairs while still seeing realistic image statistics, leading to significantly better disentanglement than dropping to pure noise.

### 5. `--curriculum_steps` (Default: `1000`)
* **What it does:** The number of training steps over which the dropout probabilities (`p_drop_content` and `p_drop_style`) ramp up from `0.0` to their target values.
* **Why use it:** Hard-negative dropout is a very difficult task. The curriculum allows the model to learn the basic FontDiffuser task (reconstruction) during the early steps, gradually phasing in the hard dropout to refine disentanglement once the model has "warmed up."

### Summary Recommendation
If you just want to test these improvements out of the box, the default parameters are highly balanced. You can simply run `train_unicalli.py` without passing these flags and it will default to: `--p_drop_content 0.1`, `--p_drop_style 0.05`, `--use_hard_negative`, and `--curriculum_steps 1000`. You might only want to explicitly pass `--style_noise_fraction 0.05` if you want to experiment with style augmentation.
