# DDPM with Diffusion Transformers (DiT) on MNIST

Implementation of Denoising Diffusion Probabilistic Models (DDPM) using a Diffusion Transformer (DiT) backbone, trained on MNIST. Covers unconditional generation, class-conditional generation, classifier-free guidance (CFG), and attention map visualisation across the denoising trajectory.

---

## Results

**Unconditional generation** — coherent digit samples after 40 epochs, converging to a training loss of ~0.022.

**Conditional generation** — class-conditional samples with clear per-digit identity and stroke diversity.

**Classifier-free guidance** — CFG scale w=5–10 produces the sharpest samples. w≥15 causes over-saturation and distortion as samples are pushed outside the learned data distribution.

| CFG Scale | Quality |
|---|---|
| w = 1 | Rough, inconsistent strokes |
| w = 5 | Sharp, class-consistent |
| w = 10 | Sharp, slight thickening |
| w = 15 | Over-saturated, bleeding edges |
| w = 20 | Distorted |

---

## Repository Structure

```
.
├── README.md
├── train.py                 # Training loop: unconditional + conditional DDPM
├── dit.py                   # DiT architecture (transformer backbone for diffusion)
│
└── figures/
    ├── forward_diffusion_process.png    # Forward noising process visualisation
    ├── evolution_epoch_*.png            # Denoising evolution grids per epoch
    ├── cfg_scale_*.png                  # Generated samples at CFG scales 1,5,10,15,20
    ├── attention_maps.png               # DiT attention maps during denoising (digit 7)
    └── loss_curve.png                   # Training loss curve
```

---

## Methodology

### Architecture
The denoising network is a DiT (Diffusion Transformer),  a transformer that operates on patchified image tokens, conditioned on both diffusion timestep and class label via adaptive LayerNorm (adaLN-Zero).

| Component | Detail |
|---|---|
| Image size | 32×32, 1 channel |
| Patch size | 4×4 |
| Hidden size | 256 |
| Depth | 4 transformer blocks |
| Heads | 4 attention heads |
| Timestep encoding | Sinusoidal embeddings → MLP |
| Class conditioning | Label embedding added to timestep embedding |

### Forward Diffusion
Gaussian noise is added over T=500 timesteps with a linear beta schedule (1e-4 to 0.03):

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### Training Objective (DDPM)
The network is trained to predict the noise added at each timestep:

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

### Classifier-free Guidance
During training, class labels are randomly dropped with p=0.1, teaching the network both conditional and unconditional denoising. At inference, predictions are combined as:

$$\tilde{\epsilon} = \epsilon_\theta(x_t, t) + w \left( \epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t) \right)$$

### Attention Visualisation
Attention maps are extracted from the first DiT block at timesteps t=490, 400, 250, 100, 10 during a single denoising trajectory (digit 7). Early timesteps show global attention across the full image; later timesteps localise to the digit shape as structure emerges.

---

## Setup & Usage

```bash
pip install torch torchvision timm einops matplotlib numpy
```

```bash
python train.py
```

All generated figures are saved to `figures/`. Key config at the top of `train.py`:

| Argument | Default | Description |
|---|---|---|
| `T` | 500 | Number of diffusion timesteps |
| `EPOCHS` | 40 | Training epochs |
| `BATCH_SIZE` | 128 | Batch size |
| `LR` | 3e-4 | Learning rate |
| `DIM` | 256 | Transformer hidden size |
| `DEPTH` | 4 | Number of DiT blocks |
| `PATCH_SIZE` | 4 | Image patch size |

---

## Report

Full analysis with figures is in [`report.pdf`](./report.pdf).

---

## Acknowledgements

The DiT architecture (`dit.py`) is adapted from [Peebles & Xie (2023)](https://github.com/facebookresearch/DiT), originally from Meta Platforms Inc., licensed under the license found in the repository. The DDPM training framework and coursework structure were provided by the University of Cambridge Department of Engineering as part of a graduate advanced computer vision course. Original framework developed by Ayush Tewari and Elliot Wu. All experimental analysis, visualisations, and findings are my own work.

---

## References

1. Ho et al. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
2. Peebles & Xie. *Scalable Diffusion Models with Transformers.* ICCV 2023.
3. Ho & Salimans. *Classifier-free Diffusion Guidance.* arXiv 2022.
