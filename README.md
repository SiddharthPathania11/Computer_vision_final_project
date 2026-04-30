# Weather Condition Classification from Outdoor Photographs

**MSML640 Final Project** | Image Classification with Transfer Learning

**Team:** Akhil Shekkari, Siddharth Pathania

---

## Overview

This project builds an image classifier that identifies the prevailing weather condition in an outdoor photograph across five classes:

| Class | Visual cue |
|-------|-----------|
| Clear | High brightness, sharp detail |
| Cloudy | Reduced brightness, overcast sky |
| Rainy | Streaking artifacts, blur |
| Foggy | Low contrast, occlusion |
| Snowy | White coverage, diffuse light |

Reliable visual weather recognition is a building block for dashcam-based driver assistance, autonomous-driving perception triggers, agricultural monitoring, and automatic photo organization.

---

## Approach

We fine-tune an ImageNet-pretrained **EfficientNet-B0** (~5.3 M parameters). EfficientNet-B0 offers a strong accuracy-per-parameter ratio, trains comfortably on a single Colab T4 GPU, and is the standard transfer-learning baseline in the literature. **ResNet-50** is kept as a fallback if optimization on our dataset proves unstable.

### Four Experimental Configurations

| Config | Training data | Augmentation |
|--------|--------------|--------------|
| 1: Baseline | Our data only | None |
| 2: Augmentation | Our data | Horizontal flip, random crop, color jitter, rotation |
| 3: Synthesis | Our data + Stable Diffusion images | None |
| 4: Synthesis + Aug | Our data + Stable Diffusion images | Same as Config 2 |

---

## Dataset

**Source (two streams)**

- **Seed pool:** publicly available weather datasets (Multi-class Weather Dataset, RFS Weather collection) to bootstrap class coverage.
- **Self-captured photos:** taken across the DMV area (College Park, Lanham, Washington DC) to satisfy the in-the-wild requirement and introduce a deliberate distribution shift between the seed pool and the test set.

**Size and splits**

- Target ~400 images total (~80 per class)
- Stratified splits: **70% train / 15% validation / 15% test**
- No overlap between splits; all self-captured images reserved for validation and test where possible.

**Synthesized data (Configs 3 & 4)**

Generated with Stable Diffusion using class-specific prompts (e.g., *"rural highway in heavy fog at dusk, photographic"*, *"snowy suburban street, overcast sky"*). Synthetic images are tagged separately so clean ablations can quantify synthesis contribution vs. distribution distortion.

---

## Evaluation & Hypothesis

For all four configs we report:
- Per-epoch train/validation loss and accuracy curves
- Test-set confusion matrix
- Side-by-side confusion matrix comparison (baseline vs. best config)
- Per-class error analysis (which classes fail, which images are misclassified, and why)

**Robustness suite:** the best model is evaluated under four synthetic perturbations applied to the clean test set:

| Perturbation | Proxy for |
|-------------|-----------|
| Gaussian blur | Rain |
| Brightness reduction | Overcast |
| Random rectangular occlusion | Fog / obstruction |
| Additive Gaussian noise | Sensor degradation |

**Central hypothesis:** Augmentation will improve robustness most on perturbations that resemble the augmentation distribution itself (color jitter ↔ overcast, blur ↔ rain), while synthesis will help most on minority classes where real samples are scarce. Evidence for or against this hypothesis (not raw accuracy) is the primary deliverable.

---

## Project Timeline

| Window | Milestone |
|--------|-----------|
| Apr 24 – Apr 26 | Dataset assembly: seed pool + self-captured images, cleaning, splits |
| Apr 27 – Apr 29 | Config 1 (baseline) and Config 2 (augmentation) training + logging |
| Apr 30 – May 1 | Synthesis pipeline; Config 3 and Config 4 training |
| May 2 – May 3 | Robustness suite, confusion matrix analysis, slide deck, video |
| May 4 | Final submission on ELMS + GitHub |

---

## Repository Structure

```
.
├── data/
│   ├── raw/          # seed pool images
│   ├── captured/     # self-captured images
│   └── synthetic/    # Stable Diffusion generated images
├── notebooks/        # training and evaluation notebooks
├── src/              # model, dataset, augmentation, utils
├── results/          # saved checkpoints, curves, confusion matrices
└── README.md
```

---

## Getting Started

```bash
# clone the repo
git clone <repo-url>
cd Computer_vision_final_project

# install dependencies
pip install -r requirements.txt

# train a configuration (example: Config 2)
python src/train.py --config 2
```

> Requirements file and training script will be added as the implementation progresses.

---

## AI Usage Disclosure

Per project policy, no AI tool was used to generate the core CV pipeline logic or the experimental design. AI assistance was used only for drafting and formatting documents.
