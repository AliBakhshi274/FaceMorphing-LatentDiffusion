# Unveiling Hidden Identities in Morphed Faces
**Latent Diffusion Subtraction via DDIM Inversion**

This repository contains the implementation and evaluation scripts for separating combined identities in morphed face images using Latent Diffusion Models (LDMs).

## Project Overview
Face morphing attacks blend the facial features of two individuals to deceive Face Recognition (FR) systems. This project leverages a negative prompting strategy within an LDM framework to suppress the known biometric identity (P-) and reveal the hidden attacker's identity.

### Key Features & Fixes
- **DDIM Inversion:** Implemented a deterministic forward process to capture the exact noise representation of the morphed images, preventing the generation of random, unconditioned faces.
- **Autoencoder Artifact Mitigation:** Enhanced denormalization (`clamp`) to reduce VQGAN compression noise.
- **Automated Pipeline:** Custom scripts to automatically iterate through multiple datasets (OpenCV, FaceMorpher, Webmorph, MIPGAN), extract contexts via ElasticFace, and evaluate cosine similarity metrics.
- **Architectural Comparison:** Evaluated and compared `NegFaceDiff` and `AdaptDiff` architectures.

## Setup & Installation (Google Colab Environment)
Before running the evaluation scripts, ensure your environment is downgraded to the compatible legacy versions. Run the following commands in your notebook:

```bash
# Uninstall conflicting packages
!pip uninstall -y pytorch-lightning lightning

# Install required legacy dependencies
!pip install hydra-core omegaconf "pytorch-lightning==1.9.5" einops torchmetrics facenet-pytorch
```

> **Note:** Dependency conflict warnings regarding numpy during installation in Colab are expected and do not affect the execution of the diffusion or FR models.

## Repository Structure

```
.
├── src/                    # Core diffusion sampling logic, helpers, and modified DDPM implementation
├── scripts/                # Automation scripts
│   ├── run_all_experiments.py     # Batch generation across datasets
│   └── evaluate_all_models.py     # Numerical evaluation and metrics
├── results/                # Averaged cosine similarity metrics, comparison plots, and visual sample grids
├── configs/                
├── Report.pdf
└── README.md
```

## Evaluation
Our experiments indicate that applying a negative guidance weight of **w=0.5** optimally decouples the identities. Higher weights (w=1.0) lead to latent space collapse, which is documented in our visual samples.

> **Note:** Pre-trained weights and large dataset files are excluded from this repository due to size constraints. The logic for dynamic path loading is available in the scripts.

## How to Run
1. Clone the repository
2. Open the provided Colab notebook
3. Execute the setup cells
4. Run `run_all_experiments.py` followed by `evaluate_all_models.py`
