# Unveiling Hidden Identities in Morphed Faces
**Latent Diffusion Subtraction via DDIM Inversion**

This repository contains the implementation and evaluation scripts for separating combined identities in morphed face images using Latent Diffusion Models (LDMs).

## Project Overview

Face morphing attacks blend the facial features of two individuals to deceive Face Recognition (FR) systems. This project leverages an identity-conditioned negative prompting strategy within an LDM framework to suppress the known biometric identity (P-) and attempts to reveal the hidden attacker's identity (Target).

### Key Features & Contributions

- **DDIM Inversion:** Implemented a deterministic forward process to capture the exact noise representation of the morphed images (x_T), preventing the generation of random, unconditioned faces.
- **Autoencoder Artifact Mitigation:** Enhanced denormalization (`clamp`) to reduce severe VQGAN compression noise during latent decoding.
- **Automated Evaluation Pipeline:** Custom scripts to iterate through multiple datasets from the SYN-MAD 2022 benchmark (OpenCV, FaceMorpher, Webmorph, MIPGAN), extract contexts via ElasticFace, and evaluate cosine similarity metrics against a strict FR threshold (tau=0.321 based on CASIA-WebFace).
- **Architectural Comparison:** Evaluated and compared static (`NegFaceDiff`) and dynamic (`AdaptDiff`) guidance architectures.

## Setup & Installation (Google Colab Environment)

Before running the evaluation scripts, ensure your environment is downgraded to compatible legacy versions. Run the following commands in your notebook:

```bash
# Uninstall conflicting packages
!pip uninstall -y pytorch-lightning lightning
# Install required legacy dependencies
!pip install hydra-core omegaconf "pytorch-lightning==1.9.5" einops torchmetrics facenet-pytorch
```

**Note:** Dependency conflict warnings regarding numpy during installation in Colab are expected and do not affect the execution of the diffusion or FR models.

## Repository Structure

```
├── configs/                  # Hydra configuration files
├── results/
│   ├── final_comparison_metrics.csv
│   ├── plot_results.py         # Script to generate separated evaluation plots
│   ├── plots/                 # FR threshold evaluation charts per dataset
│   └── visual_samples/        # Generated image grids across different guidance weights
├── scripts/                   # Automation scripts for batch processing
│   ├── evaluate_all_models.py
│   └── run_all_experiments.py
├── src/                       # Core diffusion logic, inversion, and sampling
│   ├── ddpm.py
│   ├── helpers.py
│   └── sample.py
├── Report.pdf                 # Final academic report
└── README.md
```

## Evaluation & Conclusion

Extensive multi-dataset evaluation indicates that applying a negative guidance weight successfully suppresses the known biometric identity (P-). However, experiments reveal a critical architectural limitation: the subtraction process induces severe latent space collapse at effective weights (e.g., w=0.5). Consequently, the generated images fail to reach the required FR verification threshold (tau=0.321) to successfully reveal the hidden Target identity.

## How to Run

1. Clone the repository into your environment.
2. Open the provided Colab notebook and execute the setup cells.
3. Run `scripts/run_all_experiments.py` to process the datasets.
4. Run `scripts/evaluate_all_models.py` to compute cosine similarities.
5. Run `results/plot_results.py` to generate the evaluation charts.

