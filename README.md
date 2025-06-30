# (How) Can Transformers Predict Pseudo-Random Numbers?
ICML 2025 accept(poster)
[![arXiv](https://img.shields.io/badge/arXiv-2502.10390-b31b1b.svg)](https://arxiv.org/abs/2502.10390)

This repository contains the code and experiments for the paper *"(How) Can Transformers Predict Pseudo-Random Numbers?"* which investigates the ability of Transformer networks to learn and predict pseudo-random number sequences from Linear Congruential Generators (LCGs).

## Overview

We study how Transformers can learn pseudo-random sequences defined by the recurrence relation:

$$x_{t+1} = a \cdot x_t + c \pmod{m}$$

Our analysis reveals two key scenarios of increasing complexity:

1. **Fixed Modulus (FM)**: In-context learning LCG sequences with unseen parameters $(a, c)$ but fixed modulus $m$
   - Successfully demonstrates learning up to $m = 2^{32}$
   - Models learn to factorize the modulus and use digit-wise representations

2. **Unseen Moduli (UM)**: In-context learning sequences with completely unseen moduli
   - Generalizes to unseen moduli up to $m_{\text{test}} = 2^{16}$  
   - Strategy: modulus estimation + prime factorization
   - Sharp accuracy transition at critical depth = 3

## Repository Structure

```
transformer-prng/
├── training/           # Model training scripts
│   └── fixed_modulus/  # Training for fixed modulus scenario
├── scaling/            # Scaling up the modulus experiments and evaluation
│   ├── notebooks/      # Demo notebooks 
│   ├── models/         # Pre-trained model checkpoints
│   └── utils/          # Core utilities (datasets, models, evaluation)
├── interpretability/   # Analysis and interpretability tools
│   ├── fixed_modulus/  # Analysis for fixed modulus
│   └── unseen_modules/ # Analysis for unseen moduli
└── LICENSE
```

## Quick Start

### Installation

```bash
git clone https://github.com/your-username/transformer-prng.git
cd transformer-prng
pip install torch numpy pandas matplotlib tqdm sympy
```

### Demo

Try the interactive demo notebook to see pre-trained models in action:

```bash
cd scaling/notebooks
jupyter notebook demo_fm.ipynb
```

This notebook demonstrates:
- Loading pre-trained models for fixed modulus prediction
- Generating LCG sequences with various parameters
- Evaluating model performance on unseen $(a, c)$ pairs

### Training Your Own Models

To be updated.


## Interpretability Analysis

The `interpretability/` directory contains tools for understanding how models learn:

- **Embedding Analysis**: PCA of learned embeddings
- **Attention Patterns**: Visualization of attention weights
- **Activation Patching**: Patching experiments  
- **Digit-wise Accuracy**: Fine-grained performance analysis

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@misc{tao2025howtransformerspredictpseudorandom,
      title={(How) Can Transformers Predict Pseudo-Random Numbers?}, 
      author={Tao Tao and Darshil Doshi and Dayal Singh Kalra and Tianyu He and Maissam Barkeshli},
      year={2025},
      eprint={2502.10390},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.10390}, 
}
```
