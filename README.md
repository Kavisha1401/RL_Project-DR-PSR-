# Diversity-Regularized PSR (DR-PSR)

> **Innovative Path Forward:** Address the diversity-collapse issues associated with PSR via the addition of an entropy reward within the training loss.

Based on the research findings detailed in "The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning" (NeurIPS 2025), we propose **DR-PSR**, a new training objective combining elements of PSR and an entropy term to mitigate the diversity-collapse issues inherent to regular PSR.

---

## Overview

One major weakness associated with using PSR — maximizing the log probability of the response — is the fact that such an approach leads to output collapse, meaning the model tends to generate deterministic outputs, which negatively impacts diversity-sensitive metrics like Pass@k when k is large.

By adding an explicit entropy term to PSR's loss, DR-PSR mitigates this issue through the following loss function:

```
L_DR-PSR = L_PSR - β × H(π_θ)
```
## Role of the Notebook

- Learns a **GPT-2 (124M)** language model on a minimal dataset of synthetic arithmetic problems (50 samples of the format `Q: a + b = ? A: answer`)
- Considers **four training algorithms**:
  - **PSR** — Positive Sample Reward (maximizing the log-probability of positive samples)
  - **NSR** — Negative Sample Reward (minimizing the log-probability of negative samples)
  - **DR-PSR** *(Novel)* — Positive Sample Reward with an entropy regularization term (`β × H(π_θ)`)
  - **RLVR** — a combination of PSR and NSR (baseline for comparison, similar to GRPO/PPO)
- Estimates **Pass@k** metrics for k = {1, 4, 8} using the unbiased estimator of Chen et al. (2021)
- Performs **ablation on β**, demonstrating the impact of entropy regularization on diversity
- Monitors and plots **token-level entropy** to analyze the degree of diversity loss during training

---

## Major Findings

| Approach | Pass@1 | Pass@8 | Diversity |
|----------|--------|--------|-----------|
| PSR | ✔ High accuracy | ✗ Low (collapse) | ✔ Degrading diversity |
| NSR | Competitively accurate | More accurate | ✔ Maintaining diversity |
| **DR-PSR** *(Novel)* | ✔ Matches PSR | ✔ Recovers accuracy | ✔ Preserves diversity |
| RLVR | Intermediate | Intermediate | ✔ Moderate diversity |

**Important observation**: While obtaining the same Pass@1 score as the standard PSR loss, DR-PSR recovers the Pass@8 performance — confirming that entropy regularization avoids diversity collapse without compromising the model’s correctness.

---

## Code Structure

```
notebook.ipynb
├── Section 1 — Mini Math Dataset         # Fake dataset of arithmetic QA pairs (50 examples)
├── Section 2 — Model & Tokenizer         # GPT-2 (small) from HuggingFace
├── Section 3 — Utility Functions         # Log-probability, entropy, rollout, verification, k-pass metric
├── Section 4 — Training Functions        # PSR, NSR, DR-PSR losses, and full RLVR objective
├── Section 5 — Run All Methods           # Training function for all methods combined
├── Section 6 — Output & Plots            # Graphical output of Pass@k results and final chart
├── Section 7 — β Experiment              # Impact of the entropy weight parameter
└── Section 8 — Entropy Visualization     # Diversity collapse in PSR vs DR-PSR prevention
```

---

## Dependencies


```bash
pip install transformers datasets accelerate matplotlib numpy torch
```

### Dependencies
- Python 3.8+
- PyTorch (CPU compatible; GPU highly recommended)
- HuggingFace Transformers

---

## Setup Guide

1. **Clone/Download** notebook.
2. **Install Dependencies** (above).
3. **Run all cells** from top to bottom. This notebook does everything: dataset creation, loading GPT-2, training of all methods, and plotting.

> **Note about Runtime:** Default values (`N_EPOCHS = 5`, 50 prompts, 4 rollouts) are tuned for a ~30-minute run on a free Colab/Kaggle GPU. Try `N_EPOCHS = 10 - 15` and `gpt2-medium`.

---

## Hyperparameter Table

| Parameter Name  | Default Value | Description                          |
|-----------------|---------------|--------------------------------------|
| N_EPOCHS        | 5             | Epochs to train each model          |
| LR              | 5e-5          | AdamW optimizer learning rate        |
| N_ROLLOUTS      | 4             | Number of rollout generations       |
| BETA            | 0.02          | Entropy penalty factor (only DR-PSR) |
| SEED            | 42            | Global random seed                  |

## Output Files

Running the notebook produces the following plot files:

| Filename | Description |
| -------- | ----------- |
| `pass_at_k_curves.png` | Pass@k across epochs for all four models |
| `final_pass_at_k_bar.png` | Bar chart comparing Pass@k for all four models |
| `beta_ablation.png` | Effect of β on DR-PSR Pass@k |
| `entropy_tracking.png` | Token entropy through training: PSR collapse vs DR-PSR |

---

## Core Concepts

**Pass@k** – the probability of generating at least k correct answers. Higher pass rates encourage diversity; for example, a model that produces the same output each time would do well at Pass@1 but not at Pass@8.

**Entropy bonus** – taking `β × H(π_θ)` away from the objective means that an optimizer will be punished if it causes the distribution to become too sharply concentrated (low entropy), thus preventing PSR from assigning probability mass to one and only one answer.

**Unbiased Pass@k Estimator** (Chen et al. 2021):
```
Pass@k = 1 − C(n-c, k) / C(n, k)
```
where `n` = number of samples, `c` = number of correct samples.

## Bibliography 
>Inspired by: "The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning", NeurIPS 2025.
