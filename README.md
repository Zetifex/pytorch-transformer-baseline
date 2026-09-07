# Scaled Decoder-Only Transformer (PyTorch)

A modular, from-scratch PyTorch implementation of an autoregressive, decoder-only Transformer language model trained on CUDA. The architecture scales the foundational mechanisms from *Attention Is All You Need* with modern Pre-Layer Normalization and decoupled MLOps tracking.

---

## 1. Architectural Specifications

| Hyperparameter | Baseline Configuration | Scaled Configuration |
| :--- | :--- | :--- |
| **Total Parameters** | 0.21 M | **10.79 M** |
| **Block Size (Context Window)** | 32 tokens | **256 tokens** |
| **Embedding Dimension** | 64 | **384** |
| **Attention Heads** | 4 | **6** |
| **Transformer Layers** | 4 | **6** |
| **Batch Size** | 16 | **64** |
| **Dropout Rate** | 0.1 | **0.2** |
| **Learning Rate** | 1e-3 | **3e-4** |
| **Training Steps** | 3,000 | **5,000** |

---

## 2. Quantitative Performance & Training Dynamics

The scaled run was evaluated across 200 validation batches every 500 iterations using decoupled evaluation mode.

```text
Step 0:    Train Loss 4.3972, Val Loss 4.4026
Step 1000: Train Loss 1.5477, Val Loss 1.7323
Step 2000: Train Loss 1.3136, Val Loss 1.5581
Step 3000: Train Loss 1.2044, Val Loss 1.5042
Step 4000: Train Loss 1.1287, Val Loss 1.4925
Step 4999: Train Loss 1.0530, Val Loss 1.4973



Salaise your RamIlinCe,
Fortere to bith from an Your broth,
I what no bordise neving hious poass:




Yet,
Against their little virtues committed
That in all usual as true?

Nurse:
He has everal from the falcon'st for him.
Speed, Rivers, perfect him privately, down so venom