# Cancer-ACPNet

Cancer-ACPNet is a two-stage deep learning framework for anticancer peptide (ACP) prediction and cancer-type activity prediction. It combines pretrained ESM-2 sequence embeddings with BLOSUM62 and AAIndex handcrafted descriptors, followed by capsule-based feature fusion and cancer-type-specific prediction.

## Overview

Cancer-ACPNet has two stages:

1. **Stage 1: ACP identification**  
   Predicts whether a peptide is an anticancer peptide or a non-anticancer peptide.

2. **Stage 2: Cancer-type activity prediction**  
   Predicts the likely activity of ACPs across seven cancer types: breast, lung, colon, cervix, skin, prostate, and blood.

The framework was designed for small and imbalanced ACP datasets, where both sequence-level information and physicochemical peptide properties are important.

## Main Components

- **ESM-2 embeddings** for contextual peptide sequence representation
- **BLOSUM62 descriptors** for substitution-based evolutionary information
- **AAIndex descriptors** for physicochemical amino acid properties
- **Fold-specific PCA** to reduce handcrafted features without data leakage
- **Capsule-based fusion** for ACP/non-ACP classification
- **One-vs-all ensemble learning** for cancer-type activity prediction

## Performance Summary

### Stage 1: ACP Identification

| Dataset | Accuracy | F1-score | MCC |
|---|---:|---:|---:|
| Set 1 | 84.08% | 83.95% | 68.22% |
| Set 2 | 96.74% | 96.81% | 93.59% |
| Set 3 | 84.75% | 85.39% | 70.04% |

### Stage 2: Seven Cancer-Type Prediction

| Metric | Value |
|---|---:|
| Accuracy | 89.37% |
| Sensitivity | 87.13% |
| Specificity | 89.62% |
| F1-score | 88.18% |
| AUC | 92.77% |
| MCC | 77.79% |

## Interpretability

Cancer-ACPNet includes saliency analysis, SHAP-based feature analysis, mutation validation, and residue perturbation analysis to help explain model predictions. These analyses suggest that the model focuses on biologically meaningful peptide regions and physicochemical patterns related to ACP activity.

## Repository Structure

```text
Cancer-ACPNet/
├── data/              # Datasets
├── Code/           # Training and evaluation script
├── figures/           # Figures
└── README.md
