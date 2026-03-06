# Tutorial Improvements TODO

## Dataset Context and Expected Explanations

- [ ] **Add dataset analysis section to each tutorial notebook**
  - Analyze the dataset properties (class distribution, molecular weight distribution, functional group prevalence)
  - Discuss known structure-activity relationships from the literature
  - For AqSolDB: ClogP correlation, polar surface area effects, hydrogen bonding
  - For Ames mutagenicity: known structural alerts (nitro groups, aromatic amines, epoxides)
  - Include visualizations of representative molecules from each class
  - Define what "good" explanations should highlight based on domain knowledge

- [ ] **Create ground-truth explanation baselines**
  - Identify molecules with well-documented SAR in each dataset
  - Annotate expected important substructures based on literature
  - Use these as qualitative validation for explanation methods

## Notebook 04: Uncertainty Quantification

- [ ] **Create `04_uncertainty_quantification.ipynb`**
  - Conceptual introduction distinguishing epistemic vs. aleatoric uncertainty
  - Motivation: why uncertainty matters for XAI (unreliable predictions need cautious interpretation)

- [ ] **Implement uncertainty estimation methods**
  - Monte Carlo Dropout for GNNs
  - Deep Ensembles (train multiple models, aggregate predictions)
  - Evidential deep learning (if applicable)
  - Conformal prediction for calibrated prediction intervals

- [ ] **Connect uncertainty to explanations**
  - Show how explanation reliability correlates with prediction confidence
  - Discuss when to trust/distrust explanations based on uncertainty
  - Visualize uncertainty alongside attributions

- [ ] **Practical considerations**
  - Computational cost of ensemble methods
  - Calibration plots and reliability diagrams
  - Out-of-distribution detection for novel molecules

## Gradient-Based Methods with Captum

- [ ] **Add gradient-based methods section to Tutorial 01 or create dedicated section**
  - Use `captum` library (already a dependency for LIME)
  - Methods to implement:
    - Vanilla Gradients / Saliency Maps
    - Integrated Gradients (with baseline selection discussion)
    - GradientSHAP (connection to kernel SHAP)
    - Input x Gradient

- [ ] **Adapt gradient methods for molecular fingerprints**
  - Discuss interpretability challenges with binary fingerprints
  - Show gradient flow back to fingerprint bits
  - Compare computational cost vs. SHAP/LIME

- [ ] **Add gradient-based GNN explanations to Tutorial 02**
  - Integrated Gradients for node/edge features
  - Comparison with perturbation-based methods (GNNExplainer)
  - Discuss when gradient methods are preferable (speed vs. faithfulness tradeoff)

- [ ] **Include method comparison table**
  - Computational cost
  - Theoretical guarantees (axioms satisfied)
  - Typical use cases
  - Hyperparameter sensitivity
