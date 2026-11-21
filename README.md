# Interpretable Machine Learning (Fairness-Aware Insurance Cost Prediction)

A practical project on **Interpretable Machine Learning (IML)** and **Explainable AI (XAI)**, focusing on understanding, evaluating, and communicating the behaviour of machine learning models.

The project combines predictive modelling with model interpretation techniques to investigate **which features drive predictions, how individual observations are explained, and how model behaviour changes across the feature space**.

## Project Overview

The main objective is to build machine learning models for insurance claim-related prediction tasks and then use interpretability techniques to understand their predictions.

The repository includes both **Python/Jupyter** and **R** implementations, covering predictive modelling, model evaluation, feature effects, feature importance, SHAP-based explanations, and Individual Conditional Expectation (ICE) analysis.

A key component is a **three-stage frequency–severity modelling framework** for predicting annual insurance claim costs.

The three stages are:

1. **Claim probability** — predict whether a policyholder will have a claim.
2. **Claim frequency** — estimate the number of claims for observations with at least one claim.
3. **Claim severity** — estimate the monetary cost associated with claims.

The predictions from these components are then combined to obtain an estimate of the total expected claim cost:

\[
E[\text{Cost}] =
P(\text{Claim}) \times
E[\text{Frequency} \mid \text{Claim}] \times
E[\text{Severity} \mid \text{Claim}]
\]

This provides a more structured alternative to directly predicting the total claim cost.

## Interpretability Techniques

The project explores several approaches for understanding model behaviour, including:

- **Feature Importance**
- **SHAP values**
- **SHAP summary plots**
- **Local SHAP explanations**
- **Feature Effects**
- **Individual Conditional Expectation (ICE) plots**
- **Prediction error analysis**
- **Residual analysis**
- **Learning curves**

These methods are used to provide both **global explanations** of model behaviour and **local explanations** for individual predictions.

### Global Interpretability

Global analysis investigates the overall relationship between the input variables and model predictions.

Examples include:

- Which features contribute most strongly to predictions?
- What is the direction and magnitude of feature effects?
- How does model performance change with training-set size?

### Local Interpretability

Local explanations focus on individual observations and investigate why a particular prediction was produced.

SHAP-based local explanations are used to identify the features that contribute positively or negatively to an individual prediction.

## Modelling

The repository contains several predictive modelling workflows.

### Classification

Claim occurrence is formulated as a binary classification problem:

```text
has_claim = 1  → policyholder has at least one claim
has_claim = 0  → no claim
```

An XGBoost classifier is trained with probability predictions and evaluated using cross-validation.

The preprocessing pipeline includes:

- Numerical imputation
- Categorical imputation
- One-hot encoding
- Cross-validation

### Claim Frequency

For observations with at least one claim, the number of claims per year is modelled as a regression problem.

A Random Forest model is used for the frequency component.

### Claim Severity

Claim severity is modelled using the annual claim cost after applying a logarithmic transformation:

\[
\text{log\_cost} = \log(1 + \text{Cost})
\]

An XGBoost regression model is then used to predict severity, with the transformation reversed using:

\[
\text{Cost} = \exp(\text{prediction}) - 1
\]

### Combined Frequency–Severity Model

The final prediction combines the three components:

```text
Predicted Total Cost
    =
Probability of Claim
    ×
Predicted Claim Frequency
    ×
Predicted Claim Severity
```

The resulting predictions are compared against the observed annual claim costs using metrics such as **RMSE** and correlation. The model is also compared with a simple mean-based baseline.

## Repository Structure

```text
Interpretable-ML/
│
├── Interpretable_ML.ipynb
│   └── Main Python notebook containing the interpretability analysis
│
├── IML project - version 5.R
│   └── Main R implementation of the modelling and interpretability workflow
│
├── 3 stage model.R
│   └── Three-stage frequency–severity insurance model
│
├── IML_Project_2 (1).pdf
│   └── Project report
│
├── Feature Importance.png
├── Feature Effects.png
├── FeaturesEffects.png
├── ICE plots.png
├── Learning Curve.png
├── Prediction error.png
├── Residuals plot.png
├── Shapley Summary.png
├── shaplocal1.png
├── shaplocal2.png
├── shaplocal3.png
│
└── README.md
```

The repository currently contains both the source code and the resulting visualisations used to communicate the model interpretation results.

## Tools & Technologies

**Python**

- Python
- Jupyter Notebook
- Machine Learning
- Model Interpretability
- SHAP
- XAI

**R**

- R
- `mlr3`
- `mlr3learners`
- `mlr3pipelines`
- `xgboost`
- Random Forest
- `ggplot2`
- `dplyr`
- Cross-validation

## Model Evaluation

The project evaluates predictive performance using metrics appropriate to the modelling task.

For classification:

- ROC-AUC
- Accuracy

For regression:

- RMSE
- \(R^2\)

Additional analysis includes:

- Prediction vs. actual values
- Residual analysis
- Learning curves
- Comparison against baseline models

The three-stage model explicitly compares its RMSE against a mean-prediction baseline to determine whether the modelling approach provides meaningful predictive improvement.

## Key Questions Investigated

The project uses interpretability techniques to answer questions such as:

- Which variables have the greatest influence on model predictions?
- How does each feature affect predicted claim costs?
- Are important features behaving consistently across observations?
- Why does the model make a particular prediction?
- How does predictive performance change as more training data becomes available?
- Can a more interpretable modelling structure improve understanding of insurance claim costs?

## Getting Started

### R

Install R and the required packages used by the project. The main R scripts can then be opened and executed in RStudio or another R environment.

Typical packages include:

```r
install.packages(c(
  "dplyr",
  "ggplot2",
  "mlr3",
  "mlr3pipelines",
  "mlr3learners",
  "xgboost",
  "ranger"
))
```

Then run the relevant R script:

```r
source("3 stage model.R")
```

or:

```r
source("IML project - version 5.R")
```

### Python

Open the Jupyter notebook:

```bash
jupyter notebook Interpretable_ML.ipynb
```

Run the notebook cells sequentially to reproduce the analysis and visualisations.

## Results

The repository includes the main visual outputs generated during the analysis, including:

- Feature importance
- SHAP summary explanations
- Local SHAP explanations
- Feature effect plots
- ICE plots
- Learning curves
- Prediction error plots
- Residual plots

These visualisations provide complementary perspectives on both **model performance** and **model behaviour**.

## Project Context

This project was developed as part of an **Interpretable Machine Learning** project, with an emphasis on combining predictive performance with transparency and understanding.

Rather than treating the machine learning model as a black box, the analysis investigates the reasoning patterns behind its predictions using interpretable and explainable ML techniques.

## References

- Molnar, C. *Interpretable Machine Learning*.  
  https://christophm.github.io/interpretable-ml-book/

- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.

- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD.

## Author

**Mohammadhadi Shahhosseini**

MSc Data Science for Economics  
University of Milan
University of Copenhagen

GitHub: [@mohmmadhadi](https://github.com/mohmmadhadi)

## License

This project is intended for educational and research purposes.
