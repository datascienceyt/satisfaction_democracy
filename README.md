# Satisfaction with Democracy in Ecuador: Machine Learning Analysis

This repository contains a machine learning workflow to analyze and predict satisfaction with democracy in Ecuador using Latinobarómetro survey data. The project focuses on binary classification and compares model performance across three historical periods.

## Project Objective

The study has two main objectives:

1. **Predictive evaluation**: evaluate the performance of different classification models for predicting satisfaction with democracy in Ecuador.
2. **Predictor interpretation**: identify and interpret the most important predictors associated with satisfaction with democracy across historical periods.

The target variable is `ETIQUETA`:

- `1`: Satisfied with democracy
- `0`: Dissatisfied with democracy

The analysis is restricted to Ecuadorian survey records using `X_23 = 218`.

## Historical Periods

The analysis compares three historical periods:

| Period | Years | Description |
|---|---:|---|
| `1_Pre_Correismo` | 1995–2006 | Period before Rafael Correa's government |
| `2_Correismo` | 2007–2017 | Rafael Correa's government period |
| `3_Post_Correismo` | 2018–2023 | Period after Rafael Correa's government |

These periods are used as an analytical framework to compare predictive patterns across political contexts. They should not be interpreted as a causal identification strategy.

## Dataset

The dataset comes from Latinobarómetro and includes political attitudes, institutional trust, economic perceptions, material conditions, and sociodemographic variables.

Main variable groups:

- **Political attitudes**: support for democracy, left-right self-positioning.
- **Economic perceptions**: perception of national economic situation.
- **Institutional trust**: trust in Congress, Judiciary, Church, Police, Armed Forces, and Political Parties.
- **Material conditions**: access to household goods and services.
- **Sociodemographic characteristics**: sex, age, education, employment situation, socioeconomic status, religion.

Some variables are excluded from the baseline models:

- `A_3`: political party vote intention; high cardinality and historically changing categories.
- `X_23`: country code; constant after filtering Ecuador.
- `X_25`: region/geographical area; high cardinality.

The survey year variable `X_24` is used to create historical periods but is not included as a predictor in the baseline models.

## Modeling Strategy

The notebook is organized into three classification experiments.

### Experiment 1: Class Weight Grid

This experiment evaluates:

- Logistic Regression
- Decision Tree
- Random Forest

Class imbalance is handled through `class_weight` as part of the hyperparameter grid. Therefore, models with and without class weighting are compared automatically through `GridSearchCV`.

### Experiment 2: RandomOverSampler

This experiment evaluates the same model families using `RandomOverSampler`. The minority class is oversampled only within the training folds during cross-validation using `imblearn.pipeline.Pipeline`, avoiding data leakage.

### Experiment 3: SMOTENC

This experiment evaluates `SMOTENC`, a synthetic oversampling method designed for datasets with both numeric and categorical features.

For SMOTENC:

- `S_17` is treated as the continuous numeric variable.
- Binary, ordinal, and nominal survey-coded variables are treated as categorical features.

The test set remains untouched in all experiments.

## Model Selection and Evaluation

The main selection metric is:

```python
REFIT_METRIC = "f1_macro"
```

This metric is used because the target variable is imbalanced and both classes are relevant.

The following metrics are reported:

- Accuracy
- Balanced accuracy
- Precision for the positive class (`Satisfied = 1`)
- Recall for the positive class (`Satisfied = 1`)
- F1-score for the positive class (`Satisfied = 1`)
- Macro F1-score
- ROC-AUC

The test set is used only for final evaluation. Hyperparameter tuning is performed through stratified cross-validation on the training set.

## Final Model Comparison

After running the three experiments, results are combined into a single comparison table. The best final model is selected for each historical period according to `f1_macro`.

The current final selection was:

| Period | Final selected model | Strategy |
|---|---|---|
| Pre-Correismo | Random Forest + RandomOverSampler | RandomOverSampler |
| Correismo | Random Forest + SMOTENC | SMOTENC |
| Post-Correismo | Random Forest | Class weight grid |

The results indicate that no single imbalance-handling strategy consistently dominates across all periods. The effect of class balancing is period-dependent.

## Predictor Importance Analysis

The project includes a predictor importance analysis to support interpretation.

Three complementary methods are considered:

### 1. Permutation Importance

Permutation importance is used as the main interpretation method. It measures how much `f1_macro` decreases when a predictor is randomly shuffled in the test set.

This method is model-agnostic and directly linked to the selected evaluation metric.

### 2. Tree-Based Feature Importance

Tree-based feature importance is computed for the best Random Forest model in each period. This is included as a complementary method because it is commonly used in machine learning and ICT studies.

It measures the internal contribution of each feature to impurity reduction in tree-based models.

### 3. Logistic Regression Coefficients

The best Logistic Regression model in each period is used as an interpretable linear reference. Coefficients help inspect the direction and magnitude of linear associations with the positive class, `Satisfied = 1`.

All interpretation results should be understood as predictive associations, not causal effects.

## Main Predictor Importance Findings

Using permutation importance, the top predictors by period were:

### Pre-Correismo

- (A COMPLETAR)

### Correismo

- (A COMPLETAR)
### Post-Correismo

- (A COMPLETAR)

## Notebook Structure

The notebook is organized as follows:

```text
# Project Overview and Research Objective

# PART 1 — Environment and Configuration
## Environment and Libraries
## Package Versions
## General Configuration

# PART 2 — Data Loading and Preparation
## Load Dataset
## Dataset Documentation
## Working Copy
## Survey Year Correction
## Filter Ecuador
## Exclude Variables Not Used in the Baseline Model

# PART 3 — Data Quality and Descriptive Analysis
## Initial Data Quality Review
## Expected Value Validation
## Type Conversion
## Conceptual Feature Groups
## Historical Period Definition
## Target Distribution by Historical Period
## Temporal Trend of Satisfaction and Dissatisfaction

# PART 4 — Shared Modeling Setup
## Period-Specific Dataset Preparation
## Period-Specific Train/Test Split
## Evaluation Metrics
## Base Preprocessing Pipeline
## Model Evaluation Function

# PART 5 — Experiment 1: Class Weight Grid
## Class Weight Grid Model Configuration
## Class Weight Grid Training Function
## Run Class Weight Grid Models by Historical Period
## Class Weight Grid Results Summary Table
## Best Class Weight Grid Model by Historical Period

# PART 6 — Experiment 2: RandomOverSampler
## RandomOverSampler Model Configuration
## RandomOverSampler Training Function
## Run RandomOverSampler Models by Historical Period
## RandomOverSampler Results Summary Table
## Best RandomOverSampler Model by Historical Period

# PART 7 — Experiment 3: SMOTENC
## SMOTENC Feature Groups
## Pre-SMOTENC Imputation
## Post-SMOTENC Preprocessing
## SMOTENC Model Configuration
## SMOTENC Training Function
## Run SMOTENC Models by Historical Period
## SMOTENC Results Summary Table
## Best SMOTENC Model by Historical Period

# PART 8 — Comparison of All Experiments
## Full Model Comparison Across Experiments
## Final Selected Models by Historical Period
## Best Model Performance by Experiment and Historical Period
## Best Final Model Performance by Historical Period

# PART 9 — Predictor Importance Analysis
## Permutation Importance for Final Models
## Tree-Based Feature Importance
## Logistic Regression Coefficients
## Comparison of Predictor Importance Methods
## Interpretation of Predictor Importance
```

## Recommended Environment

The experiments can be computationally expensive because they combine multiple periods, models, hyperparameter grids, cross-validation folds, and imbalance-handling strategies.

Recommended setup:

- Python 3.12 or compatible
- CPU-based execution is sufficient
- GPU is not required for the current scikit-learn and imbalanced-learn workflow
- 32 GB RAM recommended
- 64 GB RAM preferred for larger grids and SMOTENC
- Multi-core CPU recommended

For faster execution during development, use reduced grids or fewer cross-validation folds. For final results, use the full grid and keep the random seed fixed for reproducibility.

## Python Dependencies

The main dependencies are:

```text
pandas
numpy
matplotlib
scikit-learn
imbalanced-learn
openpyxl
```

If Markdown tables are exported from pandas, `tabulate` may also be useful:

```text
tabulate
```

A reproducible `requirements.txt` should pin package versions used in the final run.

## Methodological Notes

- The analysis is predictive, not causal.
- The historical periodization is used for comparison, not causal identification.
- The target is imbalanced, so `f1_macro` and `balanced_accuracy` are prioritized over accuracy.
- Oversampling is performed only inside cross-validation folds using `imblearn.pipeline.Pipeline`.
- The held-out test set is never oversampled.
- Feature importance results indicate predictive relevance, not causal effects.
- Permutation importance values represent decreases in `f1_macro`; they are not probabilities or percentages.
- Negative permutation importance values indicate that shuffling a variable slightly improved performance, usually suggesting noise, instability, or negligible predictive contribution.

