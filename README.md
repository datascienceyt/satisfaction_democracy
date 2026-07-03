# Satisfaction with Democracy in Ecuador

This repository contains a reproducible machine learning workflow for analyzing satisfaction with democracy in Ecuador using Latinobarómetro survey data. The study evaluates classification performance across three temporal periods and interprets the most relevant predictors associated with satisfaction or dissatisfaction with democracy.

The analysis is implemented in the notebook:

```text
ecuador_satisfaccion_democracia.ipynb
```

## Project Objective

This study has two main objectives:

1. To evaluate the predictive performance of different classification models across three temporal periods.
2. To identify the most relevant predictors associated with satisfaction with democracy in Ecuador, focusing on political attitudes, institutional trust, economic perceptions, material conditions, labor conditions, and sociodemographic characteristics.

The target variable is `ETIQUETA`, where:

```text
0 = Dissatisfied with democracy
1 = Satisfied with democracy
```

The study is predictive and interpretative. The results should be understood as predictive associations, not causal effects.

---

## Data Source and Study Scope

The analysis uses Latinobarómetro survey data. The unit of analysis is the individual survey respondent.

The dataset is filtered to Ecuador using the country code:

```text
X_23 = 218
```

After filtering Ecuador, the working dataset contains:

```text
17,902 observations
```

The original dataset contains multiple countries and survey years. The analysis focuses only on Ecuadorian respondents.

---

## Survey Year Harmonization

The survey year variable is `X_24`. Some year values required harmonization before defining temporal periods. The notebook applies a year correction mapping to convert specific encoded values into actual calendar years:

```python
YEAR_MAPPING = {
    16: 2011,
    17: 2013,
    18: 2015,
    23: 2023,
}
```

After correction, the available survey years for the analysis include non-continuous Latinobarómetro survey years. Not every calendar year is available.

---

## Temporal Period Definition

The analysis uses three temporal periods defined from visible changes in the temporal distribution of satisfaction and dissatisfaction with democracy.

The current period definitions are:

```text
First period: 1996–2003
Second period: 2004–2013
Third period: 2014–2024
```

However, there are no available Ecuadorian Latinobarómetro observations for 2014. Therefore, the third period is conceptually defined as 2014–2024, but the observed survey years for Ecuador in this interval start in 2015.

The empirical analysis is therefore based on:

```text
First period: observed years 1996–2003
Second period: observed years 2004–2013
Third period: observed years 2015–2024
```

The period labels used in the notebook are neutral:

```text
1_Period = First period
2_Period = Second period
3_Period = Third period
```

No political labels are used in the final period naming.

---

## Target Variable Distribution

The target variable is imbalanced, with more respondents dissatisfied than satisfied with democracy.

Overall Ecuadorian distribution:

```text
Dissatisfied (0): 11,740 observations, 65.58%
Satisfied (1):    6,162 observations, 34.42%
```

Distribution by period:

| Period | Dissatisfied (0) | Satisfied (1) | Total | % Dissatisfied | % Satisfied |
|---|---:|---:|---:|---:|---:|
| First period | 4,017 | 1,505 | 5,522 | 72.75% | 27.25% |
| Second period | 4,051 | 2,736 | 6,787 | 59.69% | 40.31% |
| Third period | 3,672 | 1,921 | 5,593 | 65.65% | 34.35% |

The class imbalance motivates the use of macro F1-score, balanced accuracy, class weighting, and resampling methods.

---

## Variable Exclusion

The following variables are excluded from the baseline predictive model:

```text
A_3  = Party vote intention
X_23 = Country code
X_25 = Region / geographical area
```

The reasons are:

- `A_3` is excluded due to high cardinality.
- `X_23` is excluded because it is constant after filtering Ecuador.
- `X_25` is excluded due to high cardinality.

---

## Predictor Set

The final predictor set contains 21 variables.

### Binary predictors

```text
C_4
D_6, D_7, D_8, D_9
S_16, S_18
```

### Ordinal predictors

```text
A_2
D_5
H_10, H_11, H_12, H_13, H_14, H_15
S_19
S_21
```

### Numeric predictors

```text
S_17
```

### Nominal predictors

```text
A_1
S_20
S_22
```

---

## Conceptual Predictor Groups

For interpretation, predictors are grouped into substantive dimensions:

```text
Political attitudes
Institutional trust
Economic perceptions
Material conditions
Labor and economic conditions
Sociodemographic characteristics
```

These groups are used to interpret how different dimensions contribute to the classification of satisfaction with democracy across periods.

---

## Data Quality Assessment

The notebook performs several data quality checks:

```text
- Dataset dimensions
- Missing values
- Expected value validation
- Data type conversion
- Duplicate row inspection
```

No missing values were detected in the selected variables.

The notebook identified 314 duplicated rows. These rows were inspected but not removed.

Duplicated rows are defined by pandas as rows where all values are identical to a previous row. In survey data, this does not necessarily imply administrative duplication. Since many variables are categorical or ordinal, different respondents may have identical response profiles. Therefore, duplicated response profiles were retained in the analysis.

Recommended methodological statement:

```text
Duplicate rows were inspected as part of the data quality assessment. Because the dataset consists of survey responses coded mostly as categorical or ordinal variables, identical rows may correspond to different respondents with the same response profile rather than true administrative duplicates. Therefore, duplicated response profiles were retained in the analysis.
```

---

## Preprocessing Pipeline

The notebook uses separate preprocessing strategies depending on the model type.

### Logistic Regression preprocessing

```text
Numeric variables:
median imputation + standardization

Ordinal variables:
most frequent imputation + standardization

Binary variables:
most frequent imputation

Nominal variables:
most frequent imputation + one-hot encoding
```

Scaling is applied to numeric and ordinal variables because Logistic Regression is sensitive to feature scale.

### Tree-based model preprocessing

```text
Numeric variables:
median imputation

Ordinal variables:
most frequent imputation

Binary variables:
most frequent imputation

Nominal variables:
most frequent imputation + one-hot encoding
```

Scaling is not required for Decision Tree or Random Forest models.

---

## Period-Specific Modeling Strategy

Models are trained independently for each temporal period. This avoids mixing observations from different temporal contexts and allows the study to compare predictive performance and predictor relevance across periods.

For each period, the workflow is:

```text
1. Filter the dataset to the current period.
2. Separate predictors and target.
3. Apply a stratified train/test split.
4. Train models using GridSearchCV on the training set.
5. Select hyperparameters using macro F1-score.
6. Evaluate the selected model once on the held-out test set.
```

The train/test split uses:

```text
80% training
20% testing
stratified by the target variable
random_state = 42
```

The held-out test set is not used during hyperparameter tuning.

---

## Classification Models

The study evaluates three classification models:

```text
Logistic Regression
Decision Tree
Random Forest
```

These models provide different levels of complexity and interpretability:

- Logistic Regression provides a linear and interpretable baseline.
- Decision Tree captures non-linear decision rules and is easy to interpret.
- Random Forest captures more complex non-linear patterns and interactions.

---

## Class Imbalance Strategies

The notebook evaluates three strategies to handle class imbalance.

### 1. Class Weight Grid

Class weighting does not modify the training data. Instead, it adjusts the penalty assigned to each class during model training.

This strategy allows the model to assign more importance to the minority class without creating or duplicating observations.

### 2. RandomOverSampler

RandomOverSampler balances the training data by randomly duplicating observations from the minority class.

It preserves real observations but may increase the risk of overfitting because minority-class cases are repeated.

### 3. SMOTENC

SMOTENC generates synthetic minority-class observations and is designed for datasets containing both numerical and categorical predictors.

In this notebook, age (`S_17`) is treated as numeric, while the remaining predictors are treated as categorical for SMOTENC.

SMOTENC introduces more variability than RandomOverSampler, but synthetic observations may not always fully represent real survey response patterns.

All resampling methods are applied only within the training folds using imbalanced-learn pipelines. The validation and test sets remain unchanged, avoiding data leakage.

---

## Hyperparameter Tuning

Hyperparameter tuning is performed using:

```text
GridSearchCV
StratifiedKFold cross-validation
refit metric = f1_macro
```

The main selection metric is macro F1-score because it gives equal weight to both classes and is more appropriate than accuracy under class imbalance.

---

## Evaluation Metrics

The notebook computes the following evaluation metrics:

```text
accuracy
balanced_accuracy
precision
recall
f1
f1_macro
roc_auc
confusion matrix
classification report
```

Important note:

```text
precision, recall, and f1 are computed for the positive class, Satisfied (1).
f1_macro averages F1-score across both classes.
```

The main metric for model selection is:

```text
f1_macro
```

---

## Final Model Selection

The best final model for each period is selected according to macro F1-score on the held-out test set.

Final selected models:

| Period | Best model | Imbalance strategy | f1_macro | Accuracy | ROC-AUC |
|---|---|---|---:|---:|---:|
| First period | Decision Tree | Class Weight Grid | 0.5967 | 0.6796 | 0.5969 |
| Second period | Logistic Regression | Class Weight Grid | 0.6866 | 0.7121 | 0.7448 |
| Third period | Random Forest | Class Weight Grid | 0.7206 | 0.7444 | 0.7873 |

Across the three temporal periods, the best final models were obtained using the class-weight strategy. In the first period, the best model was a Decision Tree trained with the class weight grid. In the second period, the best model was Logistic Regression with class weights. In the third period, the best model was Random Forest with class weights.

This result indicates that adjusting the contribution of each class during model training was more effective than modifying the training distribution through oversampling. Although RandomOverSampler and SMOTENC produced competitive results in some periods, neither strategy outperformed the best class-weighted model in the final period-specific selection.

A concise result statement is:

```text
The final model selection shows that the class-weight strategy produced the best-performing model in all three periods. This suggests that, for this dataset, adjusting class penalties during model training was more effective than generating additional minority-class observations through RandomOverSampler or SMOTENC.
```

---

## Predictor Importance and Interpretability Analysis

The interpretation follows a hierarchical approach.

### Main interpretation method: permutation importance

Permutation importance is used as the main interpretability method because it is computed on the final selected model for each period and directly measures the decrease in predictive performance after randomly shuffling each predictor.

Since the main evaluation metric is macro F1-score, permutation importance is aligned with the study's model selection criterion.

The values represent the average decrease in macro F1-score after randomly permuting each predictor. They should not be interpreted as probabilities or percentages.

Example interpretation:

```text
If a predictor has permutation importance of 0.0932, it means that randomly shuffling that predictor reduced macro F1-score by approximately 0.0932 points on average.
```

Higher values indicate greater predictive contribution. Values close to zero indicate little contribution. Negative values indicate that shuffling the variable slightly improved performance, which may reflect noise, instability, or redundancy.

### Complementary interpretation methods

Tree-based feature importance and Logistic Regression coefficients are used as complementary methods.

```text
Permutation importance:
computed on the final selected model for each period.

Tree-based feature importance:
computed on the best Random Forest model for each period.

Logistic Regression coefficients:
computed on the best Logistic Regression model for each period.
```

Tree-based feature importance measures internal impurity reduction in Random Forest models. Logistic Regression coefficients provide a linear and directional reference for the positive class, `Satisfied (1)`.

The main substantive interpretation is based on permutation importance. Predictors that also appear as relevant in tree-based importance or Logistic Regression coefficients provide additional supporting evidence.

Recommended methodological statement:

```text
The main predictor interpretation is based on permutation importance, computed on the final selected model for each period. This method directly quantifies the decrease in macro F1-score when each predictor is randomly permuted. Tree-based feature importance and Logistic Regression coefficients are reported as complementary analyses.
```

---

## Main Predictor Importance Results

### Permutation importance: top predictors by period

#### First period

```text
Trust in Political Parties: 0.0464
Age: 0.0419
Support for democracy: 0.0409
Trust in Police: 0.0302
Trust in Armed Forces: 0.0214
```

#### Second period

```text
Perception of national economy: 0.0830
Trust in Congress: 0.0529
Support for democracy: 0.0212
Trust in Political Parties: 0.0180
Trust in Armed Forces: 0.0099
```

#### Third period

```text
Perception of national economy: 0.0932
Support for democracy: 0.0327
Trust in Congress: 0.0145
Trust in Judiciary: 0.0131
Trust in Political Parties: 0.0105
```

The main pattern is that economic perception becomes the dominant predictor in the second and third periods, while institutional trust and political attitudes remain consistently relevant.

---

## Predictor Importance by Conceptual Group

Permutation importance was also aggregated by conceptual predictor group.

### First period

| Conceptual group | Total importance |
|---|---:|
| Institutional trust | 0.1486 |
| Sociodemographic characteristics | 0.1087 |
| Political attitudes | 0.0579 |
| Economic perceptions | 0.0147 |
| Material conditions | 0.0118 |
| Labor and economic conditions | -0.0020 |

### Second period

| Conceptual group | Total importance |
|---|---:|
| Institutional trust | 0.0877 |
| Economic perceptions | 0.0830 |
| Political attitudes | 0.0212 |
| Sociodemographic characteristics | 0.0094 |
| Material conditions | 0.0061 |
| Labor and economic conditions | 0.0000 |

### Third period

| Conceptual group | Total importance |
|---|---:|
| Economic perceptions | 0.0932 |
| Institutional trust | 0.0531 |
| Political attitudes | 0.0347 |
| Sociodemographic characteristics | 0.0197 |
| Material conditions | 0.0050 |
| Labor and economic conditions | 0.0009 |

Main interpretation:

```text
Institutional trust dominates the first period, while economic perceptions dominate the third period. The second period shows a transition where institutional trust and economic perception have similar total predictive importance.
```

---

## Key Findings

The main findings from the executed notebook are:

1. Class weighting produced the best final model in all three periods.
2. The best model type varied by period: Decision Tree in the first period, Logistic Regression in the second period, and Random Forest in the third period.
3. Predictive performance improved across periods, with the highest macro F1-score in the third period.
4. Institutional trust was the most important conceptual group in the first period.
5. Economic perceptions became the dominant predictor group in the third period.
6. The second period shows a transition where institutional trust and economic perceptions have similar predictive relevance.
7. Support for democracy and institutional trust variables remain relevant across periods.
8. Results should be interpreted as predictive associations, not causal effects.

---

