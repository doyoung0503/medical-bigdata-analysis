# Sleep Health Statistical Analysis Report

## 1. Analysis rationale and sequence

The reference materials were translated into a practical analysis flow for this dataset.

1. Week 3 emphasizes cross-tabulation and one-way ANOVA for checking whether group-level differences exist before modeling.
2. Week 4 emphasizes Pearson correlation, regression interpretation, and VIF-based multicollinearity checks.
3. Week 5 extends the workflow to logistic regression for binary clinical outcomes.

Following that logic, this analysis used the sequence below:

1. Descriptive statistics to confirm the distribution of sleep disorder labels and the available variable types.
2. Group-comparison tests to identify which features differ across `None`, `Insomnia`, and `Sleep Apnea`.
3. Correlation analysis to quantify linear associations with sleep disorder and detect redundancy among predictors.
4. Logistic regression to build an interpretable classification model for `Has Sleep Disorder`.

## 2. Dataset inventory and preparation

The `dataset` folder contains two different resources:

1. `Sleep_health_and_lifestyle_dataset`: person-level tabular data with an explicit `Sleep Disorder` label
2. `FitBit Fitness Tracker Data`: device logs for activity and sleep minutes, but without a direct `Sleep Disorder` target and without a person-level linkage to the first dataset

Because the user asked for correlation with **sleep disorder** and for a **logistic regression classifier**, the main inferential analysis was performed on `Sleep_health_and_lifestyle_dataset`. The Fitbit files were reviewed as contextual sleep/activity logs, but they are not suitable for supervised sleep-disorder classification in this workspace as provided.

## 3. Data preparation

- Source file: `dataset/Sleep_health_and_lifestyle_dataset/Sleep_health_and_lifestyle_dataset.csv`
- Total rows: 374
- Distinct profiles excluding `person_id`: 132
- Repeated profiles excluding `person_id`: 242
- Positive class rate for `Has Sleep Disorder`: 0.414
- `Sleep Disorder` missing values were treated as `"None"` because the raw CSV stores non-disorder cases in that form.
- `Blood Pressure` was split into `Systolic BP` and `Diastolic BP`.
- `BMI Category` was harmonized by merging `Normal Weight` into `Normal`.

Sleep disorder class balance:

| sleep_disorder | count | share |
| --- | --- | --- |
| None | 219 | 0.586 |
| Insomnia | 77 | 0.206 |
| Sleep Apnea | 78 | 0.209 |

## 4. Group-comparison results

### 4.1 Numeric features: one-way ANOVA

The strongest numeric group differences were:

| feature_label | f_stat | p_value | eta_squared |
| --- | --- | --- | --- |
| Diastolic BP | 268.098 | <0.001 | 0.591 |
| Systolic BP | 214.585 | <0.001 | 0.536 |
| Age | 58.409 | <0.001 | 0.239 |
| Physical Activity Level | 44.151 | <0.001 | 0.192 |
| Heart Rate | 32.949 | <0.001 | 0.151 |

Interpretation:

- Larger `eta_squared` means the feature separates sleep disorder groups more clearly.
- In this dataset, blood pressure, age, physical activity, and heart rate showed the largest group separation, while sleep duration and sleep quality were also statistically important.

### 4.2 Categorical features: chi-square test

| feature_label | chi2 | p_value | cramers_v |
| --- | --- | --- | --- |
| Occupation | 421.363 | <0.001 | 0.751 |
| BMI Category | 245.665 | <0.001 | 0.573 |
| Gender | 54.306 | <0.001 | 0.381 |

Interpretation:

- `BMI Category`, `Occupation`, and `Gender` all show univariate association with the disorder groups.
- `Occupation` was excluded from the predictive model because several job levels are too sparse for stable coefficient estimation.
- `BMI Category` and `Gender` were considered during modeling, but they did not survive the final multivariable reduction step.

## 5. Correlation analysis

Point-biserial correlations with `Has Sleep Disorder`:

| feature_label | correlation | p_value |
| --- | --- | --- |
| Diastolic BP | 0.705 | <0.001 |
| Systolic BP | 0.692 | <0.001 |
| Age | 0.432 | <0.001 |
| Sleep Duration | -0.339 | <0.001 |
| Heart Rate | 0.330 | <0.001 |

Interpretation:

- Negative coefficients mean the variable tends to be lower in the disorder group.
- Positive coefficients mean the variable tends to be higher in the disorder group.
- The largest practical signals came from shorter sleep duration, poorer sleep quality, higher stress, higher heart rate, and higher blood pressure.

## 6. Logistic regression for sleep-disorder classification

### 6.1 Variable selection logic

Variables were considered for the final logistic model only if they met the following conditions:

1. They showed evidence of group differences or binary association in the earlier statistical tests.
2. They were clinically interpretable for sleep disorder screening.
3. They did not create problematic sparsity or unstable multicollinearity.

Final variables used in the logistic model:

- Age
- Diastolic BP
- Quality of Sleep

VIF check for the final model:

| feature_label | vif |
| --- | --- |
| Age | 2.832 |
| Diastolic BP | 2.223 |
| Quality of Sleep | 1.856 |

All final VIF values were below 5, so multicollinearity is not a practical concern in the retained model.

### 6.2 Logistic regression coefficients

| feature_label | odds_ratio | ci_low | ci_high | p_value |
| --- | --- | --- | --- | --- |
| Quality of Sleep | 0.211 | 0.139 | 0.322 | <0.001 |
| Diastolic BP | 1.379 | 1.242 | 1.530 | <0.001 |
| Age | 1.175 | 1.094 | 1.261 | <0.001 |

Interpretation:

- The strongest risk-increasing feature in the final model was **Diastolic BP** with OR 1.38.
- The strongest protective feature in the final model was **Quality of Sleep** with OR 0.21.
- Odds ratios above 1 indicate higher odds of having a sleep disorder; below 1 indicate lower odds.

### 6.3 Classification performance

- 5-fold cross-validated ROC-AUC: 0.945 +/- 0.020
- Test ROC-AUC: 0.971
- Test accuracy: 0.903
- Test precision: 0.875
- Test recall: 0.894
- Test F1-score: 0.884

## 7. Final insights

### 7.1 Features associated with sleep disorder

This analysis consistently suggests that sleep disorder is most closely related to:

1. Reduced sleep duration
2. Lower quality of sleep
3. Higher stress level
4. Elevated heart rate
5. Higher blood pressure
6. Less favorable BMI category

These variables were the main univariate signals across ANOVA, chi-square, and correlation screening. After overlapping information was accounted for in the multivariable logistic model, the most stable retained predictors were `Age`, `Diastolic BP`, and `Quality of Sleep`.

### 7.2 Which variables should be used for logistic regression?

For an interpretable screening-oriented logistic regression model, the recommended variables are:

- Age
- Diastolic BP
- Quality of Sleep

Why these variables:

- They showed statistical evidence of association with sleep disorder.
- They offered clinically understandable directionality.
- They avoided occupation-like sparse categories that can make the model unstable.

### 7.3 Practical value

Based on these findings, the dataset supports a useful screening insight:

- People with shorter and poorer sleep, higher stress, and worse cardio-metabolic signals are more likely to belong to the sleep-disorder group.
- A lightweight logistic screening model can help prioritize who may benefit from additional sleep assessment, lifestyle coaching, or medical follow-up.
- In practice, the model can support early triage, personalized sleep-health feedback, and risk-based monitoring when full polysomnography or specialist evaluation is not immediately available.

## 8. Figures generated

1. `results/figures/01_sleep_disorder_distribution.png`
2. `results/figures/02_numeric_correlation_heatmap.png`
3. `results/figures/03_anova_effect_sizes.png`
4. `results/figures/04_group_boxplots.png`
5. `results/figures/05_categorical_effect_sizes.png`
6. `results/figures/06_categorical_panels.png`
7. `results/figures/07_vif_plot.png`
8. `results/figures/08_logistic_odds_ratios.png`
9. `results/figures/09_model_performance.png`

## 9. Caution

- This dataset contains many repeated profiles aside from `person_id`, which suggests a templated or synthetic structure. The direction of findings is still useful, but p-values may look stronger than they would in a more heterogeneous real-world cohort.
- The logistic model predicts presence of any sleep disorder, not the subtype distinction between insomnia and sleep apnea.
