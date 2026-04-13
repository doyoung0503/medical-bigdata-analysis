# 수면장애 9단계 실험 결과 시각화 문서

## 1. 문서 목적

이 문서는 이번 프로젝트에서 수행한 **9단계 실험 전체 결과를 실제 그래프와 표 중심으로 다시 정리한 문서**다.

각 단계마다 아래 순서로 구성했다.

1. 왜 이 실험을 했는가
2. 어떤 방법을 썼는가
3. 핵심 결과 표
4. 실제 시각화 자료
5. 무엇을 해석할 수 있는가

즉, 이 문서는 설명용 요약이 아니라 **결과를 직접 눈으로 확인하면서 해석할 수 있는 시각화 중심 문서**다.

---

## 2. 단계 개요

1. 데이터셋 선택과 기초 통계분석
2. 상관관계와 다중공선성 확인
3. 이진 로지스틱 회귀 초기 모델
4. Ridge 로지스틱 회귀와 파생변수 탐색
5. 다항 로지스틱 회귀와 민감도 분석
6. 후속 엄밀성 검증
7. 비판 포인트 해소를 위한 추가 검증
8. grouped bootstrap 모델 비교와 변수 안정성 후속 검증
9. exact-row dedup sensitivity와 nested grouped bootstrap 최종 보강

---

## 3. 1단계: 데이터셋 선택과 기초 통계분석

### 왜 이 실험을 했는가

먼저 어떤 데이터가 실제 분석에 적합한지, 그리고 수면장애 분포와 기초 집단 차이가 어떻게 생겼는지 파악해야 이후 분석의 출발점이 안정된다.

### 사용한 방법과 원리

- 기술통계
- 클래스 분포 시각화
- 집단별 박스플롯
- 범주형 분포 패널

원리는 단순하다. 분포, 평균, 범주 비율을 먼저 보면 이후 검정과 회귀 해석의 방향을 훨씬 안정적으로 잡을 수 있다.

### 핵심 결과 표

| 수면장애 범주 | count | share |
| --- | ---: | ---: |
| None | 219 | 0.586 |
| Insomnia | 77 | 0.206 |
| Sleep Apnea | 78 | 0.209 |

### 대표 시각화

![수면장애 분포](../figures/01_sleep_disorder_distribution.png)

![집단별 수치형 변수 박스플롯](../figures/04_group_boxplots.png)

### 추가 시각화

<details>
<summary>1단계의 나머지 시각화 보기</summary>

![ANOVA 효과크기](../figures/03_anova_effect_sizes.png)

![범주형 효과크기](../figures/05_categorical_effect_sizes.png)

![범주형 패널](../figures/06_categorical_panels.png)

</details>

### 해석

- `None`이 약 58.6%로 가장 크지만, `Insomnia`와 `Sleep Apnea`도 각각 약 20% 수준이라 subtype 분석이 가능한 구조였다.
- 박스플롯만 봐도 혈압, 나이, 수면시간, 수면의 질, 활동량이 집단별로 다르게 움직이는 신호가 보였다.
- 이 단계는 “무슨 변수가 중요할 것 같은가”를 잡는 탐색 단계였다.

---

## 4. 2단계: 상관관계와 다중공선성 확인

### 왜 이 실험을 했는가

집단 차이가 보여도 변수들이 서로 강하게 얽혀 있으면 회귀계수 해석이 흔들릴 수 있다. 그래서 본격적인 모델링 전에 공선성 구조를 확인했다.

### 사용한 방법과 원리

- 상관행렬
- 높은 상관쌍 시각화
- VIF 비교

상관분석은 변수들이 함께 움직이는 정도를, VIF는 한 변수가 나머지 변수들로 얼마나 설명되는지를 보여준다.

### 대표 시각화

![원 변수 상관행렬](../ridge_feature_study/figures/01_original_correlation_heatmap.png)

![높은 상관쌍](../ridge_feature_study/figures/02_high_correlation_pairs.png)

![VIF 비교](../ridge_feature_study/figures/03_vif_comparison.png)

### 해석

- `Systolic BP`와 `Diastolic BP`가 매우 강하게 결합돼 있었다.
- `Sleep Duration`, `Quality of Sleep`, `Stress Level`도 같은 축 안에 묶여 보였다.
- 이 결과는 일반 로지스틱만으로 끝내지 않고, Ridge와 파생변수 모델까지 봐야 할 이유를 제공했다.

---

## 5. 3단계: 이진 로지스틱 회귀 초기 모델

### 왜 이 실험을 했는가

프로젝트의 직접적인 질문이 “수면장애가 있는가 없는가”였기 때문에, 먼저 해석 가능한 이진 로지스틱 회귀를 기준점으로 만들었다.

### 사용한 방법과 원리

- 이진 로지스틱 회귀
- Odds Ratio 시각화
- 기본 분류 성능 평가

로지스틱 회귀는 확률 자체가 아니라 `log-odds`를 선형식으로 모델링한다. 그래서 각 변수의 방향과 `OR`를 해석할 수 있다.

### 핵심 결과 표

| 변수 | Odds Ratio | 95% CI | p-value |
| --- | ---: | --- | ---: |
| Quality of Sleep | 0.211 | 0.139–0.322 | <0.001 |
| Diastolic BP | 1.379 | 1.242–1.530 | <0.001 |
| Age | 1.175 | 1.094–1.261 | <0.001 |

| 지표 | 값 |
| --- | ---: |
| Accuracy | 0.903 |
| Macro F1 | 0.900 |

### 대표 시각화

![초기 로지스틱 OR](../figures/08_logistic_odds_ratios.png)

![초기 모델 성능](../figures/09_model_performance.png)

### 해석

- 초기 해석형 모델 기준으로는 `Diastolic BP`, `Quality of Sleep`, `Age`가 가장 강한 변수였다.
- 특히 `Diastolic BP`는 이후 단계들에서도 계속 핵심 신호로 남았다.
- 다만 이 단계의 해석은 공선성을 충분히 처리하기 전 결과이므로, 최종 결론은 아니었다.

---

## 6. 4단계: Ridge 로지스틱 회귀와 파생변수 탐색

### 왜 이 실험을 했는가

공선성이 강한 상황에서는 일반 로지스틱보다 Ridge가 더 안정적일 수 있다. 동시에 임상적으로 해석 가능한 파생변수를 설계해 정보 압축 가능성을 확인했다.

### 사용한 방법과 원리

- Ridge 로지스틱 회귀
- 파생변수 스크리닝
- 원 변수 vs 압축형 vs 확장형 모델 비교

Ridge는 `L2 penalty`를 통해 상관된 변수들에 계수를 분산시켜 과도한 변동성을 줄인다.

### 핵심 결과 표

| 모델 | CV AUC | Test AUC | Accuracy | F1 | Best C |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.946 | 0.976 | 0.929 | 0.917 | 0.0001 |
| Compressed Derived | 0.945 | 0.970 | 0.938 | 0.928 | 0.0001 |
| Augmented | 0.941 | 0.977 | 0.938 | 0.926 | 0.0100 |

### 대표 시각화

![파생변수 스크리닝](../ridge_feature_study/figures/04_derived_feature_screening.png)

![Ridge 모델 비교](../ridge_feature_study/figures/06_model_performance_comparison.png)

![ROC 비교](../ridge_feature_study/figures/10_roc_curve_comparison.png)

### 추가 시각화

<details>
<summary>4단계의 나머지 시각화 보기</summary>

![원 변수와 파생변수 상관맵](../ridge_feature_study/figures/05_original_derived_correlation_map.png)

![Baseline Ridge 계수](../ridge_feature_study/figures/07_baseline_ridge_coefficients.png)

![Compressed Ridge 계수](../ridge_feature_study/figures/08_compressed_ridge_coefficients.png)

![Augmented Ridge 계수](../ridge_feature_study/figures/09_augmented_ridge_coefficients.png)

![혼동행렬 비교](../ridge_feature_study/figures/11_confusion_matrix_comparison.png)

</details>

### 해석

- 공선성이 있는 상황에서도 Ridge 모델들은 전반적으로 높은 AUC를 유지했다.
- `Compressed Derived`는 공선성 완화와 성능의 균형이 좋았다.
- 다만 “최종 추천 모델”을 바로 Ridge로 고정하기보다는, subtype과 민감도 분석을 추가로 본 뒤 판단하는 편이 더 타당했다.

---

## 7. 5단계: 다항 로지스틱 회귀와 민감도 분석

### 왜 이 실험을 했는가

이진 분류는 `Insomnia`와 `Sleep Apnea`를 하나로 묶기 때문에 subtype 차이를 놓칠 수 있다. 또한 중복 구조에 따라 성능이 얼마나 민감한지도 점검해야 했다.

### 사용한 방법과 원리

- 다항 로지스틱 회귀
- full vs deduplicated 성능 비교
- 계수 안정성 및 효과크기 민감도

다항 로지스틱은 기준 범주 대비 각 subtype의 log-odds를 따로 모델링한다.

### 핵심 결과 표

| 데이터 | CV Macro-F1 | Test Accuracy | Test Macro-F1 | Test Macro AUC |
| --- | ---: | ---: | ---: | ---: |
| Full | 0.858 | 0.920 | 0.898 | 0.933 |
| Deduplicated | 0.612 | 0.650 | 0.581 | 0.703 |

### 대표 시각화

![클래스 분포 Full vs Dedup](../multinomial_sensitivity_study/figures/01_class_balance_full_vs_dedup.png)

![subtype별 선택 변수](../multinomial_sensitivity_study/figures/02_selected_features_by_class.png)

![다항 로지스틱 OR](../multinomial_sensitivity_study/figures/03_multinomial_odds_ratios_full.png)

![성능 민감도](../multinomial_sensitivity_study/figures/05_multinomial_performance_sensitivity.png)

### 추가 시각화

<details>
<summary>5단계의 나머지 시각화 보기</summary>

![효과크기 민감도](../multinomial_sensitivity_study/figures/04_effect_size_sensitivity.png)

![계수 안정성 heatmap](../multinomial_sensitivity_study/figures/06_coefficient_stability_heatmap.png)

![혼동행렬 Full vs Dedup](../multinomial_sensitivity_study/figures/07_confusion_matrices_full_vs_dedup.png)

</details>

### 해석

- `Insomnia`와 `Sleep Apnea`는 서로 다른 변수 축을 보였다.
- dedup 후 성능은 낮아졌지만 핵심 방향성은 유지됐기 때문에, 이진 screening은 가능하되 subtype 해석은 별도로 가져가야 한다는 결론이 나왔다.

---

## 8. 6단계: 후속 엄밀성 검증

### 왜 이 실험을 했는가

기존 결과가 특정 검정 방식이나 느슨한 검증 설계에만 기대고 있지 않은지 확인하기 위해, 강건한 단변량 검정과 grouped validation, calibration, 진단을 한꺼번에 수행했다.

### 사용한 방법과 원리

- `Welch ANOVA`, `Kruskal-Wallis`, permutation chi-square
- grouped CV / grouped holdout bootstrap
- calibration, threshold sweep
- `Box-Tidwell`, 영향점 진단, bootstrap OR

### 대표 시각화

![수치형 강건성 heatmap](../rigorous_validation/figures/01_numeric_method_significance_heatmap.png)

![범주형 강건성](../rigorous_validation/figures/03_categorical_robustness.png)

![Random vs Grouped Validation](../rigorous_validation/figures/04_random_vs_grouped_validation.png)

![Calibration과 Threshold](../rigorous_validation/figures/06_calibration_and_thresholds.png)

![로지스틱 진단](../rigorous_validation/figures/07_logistic_diagnostics.png)

![Bootstrap OR](../rigorous_validation/figures/08_bootstrap_odds_ratios.png)

### 추가 시각화

<details>
<summary>6단계의 나머지 시각화 보기</summary>

![수치형 효과크기](../rigorous_validation/figures/02_numeric_effect_sizes.png)

![Grouped bootstrap metric CI](../rigorous_validation/figures/05_grouped_bootstrap_metric_cis.png)

![Grouped nested model comparison](../rigorous_validation/figures/09_grouped_nested_model_comparison.png)

</details>

### 해석

- 혈압, 나이, 수면시간, 수면의 질, 활동량 축은 강건한 검정에서도 유지됐다.
- grouped validation으로 바꾸면 성능이 조금 보수적으로 내려가지만 핵심 모델군은 유지됐다.
- 이 단계에서는 `Sleep + PA`가 가장 실전적인 기본형처럼 보였고, 파생변수 모델은 calibration을 더 봐야 한다는 결론이 나왔다.

---

## 9. 7단계: 비판 포인트 해소를 위한 추가 검증

### 왜 이 실험을 했는가

threshold가 single holdout에 너무 의존하는지, tuning objective에 따라 결론이 달라지는지, 그리고 clustered 추론과 subtype 검증을 더 보강할 필요가 있었다.

### 사용한 방법과 원리

- repeated grouped model comparison
- repeated grouped calibration / threshold 재검증
- profile-cluster GEE
- repeated grouped multinomial validation

### 핵심 결과 표

#### repeated grouped binary summary (`neg_log_loss`)

| 모델 | ROC-AUC | F1 | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| Original: Quality + HR | 0.938 | 0.898 | 0.075 | 0.082 |
| Original: Quality + PA | 0.936 | 0.895 | 0.075 | 0.085 |
| Original: Sleep + PA | 0.937 | 0.892 | 0.081 | 0.093 |

#### repeated grouped multinomial summary

| Metric | Mean | SD |
| --- | ---: | ---: |
| Accuracy | 0.880 | 0.042 |
| Macro-F1 | 0.849 | 0.055 |
| Macro-AUC OVR | 0.911 | 0.035 |

### 대표 시각화

![Repeated grouped model stability](../critical_resolution/figures/01_repeated_grouped_model_stability.png)

![Pairwise model differences](../critical_resolution/figures/02_pairwise_model_differences.png)

![Repeated calibration resolution](../critical_resolution/figures/03_repeated_calibration_resolution.png)

![Clustered inference alignment](../critical_resolution/figures/05_clustered_inference_alignment.png)

![Grouped multinomial validation](../critical_resolution/figures/06_grouped_multinomial_validation.png)

### 추가 시각화

<details>
<summary>7단계의 나머지 시각화 보기</summary>

![Threshold distribution](../critical_resolution/figures/04_threshold_distribution.png)

</details>

### 해석

- 이 단계에서 `Quality + PA`가 prediction-first 기본형으로 부상했다.
- 동시에 `neg_log_loss`로 직접 튜닝하는 것이 calibration 자체를 따로 붙이는 것보다 더 중요하다는 점이 드러났다.
- `Diastolic BP`와 `Quality of Sleep`는 profile-cluster GEE에서도 유지됐다.

---

## 10. 8단계: grouped bootstrap 모델 비교와 변수 안정성 후속 검증

### 왜 이 실험을 했는가

상위 모델 간 차이가 실제로도 큰지, 그리고 최종 변수들이 표본 재추출에도 안정적인지를 마지막으로 점검하기 위해서였다.

### 사용한 방법과 원리

- grouped bootstrap OOB model comparison
- pairwise superiority analysis
- `L1 stability selection`
- final-model coefficient bootstrap

### 핵심 결과 표

#### grouped bootstrap model summary

| 모델 | ROC-AUC | F1 | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| Original: Quality + HR | 0.934 | 0.897 | 0.076 | 0.068 |
| Original: Quality + PA | 0.934 | 0.893 | 0.076 | 0.070 |
| Original: Sleep + PA | 0.933 | 0.889 | 0.081 | 0.076 |

#### stability selection 상위 변수

| 변수 | Selection Frequency |
| --- | ---: |
| BMI Risk | 0.979 |
| Quality of Sleep | 0.843 |
| Systolic BP | 0.757 |
| Diastolic BP | 0.736 |

### 대표 시각화

![Grouped bootstrap 모델 비교](../bootstrap_stability_followup/figures/01_grouped_bootstrap_model_comparison.png)

![Pairwise superiority](../bootstrap_stability_followup/figures/02_pairwise_superiority.png)

![Selection frequency](../bootstrap_stability_followup/figures/03_selection_frequency.png)

![Final model coefficient stability](../bootstrap_stability_followup/figures/04_final_model_coefficient_stability.png)

### 해석

- `Quality + PA`와 `Quality + HR`는 grouped bootstrap 기준으로 사실상 같은 최상위 family였다.
- 변수 수준에서는 `혈압축 + BMI Risk + Quality of Sleep`가 가장 강건했다.
- 최종 `Quality + PA` 고정 모델 안에서는 `Diastolic BP`가 가장 안정적인 단일 신호였다.

---

## 11. 9단계: exact-row dedup sensitivity와 nested grouped bootstrap 최종 보강

### 왜 이 실험을 했는가

두 가지 마지막 비판 포인트가 남아 있었다.

1. 단변량 검정이 full-row 반복에 의해 과장된 것은 아닌가
2. bootstrap 비교가 이전 하이퍼파라미터 선택에 조건부인 것은 아닌가

이를 해소하기 위해 exact-row dedup sensitivity와 nested grouped bootstrap을 수행했다.

### 사용한 방법과 원리

- exact-row dedup 단변량 재검정
- replicate마다 다시 튜닝하는 nested grouped bootstrap

원리는 간단하다. 동일 row 반복을 줄여 단변량 독립성 기준을 더 보수적으로 맞추고, bootstrap 내부에서 다시 튜닝해 모델 비교를 더 엄밀하게 만드는 것이다.

### 핵심 결과 표

#### dataset alignment

| Dataset | Rows | Unique Profiles |
| --- | ---: | ---: |
| Full rows | 374 | 109 |
| Exact-row deduplicated | 132 | 109 |

#### nested grouped bootstrap summary

| 모델 | ROC-AUC | F1 | Brier | ECE | Winner Share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original: Quality + PA | 0.934 | 0.888 | 0.076 | 0.064 | 0.388 |
| Original: Quality + HR | 0.932 | 0.889 | 0.077 | 0.065 | 0.400 |
| Original: Sleep + PA | 0.933 | 0.879 | 0.083 | 0.072 | 0.000 |

### 대표 시각화

![Dedup 수치형 alignment](../final_rigor_upgrade/figures/01_numeric_alignment_heatmap.png)

![Dedup 범주형 alignment](../final_rigor_upgrade/figures/02_categorical_alignment.png)

![Nested grouped bootstrap 비교](../final_rigor_upgrade/figures/03_nested_grouped_bootstrap_comparison.png)

![Nested pairwise comparison](../final_rigor_upgrade/figures/04_nested_pairwise_comparison.png)

### 해석

- exact-row dedup 후에도 핵심 수치형 축은 대부분 유지됐다.
- 따라서 메인 단변량 인사이트는 반복 row에만 의존한 결과가 아니었다.
- nested grouped bootstrap에서도 `Quality + PA`와 `Quality + HR`는 같은 최상위 family로 남았다.
- 다만 `Quality + PA`는 `Brier`가 가장 좋고, `Quality + HR`는 `F1`과 winner share가 약간 더 좋아서, 최종 operational default는 practical choice로 해석하는 것이 가장 엄밀하다.

---

## 12. 최종 종합 해석

9단계 전체 결과를 함께 보면, 아래 결론이 가장 안정적이다.

1. 변수군 수준에서는 `혈압축 + BMI Risk + Quality of Sleep`가 가장 강건하다.
2. 단일 변수 수준에서는 `Diastolic BP`가 가장 안정적인 핵심 위험 신호다.
3. 예측 우선 모델군은 `Quality + PA / Quality + HR`가 같은 최상위 family다.
4. 그중 `Quality + PA`는 확률 품질을 더 중시할 때의 practical default다.
5. `Sleep + PA`는 quality score를 덜 쓰는 보수적 baseline으로 여전히 유용하다.
6. `Insomnia`와 `Sleep Apnea`는 같은 메커니즘으로 뭉뚱그려 해석하면 안 된다.

---

## 13. 관련 문서

- [분석 흐름과 실험 당위성 정리](./analysis_sequence_and_rationale_ko.md)
- [분석 방법 설명서](./analysis_methods_reference_ko.md)
- [인사이트 및 활용 요약](./insights_and_applications_ko.md)
- [수면장애 통합 메인 보고서](../sleep_disorder_statistical_summary_ko.md)
