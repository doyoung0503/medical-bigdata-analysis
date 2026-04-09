# Ridge 로지스틱 회귀 및 파생변수 탐색 보고서

## 1. 분석 목적

이번 추가 분석의 목적은 세 가지다.

1. 기존 분류 문제를 **Ridge(L2) 로지스틱 회귀**로 다시 학습해, 다중공선성이 있는 상황에서 모델이 어떻게 동작하는지 확인한다.
2. 원 변수들 사이의 상관관계와 VIF를 통해 **공선성 구조를 먼저 진단**한다.
3. 이 구조를 바탕으로 **시도해볼 만한 파생변수**를 설계하고, 성능과 해석가능성을 기준으로 실제 가치가 있는지 검증한다.

## 2. 왜 이런 순서로 분석했는가

이번 흐름은 `reference` 자료의 통계 검정 -> 상관분석 -> 회귀모형 해석 흐름을 그대로 확장한 것이다.

1. **상관행렬과 VIF 확인**
   - 이유: Ridge를 쓰더라도 어떤 변수 묶음이 중복정보를 담는지 먼저 알아야 파생변수 설계에 근거가 생긴다.
2. **파생변수 후보 스크리닝**
   - 이유: 임의의 feature engineering이 아니라, 실제로 유의하고 해석 가능한 변수만 다음 단계로 넘겨야 한다.
3. **Ridge 로지스틱 비교**
   - 이유: L2 규제는 공선성이 있어도 계수를 안정화해주므로, 원 변수/압축형 파생변수/증강형 파생변수 세트를 공정하게 비교할 수 있다.
4. **성능과 해석의 균형 판단**
   - 이유: AUC가 조금 높다고 무조건 좋은 모델이 아니라, 공선성·단순성·배치 편의성까지 같이 봐야 실제 활용 모델을 추천할 수 있다.

## 3. 데이터와 전처리

- 사용 데이터: `dataset/Sleep_health_and_lifestyle_dataset/Sleep_health_and_lifestyle_dataset.csv`
- 총 관측치: 374
- 수면장애 양성 비율: 0.414
- `Sleep Disorder` 결측은 `None`으로 간주했다.
- `Blood Pressure`는 `systolic_bp`, `diastolic_bp`로 분해했다.
- `BMI Category`의 `Normal Weight`는 `Normal`로 통합했다.
- `person_id`를 제외하면 동일 프로파일 반복이 많아 실제 임상데이터보다 성능이 낙관적으로 나올 수 있다.

## 4. 원 변수의 상관관계와 다중공선성

### 4.1 높은 상관을 보인 변수쌍

| left | right | correlation |
| --- | --- | --- |
| Systolic BP | Diastolic BP | 0.973 |
| Quality of Sleep | Stress Level | -0.899 |
| Sleep Duration | Quality of Sleep | 0.883 |
| Sleep Duration | Stress Level | -0.811 |
| Physical Activity Level | Daily Steps | 0.773 |

핵심 해석:

- `Systolic BP`와 `Diastolic BP`는 거의 동일한 축으로 움직였다.
- `Quality of Sleep`, `Sleep Duration`, `Stress Level`도 매우 강하게 얽혀 있었다.
- `Physical Activity Level`과 `Daily Steps` 역시 같은 활동군집의 정보가 중복되는 경향을 보였다.

### 4.2 원 변수 VIF

| feature_label | vif |
| --- | --- |
| Diastolic BP | 37.977 |
| Systolic BP | 35.338 |
| Quality of Sleep | 13.261 |
| Stress Level | 9.568 |
| Daily Steps | 6.122 |
| Physical Activity Level | 5.664 |
| Sleep Duration | 5.500 |
| Heart Rate | 3.453 |
| Age | 3.346 |

해석:

- 혈압 변수(VIF 35 이상)와 수면-스트레스 군집(`Quality of Sleep`, `Stress Level`)의 공선성이 가장 강했다.
- 따라서 Ridge를 쓰는 것은 타당하며, 동시에 공통축을 요약하는 파생변수를 만들어볼 통계적 근거가 충분했다.

## 5. 파생변수 후보는 어떻게 정했는가

파생변수는 아래 원칙으로 설계했다.

1. 높은 상관을 보인 원 변수군을 요약할 것
2. 임상적 또는 생활습관 해석이 가능할 것
3. 실제로 `Has Sleep Disorder`와 통계적으로 관련이 있을 것

후보 스크리닝 결과:

| feature_label | domain | pointbiserial_r | pointbiserial_p | eta_squared | most_correlated_original | max_abs_corr_with_original | screening_decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mean Arterial Pressure | 혈압 군집 | 0.705 | <0.001 | 0.576 | Diastolic BP | 0.996 | retain |
| Rate Pressure Product | 심혈관 군집 | 0.632 | <0.001 | 0.477 | Heart Rate | 0.809 | explore_only |
| Pulse Pressure | 혈압 군집 | 0.452 | <0.001 | 0.205 | Systolic BP | 0.776 | retain |
| Sleep Deficit (vs 7h) | 수면-스트레스 군집 | 0.373 | <0.001 | 0.140 | Sleep Duration | 0.883 | retain |
| Sleep-Stress Balance | 수면-스트레스 군집 | -0.240 | <0.001 | 0.067 | Stress Level | 0.983 | retain |
| Sleep Quality per Hour | 수면-스트레스 군집 | -0.189 | <0.001 | 0.044 | Quality of Sleep | 0.782 | explore_only |
| Steps per Activity | 활동량 군집 | -0.184 | <0.001 | 0.094 | Physical Activity Level | 0.741 | explore_only |

해석:

- **유지(retain)** 판단을 받은 핵심 후보는 `Mean Arterial Pressure`, `Pulse Pressure`, `Sleep Deficit (vs 7h)`, `Sleep-Stress Balance`였다.
- `Rate Pressure Product`는 통계적으로 강했지만 심혈관 축과 너무 많이 겹쳐 **탐색용**으로만 두었다.
- `Sleep Quality per Hour`, `Steps per Activity`는 유의하긴 했지만 효과가 상대적으로 약해 **보조 탐색 변수**로 분류했다.

## 6. Ridge 로지스틱 회귀 비교

비교한 모델은 아래 3가지다.

1. **Baseline**: 원 변수 전체
2. **Compressed Derived**: 공선성이 강한 군집을 파생변수로 압축한 세트
3. **Augmented**: 원 변수에 파생변수를 추가해 Ridge가 스스로 가중치를 분배하게 한 세트

성능 비교:

| model | cv_auc | cv_sd | test_auc | accuracy | precision | recall | f1 | best_c |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.946 | 0.020 | 0.976 | 0.929 | 0.898 | 0.936 | 0.917 | 0.000 |
| Compressed Derived | 0.945 | 0.018 | 0.970 | 0.938 | 0.900 | 0.957 | 0.928 | 0.000 |
| Augmented | 0.941 | 0.017 | 0.977 | 0.938 | 0.917 | 0.936 | 0.926 | 0.010 |

핵심 해석:

- **가장 높은 외부형 지표(CV ROC-AUC)** 는 `Baseline`이 기록했다.
- **Compressed Derived**는 CV ROC-AUC가 baseline과 거의 비슷하면서도 정확도/재현성 지표를 크게 해치지 않았다.
- **Augmented**는 hold-out test 수치가 약간 높았지만, 원 변수와 파생변수를 동시에 넣어 해석 목적의 공선성은 더 커졌다.

## 7. 계수 해석

### 7.1 Baseline Ridge 주요 계수

| feature_label | coefficient |
| --- | --- |
| Diastolic BP | 0.009 |
| Systolic BP | 0.009 |
| Age | 0.006 |
| BMI Category: Overweight | 0.005 |
| Sleep Duration | -0.005 |
| Quality of Sleep | -0.005 |
| Heart Rate | 0.004 |
| Stress Level | 0.003 |

### 7.2 Compressed Derived Ridge 주요 계수

| feature_label | coefficient |
| --- | --- |
| Mean Arterial Pressure | 0.009 |
| Age | 0.006 |
| Pulse Pressure | 0.006 |
| Sleep Deficit (vs 7h) | 0.005 |
| BMI Category: Overweight | 0.005 |
| Heart Rate | 0.004 |
| Sleep-Stress Balance | -0.003 |
| Gender: Male | -0.002 |

### 7.3 Augmented Ridge 주요 계수

| feature_label | coefficient |
| --- | --- |
| Diastolic BP | 0.268 |
| Mean Arterial Pressure | 0.259 |
| Systolic BP | 0.242 |
| Age | 0.228 |
| BMI Category: Overweight | 0.211 |
| Quality of Sleep | -0.154 |
| Sleep Duration | -0.140 |
| Sleep Deficit (vs 7h) | 0.103 |

해석:

- `Baseline`에서는 혈압 축, 나이, 수면의 질이 핵심 신호로 남았다.
- `Compressed Derived`에서는 `Mean Arterial Pressure`, `Pulse Pressure`, `Sleep Deficit`, `Sleep-Stress Balance`가 함께 작동하며, 공선성을 줄인 상태에서도 예측력이 유지됐다.
- `Augmented`에서는 혈압 관련 원 변수와 파생변수에 계수가 분산되었다. 이는 Ridge가 공선성 축 전체에 가중치를 나눠 배분하는 전형적 패턴이다.

## 8. 압축형 파생변수 세트의 공선성

| feature_label | vif |
| --- | --- |
| Sleep-Stress Balance | 7.341 |
| Sleep Deficit (vs 7h) | 4.627 |
| Age | 3.088 |
| Mean Arterial Pressure | 2.895 |
| Heart Rate | 2.226 |
| Pulse Pressure | 2.000 |

해석:

- 압축형 세트는 원 변수 세트보다 VIF가 눈에 띄게 낮아졌다.
- 특히 혈압군을 `MAP + Pulse Pressure`로 바꾼 것이 다중공선성 완화에 가장 크게 기여했다.
- `Sleep-Stress Balance`는 여전히 수면 관련 정보와 연결되어 있으나, 원 세트의 `Quality of Sleep` / `Stress Level` 동시투입보다 해석이 훨씬 단순해졌다.

## 9. 최종 결론

### 9.1 Ridge 모델을 적용했을 때의 결론

- 이 데이터에서는 공선성이 분명히 존재하므로 **Ridge 로지스틱 회귀 적용은 타당**했다.
- Ridge를 쓰면 변수 제거 없이도 안정적인 분류가 가능했고, 세 모델 모두 ROC-AUC 0.94 이상을 유지했다.

### 9.2 시도해볼 만한 파생변수

실제로 다음 파생변수는 통계적으로 시도해볼 가치가 있었다.

1. `Mean Arterial Pressure`
2. `Pulse Pressure`
3. `Sleep Deficit (vs 7h)`
4. `Sleep-Stress Balance`

이 변수들이 유효한 이유:

- 높은 상관군을 요약한다.
- 타깃과의 상관 또는 집단 간 차이가 충분하다.
- 원 변수 대비 공선성을 줄이면서도 예측력을 크게 손상시키지 않는다.

### 9.3 어떤 모델을 추천할 것인가

- **예측 성능 최우선**이면 `Baseline Ridge`를 기본선으로 두는 것이 안전하다.
- **실무 배치와 해석 용이성**까지 고려하면 `Compressed Derived` 모델이 더 추천된다.
  - 이유: 성능 손실이 매우 작고, 변수 수와 공선성이 줄어 설명하기 쉽다.
- `Augmented`는 연구용 실험 세트로는 유용하지만, 원 변수와 파생변수를 동시에 넣는 구조라 해석형 보고모형으로는 덜 적합하다.

## 10. 실무 인사이트

이번 결과는 다음과 같은 도움을 줄 수 있다.

1. 혈압 축은 `수축기/이완기 각각`보다 `MAP`와 `Pulse Pressure`로 재구성해도 충분히 강한 예측 신호를 얻을 수 있다.
2. 수면 관련 변수는 `수면시간`, `수면질`, `스트레스`를 따로 모두 넣기보다 `Sleep Deficit`와 `Sleep-Stress Balance`로 요약하면 더 단순한 스크리닝 모델을 만들 수 있다.
3. 따라서 실제 서비스나 의료 스크리닝 환경에서는 **나이 + 심혈관 압력요약 + 수면부족/스트레스 균형지표** 조합이 간단하고 설명 가능한 고위험군 선별 틀로 활용될 수 있다.

## 11. 생성된 산출물

### 11.1 보고서

- `results/ridge_feature_study/ridge_feature_study_report_ko.md`

### 11.2 시각화

1. `results/ridge_feature_study/figures/01_original_correlation_heatmap.png`
2. `results/ridge_feature_study/figures/02_high_correlation_pairs.png`
3. `results/ridge_feature_study/figures/03_vif_comparison.png`
4. `results/ridge_feature_study/figures/04_derived_feature_screening.png`
5. `results/ridge_feature_study/figures/05_original_derived_correlation_map.png`
6. `results/ridge_feature_study/figures/06_model_performance_comparison.png`
7. `results/ridge_feature_study/figures/07_baseline_ridge_coefficients.png`
8. `results/ridge_feature_study/figures/08_compressed_ridge_coefficients.png`
9. `results/ridge_feature_study/figures/09_augmented_ridge_coefficients.png`
10. `results/ridge_feature_study/figures/10_roc_curve_comparison.png`
11. `results/ridge_feature_study/figures/11_confusion_matrix_comparison.png`
