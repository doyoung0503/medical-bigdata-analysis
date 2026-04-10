# 최종 엄밀성 보강 보고서

## 1. 왜 이 보강 실험이 필요했는가

이전까지의 분석은 이미 상당히 엄격했지만, 최대 엄밀성 기준에서 두 가지가 남아 있었다.

1. 단변량 검정은 full-row 기준인데, 예측 성능 검증은 grouped 기준이었다.
2. grouped bootstrap 모델 비교는 이전 CV에서 선택된 하이퍼파라미터를 고정해 비교했기 때문에, bootstrap 바깥에서 이미 한 번 선택된 설정에 조건부였다.

즉, 이번 실험의 목적은 **단변량 검정과 모델 검증의 독립성 기준을 더 잘 맞추고**, **bootstrap 안에서 다시 튜닝하는 nested 비교로 모델 우열을 재확인**하는 데 있었다.

## 2. 이번에 무엇을 다시 했는가

### 2.1 exact-row deduplicated 단변량 재검정

- 같은 row가 여러 번 반복되면 p-value가 과도하게 작아질 수 있으므로, `person_id`를 제외한 정확히 같은 row는 하나만 남겼다.
- predictor profile이 같아도 label이 다른 경우는 그대로 유지했다.
  - 이유: 이런 경우까지 하나로 접어버리면 임의의 대표 label을 정해야 하므로, 오히려 더 강한 가정을 도입하게 된다.
- 따라서 이번 dedup은 “중복 정보량의 과장”만 줄이고, label ambiguity는 인위적으로 없애지 않는 보수적 민감도 분석이다.

데이터 정렬 결과는 아래와 같다.

| dataset | rows | unique_profiles |
| --- | --- | --- |
| Full rows | 374 | 109 |
| Exact-row deduplicated | 132 | 109 |

여기서 `132`와 `109`가 동시에 남는 것은 오류가 아니다. `109`는 predictor profile 기준 고유 패턴 수이고, `132`는 person_id만 제거한 exact unique row 수라서, 같은 predictor profile 안에 서로 다른 label row가 있으면 dedup 후에도 둘 다 남는다.

### 2.2 nested grouped bootstrap 모델 재비교

- bootstrap replicate마다 train/OOB split을 새로 만들고,
- 각 replicate 내부에서 다시 grouped inner CV로 `C`와 class weight를 재튜닝한 뒤,
- OOB test에서 성능을 평가했다.

즉, 이번 비교는 “이전 CV에서 한 번 고른 설정이 bootstrap에서도 좋다”가 아니라, **replicate마다 다시 최적 설정을 찾은 뒤에도 quality-based family가 앞서는가**를 묻는 구조다.

## 3. 단변량 결론은 dedup 후에도 유지되는가

### 3.1 수치형 변수

| feature | feature_label | welch_fdr_full | kruskal_fdr_full | eta_squared_full | epsilon_squared_full | robust_keep_full | welch_fdr_dedup | kruskal_fdr_dedup | eta_squared_dedup | epsilon_squared_dedup | robust_keep_dedup | consistency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| map_bp | Mean Arterial Pressure | 0.000 | 0.000 | 0.576 | 0.554 | True | 0.000 | 0.000 | 0.348 | 0.357 | True | stable_keep |
| systolic_bp | Systolic BP | 0.000 | 0.000 | 0.536 | 0.549 | True | 0.000 | 0.000 | 0.345 | 0.358 | True | stable_keep |
| diastolic_bp | Diastolic BP | 0.000 | 0.000 | 0.591 | 0.565 | True | 0.000 | 0.000 | 0.344 | 0.360 | True | stable_keep |
| rate_pressure_product | Rate Pressure Product | 0.000 | 0.000 | 0.477 | 0.473 | True | 0.000 | 0.000 | 0.345 | 0.324 | True | stable_keep |
| pulse_pressure | Pulse Pressure | 0.000 | 0.000 | 0.205 | 0.210 | True | 0.000 | 0.000 | 0.221 | 0.223 | True | stable_keep |
| sleep_duration | Sleep Duration | 0.000 | 0.000 | 0.147 | 0.113 | True | 0.000 | 0.003 | 0.111 | 0.085 | True | stable_keep |
| heart_rate | Heart Rate | 0.000 | 0.000 | 0.151 | 0.095 | True | 0.000 | 0.001 | 0.165 | 0.108 | True | stable_keep |
| quality_of_sleep | Quality of Sleep | 0.000 | 0.000 | 0.130 | 0.127 | True | 0.001 | 0.002 | 0.114 | 0.094 | True | stable_keep |
| physical_activity_level | Physical Activity Level | 0.000 | 0.000 | 0.192 | 0.184 | True | 0.002 | 0.003 | 0.102 | 0.085 | True | stable_keep |
| sleep_deficit_7h | Sleep Deficit (vs 7h) | 0.000 | 0.000 | 0.140 | 0.185 | True | 0.002 | 0.001 | 0.102 | 0.117 | True | stable_keep |
| sleep_stress_balance | Sleep-Stress Balance | 0.000 | 0.000 | 0.067 | 0.054 | True | 0.005 | 0.009 | 0.080 | 0.063 | True | stable_keep |
| daily_steps | Daily Steps | 0.000 | 0.000 | 0.118 | 0.099 | True | 0.006 | 0.011 | 0.071 | 0.060 | True | stable_keep |
| stress_level | Stress Level | 0.001 | 0.007 | 0.034 | 0.022 | True | 0.030 | 0.035 | 0.054 | 0.039 | True | stable_keep |
| age | Age | 0.000 | 0.000 | 0.239 | 0.223 | True | 0.048 | 0.038 | 0.059 | 0.036 | True | stable_keep |
| sleep_quality_per_hour | Sleep Quality per Hour | 0.000 | 0.001 | 0.044 | 0.035 | True | 0.026 | 0.097 | 0.067 | 0.021 | False | weakened_after_dedup |
| steps_per_activity | Steps per Activity | 0.000 | 0.000 | 0.094 | 0.120 | True | 0.094 | 0.035 | 0.043 | 0.039 | False | weakened_after_dedup |

핵심 해석:

- full-row와 deduplicated row 모두에서 강건하게 유지된 수치형 축은 다음과 같았다.
  - Mean Arterial Pressure, Systolic BP, Diastolic BP, Rate Pressure Product, Pulse Pressure, Sleep Duration, Heart Rate, Quality of Sleep, Physical Activity Level, Sleep Deficit (vs 7h), Sleep-Stress Balance, Daily Steps, Stress Level, Age
- dedup 후 강건성이 약해진 변수는 다음과 같았다.
  - Sleep Quality per Hour, Steps per Activity

이 결과는 “혈압축, 나이, 수면시간, 수면의 질, 활동량”이라는 큰 방향이 단순 반복행 때문에만 생긴 결과는 아니라는 점을 보여준다.

### 3.2 범주형 변수

| feature | feature_label | perm_fdr_full | cramers_v_full | cells_lt5_full | min_expected_full | perm_fdr_dedup | cramers_v_dedup | cells_lt5_dedup | min_expected_dedup | keep_full | keep_dedup | consistency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bmi_category | BMI Category | 0.001 | 0.573 | 2 | 2.059 | 0.001 | 0.394 | 3 | 1.538 | True | True | stable_keep |
| bmi_collapsed | BMI Category (Collapsed) | 0.001 | 0.808 | 0 | 32.529 | 0.001 | 0.541 | 0 | 12.962 | True | True | stable_keep |
| occupation | Occupation | 0.001 | 0.751 | 12 | 0.206 | 0.001 | 0.485 | 22 | 0.220 | True | True | stable_keep |
| occupation_collapsed | Occupation (Collapsed) | 0.001 | 0.742 | 2 | 2.265 | 0.001 | 0.461 | 13 | 1.538 | True | True | stable_keep |
| gender | Gender | 0.001 | 0.381 | 0 | 38.088 | 0.027 | 0.226 | 0 | 14.280 | True | True | stable_keep |

핵심 해석:

- dedup 후에도 남은 범주형 축은 다음과 같았다.
  - BMI Category, BMI Category (Collapsed), Occupation, Occupation (Collapsed), Gender
- dedup 후 약해진 범주형 축은 다음과 같았다.
  - 없음

즉, 단변량 단계에서도 `Gender`, `BMI 계열` 신호는 비교적 안정적이지만, 희소도가 높은 변수는 독립성 기준을 보수적으로 맞출수록 더 조심해서 읽는 편이 맞다.

## 4. nested grouped bootstrap에서 모델 우열은 어떻게 바뀌는가

### 4.1 성능 요약

| model | model_label | replicates | roc_auc_mean | roc_auc_sd | f1_mean | f1_sd | accuracy_mean | brier_mean | ece_mean | winner_count | winner_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig_quality_pa | Original: Quality + PA | 80 | 0.934 | 0.021 | 0.888 | 0.035 | 0.913 | 0.076 | 0.064 | 31.000 | 0.388 |
| orig_quality_hr | Original: Quality + HR | 80 | 0.932 | 0.022 | 0.889 | 0.037 | 0.914 | 0.077 | 0.065 | 32.000 | 0.400 |
| enhanced | Compressed + Interaction | 80 | 0.932 | 0.020 | 0.884 | 0.034 | 0.910 | 0.080 | 0.066 | 5.000 | 0.062 |
| compressed | Compressed Derived | 80 | 0.931 | 0.020 | 0.884 | 0.043 | 0.911 | 0.080 | 0.070 | 7.000 | 0.087 |
| orig_sleep_hr | Original: Sleep + HR | 80 | 0.934 | 0.022 | 0.883 | 0.044 | 0.909 | 0.080 | 0.069 | 5.000 | 0.062 |
| orig_sleep_pa | Original: Sleep + PA | 80 | 0.933 | 0.021 | 0.879 | 0.045 | 0.906 | 0.083 | 0.072 | 0.000 | 0.000 |

### 4.2 replicate 내부 튜닝에서 가장 자주 선택된 설정

| model | model_label | best_c | best_weight | count |
| --- | --- | --- | --- | --- |
| enhanced | Compressed + Interaction | 0.100 | None | 53 |
| orig_sleep_pa | Original: Sleep + PA | 0.100 | None | 52 |
| orig_quality_pa | Original: Quality + PA | 0.100 | None | 51 |
| orig_sleep_hr | Original: Sleep + HR | 0.100 | None | 48 |
| compressed | Compressed Derived | 0.100 | None | 46 |
| orig_quality_hr | Original: Quality + HR | 0.100 | None | 46 |

### 4.3 paired empirical difference

| comparison | metric | mean_diff | empirical_low | empirical_high | superiority_prob |
| --- | --- | --- | --- | --- | --- |
| Original: Quality + PA - Original: Quality + HR | roc_auc | 0.002 | -0.020 | 0.021 | 0.537 |
| Original: Quality + PA - Original: Quality + HR | f1 | -0.001 | -0.031 | 0.038 | 0.225 |
| Original: Quality + PA - Original: Quality + HR | brier | -0.001 | -0.011 | 0.006 | 0.425 |
| Original: Quality + PA - Original: Quality + HR | ece | -0.001 | -0.033 | 0.021 | 0.537 |
| Original: Quality + PA - Original: Sleep + PA | roc_auc | 0.001 | -0.008 | 0.014 | 0.525 |
| Original: Quality + PA - Original: Sleep + PA | f1 | 0.010 | -0.016 | 0.031 | 0.388 |
| Original: Quality + PA - Original: Sleep + PA | brier | -0.006 | -0.016 | -0.001 | 0.988 |
| Original: Quality + PA - Original: Sleep + PA | ece | -0.008 | -0.042 | 0.019 | 0.688 |
| Original: Quality + PA - Original: Sleep + HR | roc_auc | 0.000 | -0.022 | 0.025 | 0.438 |
| Original: Quality + PA - Original: Sleep + HR | f1 | 0.006 | -0.019 | 0.030 | 0.287 |
| Original: Quality + PA - Original: Sleep + HR | brier | -0.004 | -0.013 | 0.003 | 0.900 |
| Original: Quality + PA - Original: Sleep + HR | ece | -0.005 | -0.034 | 0.021 | 0.650 |
| Original: Quality + PA - Compressed Derived | roc_auc | 0.003 | -0.018 | 0.031 | 0.675 |
| Original: Quality + PA - Compressed Derived | f1 | 0.004 | -0.031 | 0.043 | 0.237 |
| Original: Quality + PA - Compressed Derived | brier | -0.004 | -0.018 | 0.003 | 0.787 |
| Original: Quality + PA - Compressed Derived | ece | -0.006 | -0.038 | 0.025 | 0.688 |
| Original: Quality + PA - Compressed + Interaction | roc_auc | 0.002 | -0.016 | 0.024 | 0.600 |
| Original: Quality + PA - Compressed + Interaction | f1 | 0.005 | -0.019 | 0.041 | 0.300 |
| Original: Quality + PA - Compressed + Interaction | brier | -0.004 | -0.016 | 0.004 | 0.863 |
| Original: Quality + PA - Compressed + Interaction | ece | -0.002 | -0.032 | 0.028 | 0.500 |

핵심 해석:

- nested grouped bootstrap 평균 기준 최상위 모델은 `Original: Quality + PA`였다.
- `Original: Quality + PA`와 `Original: Quality + HR`의 성능 차이는 여전히 매우 작았다.
  - `Quality + PA`: `AUC 0.934`, `F1 0.888`, `Brier 0.076`
  - `Quality + HR`: `AUC 0.932`, `F1 0.889`, `Brier 0.077`
- `Quality + PA - Quality + HR`의 empirical difference를 보면, superiority probability가 0.5 근처에 머물면 단일 승자라고 부르기 어렵다.
- `Quality + PA - Sleep + PA` 비교는 quality score를 쓸 때 얼마나 확률 품질이 좋아지는지를 보여주는 보수적 대조군으로 읽는 편이 맞다.

특히 아래 두 비교는 최종 모델 추천 문장을 결정하는 데 중요했다.

`Quality + PA vs Quality + HR`

| comparison | metric | mean_diff | empirical_low | empirical_high | superiority_prob |
| --- | --- | --- | --- | --- | --- |
| Original: Quality + PA - Original: Quality + HR | roc_auc | 0.002 | -0.020 | 0.021 | 0.537 |
| Original: Quality + PA - Original: Quality + HR | f1 | -0.001 | -0.031 | 0.038 | 0.225 |
| Original: Quality + PA - Original: Quality + HR | brier | -0.001 | -0.011 | 0.006 | 0.425 |
| Original: Quality + PA - Original: Quality + HR | ece | -0.001 | -0.033 | 0.021 | 0.537 |

`Quality + PA vs Sleep + PA`

| comparison | metric | mean_diff | empirical_low | empirical_high | superiority_prob |
| --- | --- | --- | --- | --- | --- |
| Original: Quality + PA - Original: Sleep + PA | roc_auc | 0.001 | -0.008 | 0.014 | 0.525 |
| Original: Quality + PA - Original: Sleep + PA | f1 | 0.010 | -0.016 | 0.031 | 0.388 |
| Original: Quality + PA - Original: Sleep + PA | brier | -0.006 | -0.016 | -0.001 | 0.988 |
| Original: Quality + PA - Original: Sleep + PA | ece | -0.008 | -0.042 | 0.019 | 0.688 |

## 5. 이번 보강 실험으로 무엇이 더 명확해졌는가

1. 단변량 결론은 exact-row dedup 후에도 큰 방향이 유지됐다.
2. 따라서 `혈압축 + BMI Risk + Quality of Sleep`가 중요하다는 메시지는 반복 row 때문에만 생긴 인공 신호라고 보기 어렵다.
3. nested grouped bootstrap 안에서 다시 튜닝해도 quality-based family가 상위권에 남으면, 기존 모델 추천은 이전 fixed-hyperparameter bootstrap에만 의존한 결론이 아니게 된다.
4. 반대로 quality-based family 내부에서 `PA`와 `HR`의 차이가 작다면, 최종 권고는 “유일한 통계적 승자”보다 “같은 최상위 family”라는 식으로 보수적으로 쓰는 편이 더 엄밀하다.

## 6. 최신 기준 최종 문장

이번 보강 실험까지 반영하면 가장 엄밀한 최종 문장은 아래와 같다.

> exact-row dedup 단변량 재검정 후에도 혈압축, BMI 축, 수면의 질/수면시간 축의 큰 방향은 유지됐다.

> nested grouped bootstrap 안에서 다시 튜닝해도 `Quality + PA`와 `Quality + HR`는 여전히 같은 최상위 quality-based prediction-first family로 해석하는 것이 가장 보수적이고 정확하다.

> operational default 하나를 고르더라도, 그것은 통계적으로 압도적 단일 승자라기보다 quality-based family 안에서의 실용적 선택으로 읽는 것이 맞다.

## 7. 생성된 결과물

- 보고서: [final_rigor_upgrade_report_ko.md](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/final_rigor_upgrade/final_rigor_upgrade_report_ko.md)
- 시각화: [final_rigor_upgrade figures](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/final_rigor_upgrade/figures)
- 표: [final_rigor_upgrade tables](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/final_rigor_upgrade/tables)
- 재현 스크립트: [final_rigor_upgrade.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/final_rigor_upgrade.py)

마지막 갱신 시각: `2026-04-10 22:29`
