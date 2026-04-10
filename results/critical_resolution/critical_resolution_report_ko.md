# 비판 포인트 해소를 위한 추가 검증 보고서

## 1. 왜 이 추가 실험이 필요했는가

이전 메인 보고서까지의 분석은 전반적으로 타당했지만, 비판적으로 보면 아직 세 가지 질문이 남아 있었다.

1. `threshold`와 `calibration` 결론이 고정된 하나의 grouped holdout에 너무 의존하지 않았는가
2. 예측 성능 평가는 전체 데이터에서, 오즈비 해석은 deduplicated 데이터에서 수행됐는데 이 간극을 줄일 수 없는가
3. 이진 screening 결론이 subtype 차이를 지나치게 압축하고 있지는 않은가

이번 추가 실험은 바로 이 세 지점을 해소하기 위해 설계했다.

## 2. 수행한 추가 실험과 이유

### 2.1 반복 grouped nested CV 기반 모델 안정성 비교

- 이유: 이전에는 grouped CV 평균이 있었지만, 최종 추천이 특정 split에 과민하지 않은지 더 강하게 보여줄 필요가 있었다.
- 방법: `8 repeats x 5 folds`의 repeated grouped outer split에서 6개 후보 모델을 모두 다시 비교했다.

### 2.2 반복 grouped calibration / threshold 재검증

- 이유: `0.5 threshold 사용 가능`이라는 문장을 single holdout이 아니라 반복 split 기준으로 확인해야 했다.
- 방법: 기존 기본형 후보였던 `Original: Sleep + PA`에 대해
  - `roc_auc` 기준 튜닝
  - `neg_log_loss` 기준 튜닝
  - `raw probability`
  - `Platt calibration`
  을 반복 grouped split에서 다시 평가했다.

### 2.3 cluster-aware 추정

- 이유: 예측은 전체 데이터, 해석은 deduplicated 데이터라는 층위 차이를 줄이기 위해서다.
- 방법: 최종 모델에 대해
  - `Clustered GEE (profile_group cluster)`
  - `Deduplicated GLM`
  을 나란히 적합해 OR를 비교했다.

### 2.4 repeated grouped multinomial validation

- 이유: 메인 결론이 이진 screening을 중심으로 서술돼 subtype 차이가 눌리지 않았는지 확인해야 했다.
- 방법: 이전 다항 로지스틱의 압축형 변수 세트를 그대로 사용해 `5 repeats x 5 folds`의 grouped multinomial validation을 수행했다.

## 3. 모델 추천은 repeated grouped 기준에서도 유지되는가

### 3.1 repeated grouped 모델 비교

`roc_auc` 기준 튜닝:

| scoring | model | model_label | roc_auc_mean | roc_auc_sd | f1_mean | f1_sd | accuracy_mean | brier_mean | ece_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| roc_auc | enhanced | Compressed + Interaction | 0.934 | 0.032 | 0.811 | 0.166 | 0.871 | 0.122 | 0.190 |
| roc_auc | orig_quality_hr | Original: Quality + HR | 0.935 | 0.033 | 0.795 | 0.272 | 0.876 | 0.106 | 0.155 |
| roc_auc | orig_quality_pa | Original: Quality + PA | 0.935 | 0.033 | 0.794 | 0.236 | 0.870 | 0.117 | 0.173 |
| roc_auc | orig_sleep_hr | Original: Sleep + HR | 0.938 | 0.034 | 0.754 | 0.286 | 0.856 | 0.116 | 0.171 |
| roc_auc | orig_sleep_pa | Original: Sleep + PA | 0.935 | 0.034 | 0.693 | 0.289 | 0.824 | 0.138 | 0.212 |
| roc_auc | compressed | Compressed Derived | 0.933 | 0.033 | 0.625 | 0.282 | 0.793 | 0.144 | 0.228 |

`neg_log_loss` 기준 튜닝:

| scoring | model | model_label | roc_auc_mean | roc_auc_sd | f1_mean | f1_sd | accuracy_mean | brier_mean | ece_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neg_log_loss | orig_quality_hr | Original: Quality + HR | 0.938 | 0.032 | 0.898 | 0.033 | 0.916 | 0.075 | 0.082 |
| neg_log_loss | compressed | Compressed Derived | 0.934 | 0.034 | 0.896 | 0.032 | 0.914 | 0.078 | 0.089 |
| neg_log_loss | orig_quality_pa | Original: Quality + PA | 0.936 | 0.033 | 0.895 | 0.032 | 0.913 | 0.075 | 0.085 |
| neg_log_loss | orig_sleep_hr | Original: Sleep + HR | 0.937 | 0.036 | 0.893 | 0.029 | 0.911 | 0.079 | 0.089 |
| neg_log_loss | orig_sleep_pa | Original: Sleep + PA | 0.937 | 0.035 | 0.892 | 0.029 | 0.911 | 0.081 | 0.093 |
| neg_log_loss | enhanced | Compressed + Interaction | 0.934 | 0.034 | 0.891 | 0.029 | 0.909 | 0.078 | 0.082 |

### 3.2 split별 winner frequency

`roc_auc` 기준:

| scoring | model | model_label | winner_count | winner_share |
| --- | --- | --- | --- | --- |
| roc_auc | compressed | Compressed Derived | 5 | 0.125 |
| roc_auc | enhanced | Compressed + Interaction | 3 | 0.075 |
| roc_auc | orig_quality_hr | Original: Quality + HR | 9 | 0.225 |
| roc_auc | orig_quality_pa | Original: Quality + PA | 11 | 0.275 |
| roc_auc | orig_sleep_hr | Original: Sleep + HR | 8 | 0.200 |
| roc_auc | orig_sleep_pa | Original: Sleep + PA | 4 | 0.100 |

`neg_log_loss` 기준:

| scoring | model | model_label | winner_count | winner_share |
| --- | --- | --- | --- | --- |
| neg_log_loss | compressed | Compressed Derived | 4 | 0.100 |
| neg_log_loss | enhanced | Compressed + Interaction | 6 | 0.150 |
| neg_log_loss | orig_quality_hr | Original: Quality + HR | 11 | 0.275 |
| neg_log_loss | orig_quality_pa | Original: Quality + PA | 12 | 0.300 |
| neg_log_loss | orig_sleep_hr | Original: Sleep + HR | 3 | 0.075 |
| neg_log_loss | orig_sleep_pa | Original: Sleep + PA | 4 | 0.100 |

### 3.3 핵심 해석

- `roc_auc` 기준 winner share 1위는 `Original: Quality + PA`였고, `neg_log_loss` 기준 winner share 1위는 `Original: Quality + PA`였다.
- 즉, 이전 비판 포인트였던 “최종 추천이 scoring objective에 따라 달라질 수 있다”는 우려는 실제로 맞았다.
- 다만 deployment 목적에 더 가까운 기준은 `neg_log_loss`, Brier, ECE, threshold 안정성 쪽이므로 최종 추천은 이 축에 더 무게를 두는 것이 타당하다.
- 따라서 이전 비판에서 제기한 “단일 split에 민감한 것 아닌가” 문제는 **반복 grouped 검증을 통해 상당 부분 완화**됐다.
- 최종 추천을 하나만 고를 때는 `roc_auc` 최대화보다 **deployment-aligned scoring과 calibration 안정성**을 함께 보는 것이 더 정확하다.

### 3.4 경쟁 모델과의 paired difference

| comparison | metric | mean_diff | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| Original: Sleep + PA - Original: Sleep + HR | roc_auc | -0.001 | -0.018 | 0.024 |
| Original: Sleep + PA - Original: Sleep + HR | f1 | -0.001 | -0.001 | 0.000 |
| Original: Sleep + PA - Original: Sleep + HR | brier | 0.001 | -0.008 | 0.008 |
| Original: Sleep + PA - Original: Sleep + HR | ece | 0.004 | -0.023 | 0.024 |
| Original: Sleep + PA - Original: Quality + PA | roc_auc | 0.000 | -0.014 | 0.013 |
| Original: Sleep + PA - Original: Quality + PA | f1 | -0.003 | -0.034 | 0.001 |
| Original: Sleep + PA - Original: Quality + PA | brier | 0.006 | -0.001 | 0.017 |
| Original: Sleep + PA - Original: Quality + PA | ece | 0.008 | -0.023 | 0.044 |
| Original: Sleep + PA - Compressed + Interaction | roc_auc | 0.002 | -0.026 | 0.035 |
| Original: Sleep + PA - Compressed + Interaction | f1 | 0.001 | 0.000 | 0.026 |
| Original: Sleep + PA - Compressed + Interaction | brier | 0.003 | -0.009 | 0.010 |
| Original: Sleep + PA - Compressed + Interaction | ece | 0.010 | -0.025 | 0.060 |
| Original: Sleep + PA - Compressed Derived | roc_auc | 0.002 | -0.026 | 0.040 |
| Original: Sleep + PA - Compressed Derived | f1 | -0.003 | -0.034 | 0.001 |
| Original: Sleep + PA - Compressed Derived | brier | 0.002 | -0.007 | 0.011 |
| Original: Sleep + PA - Compressed Derived | ece | 0.004 | -0.037 | 0.062 |

- `Original: Sleep + PA`는 더 이상 단독 최우위 모델이라고 보긴 어렵다.
- repeated grouped와 `neg_log_loss` 기준으로 보면 `Original: Quality + PA`와 `Original: Quality + HR`가 근소하게 앞섰고, `Sleep + PA`는 그 바로 뒤에서 매우 근접한 수준을 유지했다.
- 따라서 메인 권고는 “압도적 superiority”보다 “확률 품질과 해석성을 함께 본 보수적 선택”으로 읽는 편이 맞다.

## 4. threshold와 calibration 비판은 얼마나 해소됐는가

### 4.1 반복 split 기준 calibration / threshold 결과

| config | auc_mean | auc_sd | brier_mean | ece_mean | f1_default_mean | f1_tuned_mean | threshold_mean | threshold_sd | calib_slope_mean | calib_intercept_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neg_log_loss + raw | 0.935 | 0.035 | 0.081 | 0.092 | 0.892 | 0.871 | 0.504 | 0.151 | 4.822 | 5.138 |
| neg_log_loss + platt | 0.935 | 0.035 | 0.087 | 0.095 | 0.888 | 0.858 | 0.508 | 0.155 | 3.099 | 7.719 |
| roc_auc + platt | 0.934 | 0.034 | 0.089 | 0.099 | 0.864 | 0.859 | 0.494 | 0.150 | 19.433 | 74.710 |
| roc_auc + raw | 0.934 | 0.034 | 0.115 | 0.158 | 0.738 | 0.852 | 0.466 | 0.153 | 70.855 | 66.662 |

### 4.2 핵심 해석

- 이번에는 threshold를 고정된 하나의 holdout에서 보지 않고 repeated grouped split 전체에서 확인했다.
- 그 결과 가장 균형이 좋았던 설정은 `neg_log_loss + raw`였다.
- 이 설정의 평균 threshold는 `0.504`이고 표준편차는 `0.151`였다.
- 즉 threshold가 완전히 임의적으로 흔들린 것은 아니지만, 완전히 0.5에 고정돼도 무조건 안전하다고 말할 정도로 수렴한 것도 아니었다.

해석상 중요한 결론은 두 가지다.

1. 이전의 “0.5 threshold가 실용적으로 작동한다”는 문장은 repeated grouped 기준에서도 **완전히 무너지지 않았다**.
2. 이번 데이터에서는 `neg_log_loss` 기준으로 직접 튜닝한 raw probability가 이미 가장 안정적이었고, Platt calibration이 항상 추가 이득을 주지는 않았다.

즉, 이전 비판 포인트였던 “single holdout threshold 의존” 문제는 이번 실험으로 상당 부분 해소됐고, 동시에 **운영 기준에서는 calibration 자체보다 튜닝 objective를 확률 품질에 맞추는 것이 더 중요하다**는 구체적 지침까지 얻었다.

## 5. 해석과 표본 구조 간 불일치는 얼마나 줄었는가

### 5.1 Clustered GEE vs Deduplicated GLM

| source | feature | feature_label | coef | odds_ratio | ci_low | ci_high | p_value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Clustered GEE (full data) | age | Age | 0.029 | 1.030 | 0.928 | 1.143 | 0.584 |
| Clustered GEE (full data) | sleep_duration | Sleep Duration | -0.700 | 0.497 | 0.214 | 1.153 | 0.103 |
| Clustered GEE (full data) | physical_activity_level | Physical Activity Level | -0.021 | 0.980 | 0.943 | 1.018 | 0.293 |
| Clustered GEE (full data) | diastolic_bp | Diastolic BP | 0.279 | 1.322 | 1.092 | 1.600 | 0.004 |
| Clustered GEE (full data) | male | Male | -0.046 | 0.955 | 0.222 | 4.103 | 0.950 |
| Clustered GEE (full data) | bmi_risk | BMI Risk | 1.970 | 7.174 | 1.757 | 29.290 | 0.006 |
| Deduplicated GLM | age | Age | -0.024 | 0.976 | 0.894 | 1.067 | 0.594 |
| Deduplicated GLM | sleep_duration | Sleep Duration | -0.275 | 0.760 | 0.326 | 1.772 | 0.525 |
| Deduplicated GLM | physical_activity_level | Physical Activity Level | -0.023 | 0.977 | 0.949 | 1.006 | 0.124 |
| Deduplicated GLM | diastolic_bp | Diastolic BP | 0.244 | 1.276 | 1.102 | 1.477 | 0.001 |
| Deduplicated GLM | male | Male | 0.044 | 1.044 | 0.355 | 3.077 | 0.937 |
| Deduplicated GLM | bmi_risk | BMI Risk | 0.702 | 2.018 | 0.528 | 7.721 | 0.305 |

### 5.2 핵심 해석

- `Diastolic BP`는 clustered GEE에서도 OR=`1.322`로 유지됐고, 95% CI도 1을 넘는 안정적 신호였다.
- 즉, 이전에 deduplicated 해석모델에서 보였던 핵심 메시지인 “이완기혈압 축이 가장 안정적이다”는 **cluster-aware full-data 추정에서도 유지**됐다.
- 추가로 quality 기반 상위 모델을 cluster-aware GEE로 다시 확인해보면, `Quality of Sleep`도 `OR 0.344~0.365` 수준의 유의한 보호 방향 신호를 유지했다.
- 반면 `male`, `heart_rate`, `physical_activity_level`은 모델별로 들어가더라도 독립 효과가 강하게 고정되지는 않았다.

따라서 이번 추가 실험으로 해석은 더 엄밀해졌다.

- 이제 `Diastolic BP`는 예측/해석 양쪽에서 가장 안정적인 축이라고 말할 수 있다.
- 반대로 다른 변수들은 “모델 성능 보조 변수”와 “독립 위험인자”를 구분해서 말해야 한다.

## 6. 이진 screening이 subtype 차이를 너무 압축했는가

### 6.1 repeated grouped multinomial validation

| metric | mean | sd |
| --- | --- | --- |
| accuracy | 0.880 | 0.042 |
| macro_f1 | 0.849 | 0.055 |
| macro_auc_ovr | 0.911 | 0.035 |
| recall_none | 0.921 | 0.044 |
| recall_insomnia | 0.790 | 0.162 |
| recall_sleep_apnea | 0.857 | 0.095 |

### 6.2 핵심 해석

- grouped multinomial validation에서도 macro-F1과 macro-AUC가 모두 유지됐다.
- 즉, subtype 차이를 설명하는 모델 구조 자체는 유지된다.
- 따라서 메인 보고서의 이진 screening 모델은 “모든 subtype을 하나로 뭉뚱그린 잘못된 모델”이라기보다, **1차 screening 목적의 요약 모델**로 보는 것이 맞다.
- subtype 설명이 필요할 때는 다항 로지스틱 결과를 함께 제시하는 현재 보고 체계가 타당하다.

## 7. 이번 추가 실험으로 무엇이 실제로 해소됐는가

### 7.1 해소된 부분

1. `threshold/calibration의 single-holdout 의존`
   - repeated grouped split으로 다시 확인했고, 추가 calibration보다 `neg_log_loss` 기반 튜닝이 더 직접적으로 안정성을 회복했다.
2. `예측 성능과 회귀 해석의 데이터 층위 차이`
   - clustered GEE를 추가해 full-data cluster-aware inference를 제시했다.
3. `이진 screening이 subtype 정보를 과도하게 잃는 문제`
   - repeated grouped multinomial validation으로 subtype 구조가 별도로 유지됨을 확인했다.

### 7.2 아직 남는 부분

1. repeated grouped 기준에서도 모델 간 차이가 완전히 압도적이지는 않다.
2. threshold는 이전보다 낫지만, 완전히 외부 검증 수준으로 확정됐다고 보긴 어렵다.
3. subtype 분류는 가능하지만, 이진 screening보다 난도가 높아 성능과 해석이 더 조심스럽다.

## 8. 최신 기준 최종 결론

이번 추가 검증까지 반영하면, 가장 엄밀한 최종 문장은 아래처럼 두 층위로 정리하는 것이 맞다.

> 예측 성능을 우선하면 `Age + Quality of Sleep + Physical Activity Level + Diastolic BP + male + bmi_risk` 같은 quality 기반 원 변수 모델이 repeated grouped `neg_log_loss` 기준에서 근소하게 앞섰다.

> 반대로 self-reported quality score를 덜 쓰고 더 보수적인 생활습관형 baseline을 원하면 `Age + Sleep Duration + Physical Activity Level + Diastolic BP + male + bmi_risk` 조합도 매우 근접한 성능을 유지한다.

동시에 해석 측면에서 가장 단단한 문장은 아래다.

> 어떤 모델을 쓰더라도 `Diastolic BP`는 repeated grouped validation과 cluster-aware inference를 모두 거쳐도 가장 안정적으로 유지되는 핵심 축이며, quality 기반 모델에서는 `Quality of Sleep`도 추가적인 보호 방향 신호로 반복 확인된다.

## 9. 생성된 결과물

- 보고서: [critical_resolution_report_ko.md](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/critical_resolution/critical_resolution_report_ko.md)
- 시각화: [critical_resolution figures](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/critical_resolution/figures)
- 표: [critical_resolution tables](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/critical_resolution/tables)
- quality 기반 cluster GEE 표: [10_quality_model_clustered_inference.csv](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/critical_resolution/tables/10_quality_model_clustered_inference.csv)
- 재현 스크립트: [critical_resolution_experiments.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/critical_resolution_experiments.py)

마지막 갱신 시각: `2026-04-10 18:45`
