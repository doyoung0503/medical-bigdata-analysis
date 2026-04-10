# Grouped Bootstrap 모델 비교 및 변수 안정성 후속 보고서

## 1. 왜 이 후속 실험이 필요했는가

직전 단계까지의 분석으로 `Quality + PA`가 prediction-first 기본형이라는 결론에는 도달했지만, 두 가지 질문은 여전히 남아 있었다.

1. 상위 모델 간 차이가 실제로도 의미 있게 크다고 볼 수 있는가
2. 최종 변수와 후보 변수들은 분할이 바뀌어도 안정적으로 남는가

평균 ROC-AUC나 F1만 보고 모델을 고르면, 데이터 분할 우연에 민감한 결론을 과하게 일반화할 수 있다. 또한 최종 모델에 들어간 변수가 해석상 중요해 보여도, 다른 표본 재추출에서는 쉽게 빠지는 변수라면 “핵심 변수”라는 표현을 조심해야 한다. 그래서 이번 후속 실험은 **모델 추천의 엄밀성**과 **변수 안정성의 엄밀성**을 각각 보강하기 위해 설계했다.

## 2. 어떤 방식으로 검증했는가

### 2.1 grouped bootstrap OOB 모델 비교

- 이유: repeated grouped CV 결과는 이미 있었지만, 모델 차이를 다른 resampling 프레임에서도 다시 확인할 필요가 있었다.
- 방법:
  - predictor profile 기준 group을 bootstrap 단위로 사용했다.
  - bootstrap train은 group을 복원추출로 다시 뽑아 만들고, OOB group을 test로 사용했다.
  - 각 모델은 직전 `neg_log_loss` grouped CV에서 가장 자주 선택된 하이퍼파라미터를 고정한 뒤 비교했다.
  - 총 유효 replicate 수는 `220`회였다.

고정 하이퍼파라미터는 아래와 같다.

| model | model_label | fixed_c | fixed_weight | supporting_folds | mean_brier | mean_auc | mean_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| orig_quality_hr | Original: Quality + HR | 0.100 | None | 19 | 0.068 | 0.951 | 0.907 |
| orig_sleep_pa | Original: Sleep + PA | 0.100 | None | 28 | 0.077 | 0.946 | 0.896 |
| orig_sleep_hr | Original: Sleep + HR | 0.100 | None | 30 | 0.077 | 0.943 | 0.895 |
| orig_quality_pa | Original: Quality + PA | 0.100 | None | 22 | 0.072 | 0.948 | 0.899 |
| compressed | Compressed Derived | 0.100 | None | 31 | 0.078 | 0.934 | 0.896 |
| enhanced | Compressed + Interaction | 0.100 | None | 36 | 0.078 | 0.936 | 0.892 |

### 2.2 stability selection

- 이유: 최종 변수들이 정말 안정적인지 보려면, 사람이 고른 모델만 보는 것보다 더 넓은 원 변수 후보군에서 selection frequency를 확인하는 편이 낫다.
- 후보 변수:
  - `Age`, `Sleep Duration`, `Quality of Sleep`, `Physical Activity Level`, `Stress Level`, `Heart Rate`, `Daily Steps`, `Systolic BP`, `Diastolic BP`, `Male`, `BMI Risk`
- 방법:
  - grouped bootstrap train을 다시 만들고
  - 그 안에서 `L1 logistic + grouped inner CV`로 `C`와 class weight를 튜닝해
  - 각 replicate에서 어떤 변수가 0이 아닌 계수로 남는지 기록했다.
- 가장 자주 선택된 stability-selection 설정은 `C=0.1`, `class_weight=None`였다.

### 2.3 최종 모델 coefficient bootstrap

- 이유: 최종 모델 변수는 selection 여부뿐 아니라, 방향성과 계수 부호가 안정적인지도 봐야 한다.
- 방법:
  - `Original: Quality + PA` 모델을 grouped bootstrap으로 `220`회 다시 적합하고
  - 표준화 계수의 중앙값, 95% percentile interval, 부호 안정성(sign stability)을 계산했다.

## 3. 모델 차이는 실제로 얼마나 큰가

### 3.1 grouped bootstrap 성능 요약

| model | model_label | replicates | roc_auc_mean | roc_auc_sd | f1_mean | f1_sd | accuracy_mean | brier_mean | ece_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig_quality_hr | Original: Quality + HR | 220 | 0.934 | 0.019 | 0.897 | 0.037 | 0.917 | 0.076 | 0.068 |
| orig_quality_pa | Original: Quality + PA | 220 | 0.934 | 0.021 | 0.893 | 0.037 | 0.914 | 0.076 | 0.070 |
| compressed | Compressed Derived | 220 | 0.933 | 0.020 | 0.895 | 0.038 | 0.916 | 0.078 | 0.075 |
| enhanced | Compressed + Interaction | 220 | 0.931 | 0.021 | 0.892 | 0.038 | 0.913 | 0.079 | 0.070 |
| orig_sleep_hr | Original: Sleep + HR | 220 | 0.934 | 0.020 | 0.891 | 0.039 | 0.913 | 0.080 | 0.072 |
| orig_sleep_pa | Original: Sleep + PA | 220 | 0.933 | 0.023 | 0.889 | 0.038 | 0.911 | 0.081 | 0.076 |

### 3.2 핵심 해석

- bootstrap 평균 기준으로도 `Original: Quality + PA`는 상위권 모델군에 남았다.
- 다만 이번 grouped bootstrap OOB 평균만 놓고 보면 가장 좋은 평균 성능은 `Original: Quality + HR` 쪽에 더 가까웠다.
- `Original: Quality + PA`와 `Original: Quality + HR`의 차이는 매우 작았고, `Compressed Derived`, `Original: Sleep + HR`, `Original: Sleep + PA`도 바로 뒤를 따랐다.
- 즉, 이번 실험은 `Quality + PA`를 단독 승자로 강화했다기보다, **quality 기반 원 변수 모델군이 가장 안정적인 최상위권**이라는 해석을 더 강하게 만들었다.

### 3.3 paired superiority

| comparison | metric | mean_diff | ci_low | ci_high | superiority_prob |
| --- | --- | --- | --- | --- | --- |
| Original: Quality + PA - Original: Quality + HR | roc_auc | 0.000 | -0.018 | 0.018 | 0.491 |
| Original: Quality + PA - Original: Quality + HR | f1 | -0.004 | -0.026 | 0.011 | 0.059 |
| Original: Quality + PA - Original: Quality + HR | brier | 0.000 | -0.006 | 0.004 | 0.409 |
| Original: Quality + PA - Original: Quality + HR | ece | 0.001 | -0.025 | 0.031 | 0.432 |
| Original: Quality + PA - Original: Sleep + PA | roc_auc | 0.001 | -0.011 | 0.020 | 0.500 |
| Original: Quality + PA - Original: Sleep + PA | f1 | 0.004 | -0.004 | 0.027 | 0.282 |
| Original: Quality + PA - Original: Sleep + PA | brier | -0.005 | -0.010 | 0.000 | 0.968 |
| Original: Quality + PA - Original: Sleep + PA | ece | -0.006 | -0.038 | 0.020 | 0.695 |
| Original: Quality + PA - Original: Sleep + HR | roc_auc | -0.000 | -0.020 | 0.024 | 0.450 |
| Original: Quality + PA - Original: Sleep + HR | f1 | 0.002 | -0.019 | 0.026 | 0.186 |
| Original: Quality + PA - Original: Sleep + HR | brier | -0.003 | -0.011 | 0.002 | 0.905 |
| Original: Quality + PA - Original: Sleep + HR | ece | -0.003 | -0.034 | 0.025 | 0.559 |
| Original: Quality + PA - Compressed Derived | roc_auc | 0.001 | -0.014 | 0.019 | 0.545 |
| Original: Quality + PA - Compressed Derived | f1 | -0.002 | -0.023 | 0.018 | 0.077 |
| Original: Quality + PA - Compressed Derived | brier | -0.002 | -0.010 | 0.003 | 0.782 |
| Original: Quality + PA - Compressed Derived | ece | -0.005 | -0.041 | 0.028 | 0.673 |
| Original: Quality + PA - Compressed + Interaction | roc_auc | 0.003 | -0.014 | 0.031 | 0.609 |
| Original: Quality + PA - Compressed + Interaction | f1 | 0.002 | -0.015 | 0.025 | 0.159 |
| Original: Quality + PA - Compressed + Interaction | brier | -0.003 | -0.012 | 0.005 | 0.777 |
| Original: Quality + PA - Compressed + Interaction | ece | 0.000 | -0.033 | 0.035 | 0.495 |

이 표는 `Quality + PA`가 비교 모델보다 얼마나 자주 더 좋은지를 보여준다.

- superiority probability가 `0.5` 근처면 우열이 사실상 애매하다는 뜻이다.
- `Quality + PA`는 `Sleep + PA` 대비 Brier에서 개선되는 방향으로 더 자주 나타났지만, 그 확률이 1에 가깝진 않았다.
- `Quality + HR`와의 비교에서는 실질적으로 거의 동급이라고 보는 편이 더 맞다. 실제로 bootstrap 평균 성능은 `Quality + HR`가 아주 근소하게 앞섰다 (`Brier 0.076` vs `Quality + PA 0.076`).

따라서 모델 비교 차원에서 더 엄밀한 최종 문장은 아래처럼 정리된다.

> grouped bootstrap은 `Quality + PA` 단독 우위를 강하게 지지하지 않았다. 대신 `Quality + PA`와 `Quality + HR`가 같은 최상위 quality-based model family로 남는다고 해석하는 것이 가장 정확하다.

## 4. 어떤 변수가 안정적으로 선택되는가

### 4.1 stability selection 결과

| feature | feature_label | selection_freq | positive_freq | negative_freq | median_coef | median_abs_coef | nonzero_median_coef |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bmi_risk | BMI Risk | 0.979 | 0.979 | 0.000 | 1.015 | 1.015 | 1.029 |
| quality_of_sleep | Quality of Sleep | 0.843 | 0.000 | 0.843 | -0.312 | 0.312 | -0.395 |
| systolic_bp | Systolic BP | 0.757 | 0.757 | 0.000 | 0.443 | 0.443 | 0.616 |
| diastolic_bp | Diastolic BP | 0.736 | 0.729 | 0.007 | 0.397 | 0.397 | 0.676 |
| male | Male | 0.550 | 0.100 | 0.450 | 0.000 | 0.030 | -0.158 |
| heart_rate | Heart Rate | 0.443 | 0.393 | 0.050 | 0.000 | 0.000 | 0.167 |
| age | Age | 0.400 | 0.400 | 0.000 | 0.000 | 0.000 | 0.660 |
| daily_steps | Daily Steps | 0.279 | 0.057 | 0.221 | 0.000 | 0.000 | -0.087 |
| sleep_duration | Sleep Duration | 0.243 | 0.121 | 0.121 | 0.000 | 0.000 | 0.008 |
| physical_activity_level | Physical Activity Level | 0.207 | 0.079 | 0.129 | 0.000 | 0.000 | -0.164 |
| stress_level | Stress Level | 0.129 | 0.050 | 0.079 | 0.000 | 0.000 | -0.183 |

### 4.2 핵심 해석

- 높은 selection frequency는 “표본이 바뀌어도 자주 살아남는 변수”라는 뜻이다.
- 낮은 selection frequency는 예측에 조금 기여하더라도 다른 변수와 정보가 겹치거나, 독립 신호가 약하다는 뜻일 수 있다.

상위 안정 변수 5개는 아래와 같았다.

| feature | feature_label | selection_freq | positive_freq | negative_freq | median_coef | median_abs_coef | nonzero_median_coef |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bmi_risk | BMI Risk | 0.979 | 0.979 | 0.000 | 1.015 | 1.015 | 1.029 |
| quality_of_sleep | Quality of Sleep | 0.843 | 0.000 | 0.843 | -0.312 | 0.312 | -0.395 |
| systolic_bp | Systolic BP | 0.757 | 0.757 | 0.000 | 0.443 | 0.443 | 0.616 |
| diastolic_bp | Diastolic BP | 0.736 | 0.729 | 0.007 | 0.397 | 0.397 | 0.676 |
| male | Male | 0.550 | 0.100 | 0.450 | 0.000 | 0.030 | -0.158 |

이 단계의 가장 중요한 해석은 다음과 같다.

1. `BMI Risk`와 `Quality of Sleep`는 광범위한 후보군에서도 매우 안정적으로 살아남았다.
2. 혈압 변수는 `Systolic BP`와 `Diastolic BP`가 모두 높게 선택돼, 개별 변수보다 **혈압축 전체가 안정적**이라고 해석하는 편이 맞다.
3. `Sleep Duration`, `Heart Rate`, `Physical Activity Level`은 상황에 따라 대체 가능한 보조 축으로 보인다.

혈압축 selection frequency는 아래처럼 읽는 것이 자연스럽다.

| feature | feature_label | selection_freq | positive_freq | negative_freq | median_coef | median_abs_coef | nonzero_median_coef |
| --- | --- | --- | --- | --- | --- | --- | --- |
| systolic_bp | Systolic BP | 0.757 | 0.757 | 0.000 | 0.443 | 0.443 | 0.616 |
| diastolic_bp | Diastolic BP | 0.736 | 0.729 | 0.007 | 0.397 | 0.397 | 0.676 |

## 5. 최종 모델 변수의 방향성은 얼마나 안정적인가

| feature | feature_label | coef_median | coef_ci_low | coef_ci_high | prob_positive | prob_negative | sign_stability | or_per_1sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diastolic_bp | Diastolic BP | 0.941 | 0.677 | 1.182 | 1.000 | 0.000 | 1.000 | 2.563 |
| bmi_risk | BMI Risk | 0.933 | 0.644 | 1.231 | 1.000 | 0.000 | 1.000 | 2.543 |
| quality_of_sleep | Quality of Sleep | -0.648 | -0.904 | -0.432 | 0.000 | 1.000 | 1.000 | 0.523 |
| age | Age | 0.402 | 0.117 | 0.717 | 1.000 | 0.000 | 1.000 | 1.494 |
| male | Male | -0.208 | -0.519 | 0.179 | 0.132 | 0.868 | 0.868 | 0.812 |
| physical_activity_level | Physical Activity Level | -0.070 | -0.308 | 0.136 | 0.286 | 0.714 | 0.714 | 0.933 |

### 5.1 핵심 해석

아래 변수들은 부호 안정성이 특히 높은 축이었다.

| feature | feature_label | coef_median | coef_ci_low | coef_ci_high | prob_positive | prob_negative | sign_stability | or_per_1sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diastolic_bp | Diastolic BP | 0.941 | 0.677 | 1.182 | 1.000 | 0.000 | 1.000 | 2.563 |
| bmi_risk | BMI Risk | 0.933 | 0.644 | 1.231 | 1.000 | 0.000 | 1.000 | 2.543 |
| quality_of_sleep | Quality of Sleep | -0.648 | -0.904 | -0.432 | 0.000 | 1.000 | 1.000 | 0.523 |
| age | Age | 0.402 | 0.117 | 0.717 | 1.000 | 0.000 | 1.000 | 1.494 |
| male | Male | -0.208 | -0.519 | 0.179 | 0.132 | 0.868 | 0.868 | 0.812 |

- `sign_stability`가 높다는 것은 bootstrap 재표집을 해도 방향이 거의 바뀌지 않는다는 뜻이다.
- `Diastolic BP`가 양(+) 방향으로, `Quality of Sleep`가 음(-) 방향으로 유지된다면 이는 이전 보고서의 핵심 해석을 더 강하게 뒷받침한다.
- 반면 `Physical Activity Level`과 `Male`는 모델에 포함되더라도 계수 방향의 안정성이 상대적으로 더 약하고, `Age`와 `BMI Risk`는 이번 bootstrap에서는 부호가 매우 안정적으로 유지됐다.

## 6. 이번 후속 실험으로 무엇이 더 명확해졌는가

### 6.1 더 강해진 결론

1. `Quality of Sleep`, `BMI Risk`, `혈압축`은 표본 재추출을 해도 가장 안정적으로 남는 핵심 변수군이다.
2. 최종 고정 모델 안에서는 `Diastolic BP`, `Quality of Sleep`, `BMI Risk`, `Age`의 방향성이 특히 안정적이다.
3. 최종 모델 추천은 “단일 절대 승자”보다 “quality-based prediction-first family”로 표현하는 것이 더 엄밀하다.

### 6.2 여전히 남는 한계

1. grouped bootstrap에서도 상위 모델 간 차이가 작으면, 최종 추천은 여전히 목적 의존적이다.
2. stability selection은 L1 벌점의 성질상 상관된 변수들 사이에서 선택을 나눠 가질 수 있다.
3. 따라서 낮은 selection frequency를 곧바로 “중요하지 않다”로 읽으면 안 되고, “중복 정보가 있어 단독 신호가 약하다”로 보는 편이 더 안전하다.

## 7. 최신 기준 최종 문장

이번 후속 검증까지 포함하면 가장 엄밀한 최종 문장은 아래와 같다.

> grouped bootstrap 기준으로는 `Quality + PA`와 `Quality + HR`가 사실상 같은 최상위 quality-based prediction-first family로 남았고, 둘의 차이는 근소했다.

> 변수 수준에서는 `Quality of Sleep`, `BMI Risk`, 혈압축 전체가 가장 안정적이었고, 최종 `Quality + PA` 고정 모델 안에서는 `Diastolic BP`가 특히 안정적인 양(+) 방향 신호로 유지됐다.

## 8. 생성된 결과물

- 보고서: [bootstrap_stability_followup_report_ko.md](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/bootstrap_stability_followup/bootstrap_stability_followup_report_ko.md)
- 시각화: [bootstrap_stability_followup figures](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/bootstrap_stability_followup/figures)
- 표: [bootstrap_stability_followup tables](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/bootstrap_stability_followup/tables)
- 재현 스크립트: [bootstrap_stability_followup.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/bootstrap_stability_followup.py)

마지막 갱신 시각: `2026-04-10 21:21`
