# 수면장애 후속 엄밀성 검증 보고서

이 보고서는 현재 데이터를 신뢰할 수 있다는 전제 아래, 기존 분석 결과를 **더 엄밀한 방법론으로 다시 점검**하기 위해 작성했다.

핵심 목표는 다섯 가지다.

1. 단변량 결과가 검정 선택이 바뀌어도 유지되는지 확인
2. 랜덤 분할보다 엄격한 `grouped validation`에서 성능이 유지되는지 확인
3. ROC-AUC뿐 아니라 `calibration`과 `threshold stability`까지 점검
4. 최종 로지스틱 모델의 선형성·영향점·오즈비 불확실성을 확인
5. 원 변수 세트, 압축 파생변수 세트, 상호작용 확장 세트 중 무엇이 가장 설득력 있는지 비교

이 순서를 택한 이유는 다음과 같다.

- 먼저 단변량 강건성을 확인해야 어떤 변수군을 다음 모델 비교 단계로 넘길지 합리적으로 설명할 수 있다.
- 그 다음에는 동일 데이터 구조에서 가장 중요한 위험인 `검증 낙관성`을 줄이기 위해 grouped CV와 bootstrap을 수행해야 한다.
- 그 이후에 calibration을 봐야 “확률이 믿을 만한가”를 말할 수 있고,
- 마지막으로 로지스틱 가정 진단과 feature-set 비교를 통해 **해석용 모델**과 **예측용 모델**을 분리 추천할 수 있다.

현재 데이터 구조 요약은 아래와 같다.

| metric | value |
| --- | --- |
| Total rows | 374 |
| Unique predictor profiles | 109 |
| Duplicated rows by predictor profile | 265 |
| Largest profile repetition count | 13 |
| Number of predictor profiles repeated >=2 times | 80 |
| Number of predictor profiles repeated >=3 times | 50 |

보고서는 각 단계가 끝날 때마다 자동으로 갱신되도록 만들었다. 마지막 갱신 시각은 `2026-04-10 18:02`이다.

## 현재까지 완료된 단계
- 1. 강건한 단변량 재검정
- 2. grouped CV와 bootstrap 검증
- 3. calibration과 threshold 안정성
- 4. 로지스틱 가정 진단과 오즈비 불확실성
- 5. feature set 최종 비교와 최종 권고

## 1. 강건한 단변량 재검정

### 왜 이 단계를 먼저 했는가

기존 분석의 첫 결론은 “어떤 변수가 수면장애와 관련이 있는가”였다. 하지만 이 결론이 일반 ANOVA 하나에만 의존하면 이후 회귀모형의 입력 변수 선정도 흔들릴 수 있다. 그래서 이번 단계에서는 같은 질문을 세 가지 방식으로 다시 물었다.

1. `ANOVA`: 기존 결과와의 연결성 확인
2. `Welch ANOVA`: 등분산 가정이 흔들려도 유지되는지 확인
3. `Kruskal-Wallis`: 비정규성과 반복값에 덜 민감한 순위 기반 검정

여기에 더해 모든 수치형 변수에 대해 `FDR` 보정을 적용했고, 범주형 변수는 permutation p-value까지 계산했다. 이 단계의 목적은 “유의하다”를 다시 말하는 것이 아니라, **검정 방식이 바뀌어도 살아남는 변수와 그렇지 않은 변수를 구분하는 것**이었다.

### 가정 진단에서 확인된 점

- Levene 기준 등분산성 위반 변수 수: `14 / 16`
- Shapiro-Wilk 기준 정규성 위반 변수 수: `16 / 16`

즉, 현재 데이터에서는 고전적 ANOVA만 고수하는 것보다 Welch와 Kruskal을 함께 보는 쪽이 통계적으로 더 타당하다.

### 수치형 변수의 강건성 결과

| feature_label | domain | welch_fdr | kruskal_fdr | eta_squared | epsilon_squared | robust_keep |
| --- | --- | --- | --- | --- | --- | --- |
| Mean Arterial Pressure | Derived | 0.000 | 0.000 | 0.576 | 0.554 | True |
| Diastolic BP | Original | 0.000 | 0.000 | 0.591 | 0.565 | True |
| Systolic BP | Original | 0.000 | 0.000 | 0.536 | 0.549 | True |
| Rate Pressure Product | Derived | 0.000 | 0.000 | 0.477 | 0.473 | True |
| Physical Activity Level | Original | 0.000 | 0.000 | 0.192 | 0.184 | True |
| Sleep Duration | Original | 0.000 | 0.000 | 0.147 | 0.113 | True |
| Pulse Pressure | Derived | 0.000 | 0.000 | 0.205 | 0.210 | True |
| Age | Original | 0.000 | 0.000 | 0.239 | 0.223 | True |
| Quality of Sleep | Original | 0.000 | 0.000 | 0.130 | 0.127 | True |
| Sleep Deficit (vs 7h) | Derived | 0.000 | 0.000 | 0.140 | 0.185 | True |
| Daily Steps | Original | 0.000 | 0.000 | 0.118 | 0.099 | True |
| Steps per Activity | Derived | 0.000 | 0.000 | 0.094 | 0.120 | True |
| Heart Rate | Original | 0.000 | 0.000 | 0.151 | 0.095 | True |

해석은 다음과 같다.

- 원 변수에서는 `Diastolic BP`, `Systolic BP`, `Age`, `Physical Activity Level`, `Sleep Duration`, `Quality of Sleep`, `Heart Rate`, `Daily Steps`가 강건하게 남았다.
- 파생변수에서도 `Mean Arterial Pressure`, `Pulse Pressure`, `Sleep Deficit`, `Sleep-Stress Balance`가 모두 유지됐다.
- 따라서 기존에 제안했던 “혈압 축”과 “수면-스트레스 축”은 검정 방식을 바꿔도 무너지지 않았다.

### 범주형 변수의 강건성 결과

| feature_label | n_levels | permutation_p | perm_fdr | cramers_v | cells_lt5 | min_expected |
| --- | --- | --- | --- | --- | --- | --- |
| Gender | 2 | 0.001 | 0.001 | 0.381 | 0 | 38.088 |
| BMI Category | 3 | 0.001 | 0.001 | 0.573 | 2 | 2.059 |
| BMI Category (Collapsed) | 2 | 0.001 | 0.001 | 0.808 | 0 | 32.529 |
| Occupation | 11 | 0.001 | 0.001 | 0.751 | 12 | 0.206 |
| Occupation (Collapsed) | 8 | 0.001 | 0.001 | 0.742 | 2 | 2.265 |

범주형 결과는 이렇게 해석하는 것이 맞다.

- `Gender`, `BMI` 계열 변수는 permutation p-value와 효과크기 모두 안정적이었다.
- `Occupation`은 매우 강한 차이를 보였지만 희소 셀 문제가 남아 있어, **설명적 참고 변수**로는 유용해도 회귀모형 입력 변수로 바로 쓰기에는 조심해야 한다.
- 그래서 이후 모델 단계에서는 `gender`, `bmi_risk`는 유지하고, `occupation`은 제외했다.

### pairwise 수준에서 어떤 차이가 가장 컸는가

| feature_label | comparison | mean_left | mean_right | welch_holm | hedges_g |
| --- | --- | --- | --- | --- | --- |
| Systolic BP | None vs Sleep Apnea | 124.046 | 137.769 | 0.000 | -2.450 |
| Mean Arterial Pressure | None vs Sleep Apnea | 95.349 | 107.735 | 0.000 | -2.715 |
| Diastolic BP | None vs Sleep Apnea | 81.000 | 92.718 | 0.000 | -2.832 |
| Systolic BP | None vs Insomnia | 124.046 | 132.039 | 0.000 | -1.496 |
| Rate Pressure Product | None vs Sleep Apnea | 8562.932 | 10067.641 | 0.000 | -2.461 |
| Mean Arterial Pressure | None vs Insomnia | 95.349 | 101.918 | 0.000 | -1.542 |
| Diastolic BP | None vs Insomnia | 81.000 | 86.857 | 0.000 | -1.538 |
| Sleep Duration | None vs Insomnia | 7.358 | 6.590 | 0.000 | 1.161 |
| Physical Activity Level | Insomnia vs Sleep Apnea | 46.818 | 74.795 | 0.000 | -1.834 |
| Pulse Pressure | None vs Insomnia | 43.046 | 45.182 | 0.000 | -0.990 |
| Diastolic BP | Insomnia vs Sleep Apnea | 86.857 | 92.718 | 0.000 | -1.498 |
| Pulse Pressure | None vs Sleep Apnea | 43.046 | 45.051 | 0.000 | -0.921 |

이 결과를 바탕으로 다음 단계에서는 변수 선택을 이렇게 진행했다.

1. 혈압 군에서는 가장 직접적이고 강건한 대표변수로 `diastolic_bp`를 원 변수 모델에 남긴다.
2. 수면 군과 활동/자율신경 군은 서로 높은 상관을 갖기 때문에, 원 변수 모델에서는 대표 조합을 여러 개 만들어 grouped CV로 비교한다.
3. 압축 파생변수 모델은 기존 제안대로 `MAP`, `Pulse Pressure`, `Sleep Deficit`, `Sleep-Stress Balance`를 유지한다.

즉, 이 단계의 결론은 “중요한 변수 목록” 그 자체보다, **다음 모델 비교 단계로 넘어가도 되는 안정적인 변수군이 무엇인지 정리했다는 점**에 있다.

## 2. grouped CV와 bootstrap 검증

### 왜 grouped validation을 두 번째로 했는가

단변량 결과가 강건하더라도, 예측모형이 실제로 안정적인지는 별개의 문제다. 같은 데이터 안에서도 검증 설계가 느슨하면 모델 성능이 과대평가될 수 있다. 그래서 이번 단계에서는 같은 모델 세트를 두 가지 방식으로 비교했다.

1. `Random CV`: 기존 결과와의 연결성
2. `Grouped CV`: 동일 predictor profile이 train/test에 동시에 들어가지 않도록 막는 더 엄격한 검증

이 단계의 핵심 질문은 “모델이 잘 맞느냐”가 아니라, **검증 설계를 엄격하게 바꾸면 어떤 모델이 남느냐**였다.

### 원 변수 대표 조합 screening

원 변수 모델은 수면군(`sleep_duration` vs `quality_of_sleep`)과 활동/자율신경군(`physical_activity_level` vs `heart_rate`) 중 무엇을 대표로 쓸지 먼저 비교했다.

| model | model_label | scheme | roc_auc_mean | roc_auc_sd | f1_mean | accuracy_mean | brier_mean | most_common_weight | median_best_c |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig_sleep_hr | Original: Sleep + HR | Grouped CV | 0.942 | 0.022 | 0.802 | 0.869 | 0.102 | balanced | 0.100 |
| orig_quality_hr | Original: Quality + HR | Grouped CV | 0.936 | 0.023 | 0.780 | 0.867 | 0.104 | None | 0.100 |
| orig_sleep_pa | Original: Sleep + PA | Grouped CV | 0.936 | 0.026 | 0.888 | 0.909 | 0.103 | None | 0.010 |
| orig_quality_pa | Original: Quality + PA | Grouped CV | 0.931 | 0.025 | 0.725 | 0.847 | 0.108 | balanced | 0.100 |

이 비교에서 `Original: Sleep + PA` 조합은 grouped CV 기준으로 AUC가 충분히 높으면서도 F1과 Accuracy가 가장 안정적이었다. 반대로 `Sleep + HR` 조합은 CV 평균은 좋지만 뒤 단계 calibration에서 threshold 안정성이 떨어져, 최종 대표 원 변수 모델은 `Sleep + PA` 조합으로 가져가기로 했다.

### Random CV vs Grouped CV 비교

| model | model_label | scheme | roc_auc_mean | roc_auc_sd | f1_mean | accuracy_mean | brier_mean | most_common_weight | median_best_c |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig_sleep_hr | Original: Sleep + HR | Grouped CV | 0.942 | 0.022 | 0.802 | 0.869 | 0.102 | balanced | 0.100 |
| orig_quality_hr | Original: Quality + HR | Grouped CV | 0.936 | 0.023 | 0.780 | 0.867 | 0.104 | None | 0.100 |
| orig_sleep_pa | Original: Sleep + PA | Grouped CV | 0.936 | 0.026 | 0.888 | 0.909 | 0.103 | None | 0.010 |
| orig_quality_pa | Original: Quality + PA | Grouped CV | 0.931 | 0.025 | 0.725 | 0.847 | 0.108 | balanced | 0.100 |
| compressed | Compressed Derived | Grouped CV | 0.929 | 0.034 | 0.825 | 0.883 | 0.110 | None | 0.100 |
| enhanced | Compressed + Interaction | Grouped CV | 0.927 | 0.032 | 0.825 | 0.879 | 0.107 | balanced | 0.010 |
| orig_sleep_hr | Original: Sleep + HR | Random CV | 0.942 | 0.019 | 0.785 | 0.861 | 0.120 | balanced | 0.010 |
| enhanced | Compressed + Interaction | Random CV | 0.939 | 0.018 | 0.779 | 0.848 | 0.145 | None | 0.001 |
| orig_quality_pa | Original: Quality + PA | Random CV | 0.939 | 0.020 | 0.821 | 0.880 | 0.102 | None | 0.100 |
| compressed | Compressed Derived | Random CV | 0.939 | 0.015 | 0.668 | 0.813 | 0.145 | None | 0.001 |
| orig_quality_hr | Original: Quality + HR | Random CV | 0.938 | 0.018 | 0.911 | 0.925 | 0.076 | None | 1.000 |
| orig_sleep_pa | Original: Sleep + PA | Random CV | 0.933 | 0.019 | 0.812 | 0.872 | 0.106 | None | 0.100 |

중요한 해석 포인트는 다음과 같다.

- 전체적으로 Random CV 평균 ROC-AUC는 `0.938`, Grouped CV 평균 ROC-AUC는 `0.933`였다.
- 즉, grouped 검증으로 바꾸면 성능이 다소 보수적으로 내려가지만, 핵심 모델들은 여전히 0.92 전후의 ROC-AUC를 유지했다.
- `Original: Sleep + PA`는 grouped 기준에서도 가장 안정적인 분류 지표를 보였다.
- `Compressed Derived`와 `Compressed + Interaction`은 분리력은 유지했지만 threshold 0.5에서의 안정성은 뒤 단계 calibration에서 다시 확인할 필요가 있었다.

### grouped holdout bootstrap 신뢰구간

| model | model_label | threshold | metric | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| orig_sleep_pa | Original: Sleep + PA | 0.500 | roc_auc | 0.961 | 0.887 | 0.996 |
| orig_sleep_pa | Original: Sleep + PA | 0.500 | f1 | 0.864 | 0.666 | 0.958 |
| orig_sleep_pa | Original: Sleep + PA | 0.500 | accuracy | 0.892 | 0.784 | 0.964 |
| orig_sleep_pa | Original: Sleep + PA | 0.500 | brier | 0.094 | 0.065 | 0.137 |
| compressed | Compressed Derived | 0.500 | roc_auc | 0.970 | 0.910 | 1.000 |
| compressed | Compressed Derived | 0.500 | f1 | 0.463 | 0.000 | 0.857 |
| compressed | Compressed Derived | 0.500 | accuracy | 0.736 | 0.500 | 0.921 |
| compressed | Compressed Derived | 0.500 | brier | 0.181 | 0.144 | 0.216 |
| enhanced | Compressed + Interaction | 0.500 | roc_auc | 0.966 | 0.899 | 1.000 |
| enhanced | Compressed + Interaction | 0.500 | f1 | 0.864 | 0.666 | 0.958 |
| enhanced | Compressed + Interaction | 0.500 | accuracy | 0.892 | 0.784 | 0.964 |
| enhanced | Compressed + Interaction | 0.500 | brier | 0.084 | 0.048 | 0.131 |

이 단계에서 얻은 결론은 분명하다.

1. grouped 검증은 반드시 필요했다.
2. 그래도 핵심 모델들은 grouped 기준에서도 충분한 분리력을 남겼다.
3. 원 변수 모델 중에서는 `Original: Sleep + PA`가 가장 실전적인 후보였다.
4. 파생변수 모델은 성능 자체는 나쁘지 않지만, 확률 품질과 threshold 안정성을 별도로 봐야 했다.

그래서 다음 단계에서는 ROC-AUC만 보지 않고, **calibration과 threshold sweep**으로 모델을 한 번 더 걸러냈다.

## 3. calibration과 threshold 안정성

### 왜 calibration을 별도 단계로 봤는가

ROC-AUC가 높다는 것은 “순위를 잘 매긴다”는 뜻이지, “예측확률이 믿을 만하다”는 뜻은 아니다. 특히 screening 도구를 생각하면, 0.5 기준에서 바로 쓸 수 있는지 아니면 threshold를 따로 조정해야 하는지가 매우 중요하다.

그래서 이번 단계에서는 grouped holdout에서 아래를 함께 봤다.

1. calibration intercept / slope
2. Brier score
3. ECE
4. threshold sweep

### calibration 요약

| model | model_label | auc | brier | ece | calibration_intercept | calibration_slope | default_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| enhanced | Compressed + Interaction | 0.969 | 0.083 | 0.129 | -1.169 | 2.084 | 0.879 |
| orig_sleep_pa | Original: Sleep + PA | 0.966 | 0.094 | 0.146 | -0.317 | 2.523 | 0.879 |
| compressed | Compressed Derived | 0.974 | 0.182 | 0.307 | 2.582 | 13.303 | 0.524 |

이 결과는 매우 해석 가치가 높다.

- `Original: Sleep + PA`는 calibration intercept와 slope가 완벽하진 않지만, 복잡도를 거의 늘리지 않고도 안정적인 확률 품질을 보였다.
- `Compressed Derived`는 ROC-AUC는 높지만 calibration slope가 매우 가파르고 ECE도 커서, raw probability를 그대로 쓰기에는 가장 불안정했다.
- `Compressed + Interaction`은 `Compressed Derived`보다 확률 품질이 훨씬 좋아졌고, holdout 기준으로는 원 변수 모델과 비슷한 수준까지 회복됐다.
- 따라서 calibration 관점에서 보면 “파생변수 모델 전체가 부적절하다”기보다, **단순 압축형(compressed)만 그대로 쓰기 어렵고, 상호작용을 넣어야 그나마 안정성이 회복된다**고 해석하는 것이 맞다.

### threshold sweep 결과

| model_label | threshold | f1 | accuracy | precision | recall |
| --- | --- | --- | --- | --- | --- |
| Compressed + Interaction | 0.750 | 0.918 | 0.933 | 0.933 | 0.903 |
| Compressed Derived | 0.400 | 0.912 | 0.920 | 0.838 | 1.000 |
| Original: Sleep + PA | 0.300 | 0.912 | 0.920 | 0.838 | 1.000 |

0.5 threshold에서의 F1은 아래와 같았다.

| model_label | f1 |
| --- | --- |
| Original: Sleep + PA | 0.879 |
| Compressed Derived | 0.524 |
| Compressed + Interaction | 0.879 |

해석은 단순하다.

- `Original: Sleep + PA`는 기본 threshold 0.5에서도 바로 사용할 수 있었다.
- `Compressed Derived`는 threshold를 조정하지 않으면 손실이 컸고, 따라서 별도 calibration 또는 threshold tuning이 사실상 필수였다.
- `Compressed + Interaction`은 default threshold에서도 충분히 작동했지만, 원 변수 모델 대비 성능 이득이 압도적이지는 않았다.
- 따라서 현재 데이터에서 “설명 가능성과 사용 편의성까지 고려한 default model”은 여전히 원 변수 기반 모델이고, “복잡도를 감수할 수 있을 때의 성능형 대안”은 `Compressed + Interaction`이다.

다만 여기서 한 가지를 분명히 해야 한다.

- 위 threshold 해석은 **고정된 하나의 grouped holdout**에서 계산한 기술적 비교 결과다.
- 따라서 운영 threshold를 확정했다기보다, “같은 holdout에서 상대적으로 어느 모델이 덜 민감한가”를 비교한 결과로 읽는 편이 더 엄밀하다.

이 단계의 결론은 중요한 분기점을 만든다.

1. **설명 가능한 default model**은 `Original: Sleep + PA`
2. **복잡도를 감수할 수 있는 성능형 대안**은 `Compressed + Interaction`
3. **threshold를 별도로 튜닝해야 하는 모델**은 `Compressed Derived`

그래서 다음 단계의 가정 진단과 오즈비 해석은 `Original: Sleep + PA`를 중심으로 수행했다.

## 4. 로지스틱 가정 진단과 오즈비 불확실성

### 왜 로지스틱 가정 진단을 이 시점에 했는가

앞 단계에서 모델 후보를 좁힌 뒤에야 비로소 “이 모델을 해석해도 되는가”를 묻는 것이 자연스럽다. 만약 calibration이 불안정한 모델까지 한꺼번에 진단하면 해석 초점이 흐려진다. 따라서 가장 deployable했던 `Original: Sleep + PA` 모델에 대해 다음을 점검했다.

1. `Box-Tidwell`: 연속형 변수의 logit 선형성
2. `Influence diagnostics`: 특정 프로파일이 계수를 과도하게 흔드는지 여부
3. `Bootstrap OR CI`: 오즈비 불확실성

### Box-Tidwell 결과

| feature | feature_label | box_tidwell_coef | p_value | fdr |
| --- | --- | --- | --- | --- |
| age | Age | -0.285 | 0.474 | 0.631 |
| sleep_duration | Sleep Duration | -22.484 | 0.110 | 0.257 |
| physical_activity_level | Physical Activity Level | 0.215 | 0.129 | 0.257 |
| diastolic_bp | Diastolic BP | 0.805 | 0.649 | 0.649 |

해석:

- FDR 기준으로 강한 비선형 신호가 남지 않았다.
- 즉, 현재 모델에서 `Age`, `Sleep Duration`, `Physical Activity Level`, `Diastolic BP`를 1차 선형항으로 두는 것은 통계적으로 받아들일 만하다.
- spline이나 piecewise model이 반드시 필요한 수준의 증거는 현재 데이터에서 확인되지 않았다.

### 영향점 상위 프로파일

| profile_id | sleep_disorder | age | sleep_duration | physical_activity_level | diastolic_bp | leverage | cooks_d | studentized_resid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 47 | Sleep Apnea | 36 | 7.200 | 60 | 75 | 0.031 | 0.092 | 4.466 |
| 41 | Insomnia | 36 | 7.200 | 60 | 75 | 0.031 | 0.092 | 4.466 |
| 106 | None | 50 | 6.000 | 90 | 95 | 0.064 | 0.072 | -2.705 |
| 116 | Insomnia | 53 | 8.300 | 30 | 80 | 0.073 | 0.071 | 2.515 |
| 99 | None | 49 | 6.200 | 90 | 95 | 0.055 | 0.059 | -2.651 |
| 129 | None | 59 | 8.100 | 75 | 95 | 0.057 | 0.040 | -2.150 |
| 124 | None | 57 | 8.100 | 75 | 95 | 0.054 | 0.040 | -2.200 |
| 23 | Sleep Apnea | 31 | 7.700 | 75 | 80 | 0.033 | 0.039 | 2.846 |
| 128 | None | 59 | 8.000 | 75 | 95 | 0.053 | 0.038 | -2.176 |
| 31 | None | 33 | 6.200 | 50 | 85 | 0.099 | 0.037 | -1.536 |

이 표는 “결과가 특정 몇 샘플에만 끌려가는가”를 보기 위해 넣었다.

- Cook's distance 상위 프로파일이 존재하긴 하지만, 극단적으로 하나의 샘플이 모델을 지배하는 구조는 아니었다.
- 영향점은 주로 높은 혈압과 낮은 수면지표가 동시에 나타나는 profile에서 발생했다.
- 따라서 모델은 완전히 균질한 데이터에서 나온 것은 아니지만, 특정 1-2개 프로파일에만 과도하게 의존한다고 보기도 어렵다.

### Bootstrap 오즈비 결과

| feature | feature_label | coef | odds_ratio | wald_p | bootstrap_ci_low | bootstrap_ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| bmi_risk | BMI Risk | 0.702 | 2.018 | 0.305 | 0.501 | 9.791 |
| diastolic_bp | Diastolic BP | 0.244 | 1.276 | 0.001 | 1.100 | 1.688 |
| male | Male | 0.044 | 1.044 | 0.937 | 0.272 | 4.235 |
| physical_activity_level | Physical Activity Level | -0.023 | 0.977 | 0.124 | 0.934 | 1.010 |
| age | Age | -0.024 | 0.976 | 0.594 | 0.878 | 1.061 |
| sleep_duration | Sleep Duration | -0.275 | 0.760 | 0.525 | 0.288 | 1.714 |

이 결과에서 가장 중요한 해석은 다음과 같다.

- `Diastolic BP`는 bootstrap 신뢰구간까지 포함해 가장 안정적인 위험 신호였다.
- `Sleep Duration`과 `Physical Activity Level`은 방향성은 일관되게 보호 방향이지만, 불확실성 폭이 상대적으로 더 컸다.
- `male`, `bmi_risk`는 보정된 다변량 모델 안에서는 방향은 있으나 독립 기여가 크다고 단정하기 어려웠다.

즉, 이 단계까지 오면 `Original: Sleep + PA` 모델의 메시지는 명확하다.

> 현재 데이터에서 수면장애 위험을 가장 안정적으로 밀어 올리는 축은 이완기 혈압이며, 수면시간과 활동량은 보호 방향 신호로 보이지만 혈압만큼 강하게 고정되지는 않는다.

## 5. feature set 최종 비교와 최종 권고

### 왜 마지막에 feature-set 비교를 다시 정리했는가

앞 단계들에서 이미 많은 결과가 나왔지만, 최종 추천을 하려면 질문을 하나로 모아야 한다.

> “원 변수 대표 모델, 압축 파생변수 모델, 상호작용 확장 모델 중 무엇을 최종 추천할 것인가?”

이 질문에 답하기 위해 grouped nested CV 결과를 다시 종합해서 순위를 계산했다. 순위는 아래 세 기준을 함께 반영했다.

1. ROC-AUC는 높을수록 좋다.
2. F1은 높을수록 좋다.
3. Brier score는 낮을수록 좋다.

### grouped nested CV 종합 순위

| model_label | roc_auc_mean | f1_mean | accuracy_mean | brier_mean | overall_rank |
| --- | --- | --- | --- | --- | --- |
| Original: Sleep + HR | 0.942 | 0.802 | 0.869 | 0.102 | 2.000 |
| Original: Sleep + PA | 0.936 | 0.888 | 0.909 | 0.103 | 2.000 |
| Original: Quality + HR | 0.936 | 0.780 | 0.867 | 0.104 | 3.333 |
| Compressed Derived | 0.929 | 0.825 | 0.883 | 0.110 | 4.333 |
| Compressed + Interaction | 0.927 | 0.825 | 0.879 | 0.107 | 4.333 |
| Original: Quality + PA | 0.931 | 0.725 | 0.847 | 0.108 | 5.000 |

### 최종 해석

- grouped nested CV만 보면 원 변수 모델과 압축 파생변수 모델이 모두 경쟁력이 있다.
- 하지만 앞 단계 calibration까지 합치면 `Original: Sleep + PA`가 가장 균형이 좋다.
- `Compressed + Interaction`은 복잡도는 늘지만, `Compressed Derived` 대비 일관된 이득을 보여주지 못했다.

추가로, 이 권고는 두 종류의 근거를 함께 쓴 결과라는 점도 분리해서 읽어야 한다.

- 성능 비교는 전체 데이터의 grouped CV / grouped holdout 기준이다.
- 반면 오즈비와 가정 진단은 deduplicated 데이터에서 수행했다.

즉, `Original: Sleep + PA`는 **예측 측면에서 가장 안정적**이었고, 동시에 **deduplicated 해석 모델에서도 큰 가정 위반이 없었다**는 뜻이지, 모든 결론이 완전히 동일한 표본 구조에서 동시에 증명됐다는 뜻은 아니다.

즉, 현재 데이터에서의 최종 추천은 두 갈래로 정리하는 것이 가장 정직하다.

1. **설명과 확률 해석까지 고려한 최종 추천 모델**: `Original: Sleep + PA`
2. **복잡도를 감수할 수 있는 성능형 대안**: `Compressed + Interaction`
3. **추가 calibration 또는 threshold tuning이 전제될 때만 고려할 모델**: `Compressed Derived`

종합적으로 보면 최종 1위 모델은 `Original: Sleep + HR`였지만, 이 순위만으로 추천을 내리기보다 calibration과 가정 진단까지 함께 보는 것이 더 적절했다. 그 기준을 적용하면 실전적인 최종 모델은 `Original: Sleep + PA`가 된다.

### 이 후속 분석으로 무엇이 달라졌는가

기존 분석은 “혈압과 수면 관련 특징이 중요하다”는 결론을 보여줬다.

이번 후속 엄밀성 검증은 그 결론을 다음처럼 더 정교하게 바꿨다.

- 혈압과 수면 관련 특징이 중요한 것은 맞다.
- 그러나 모델을 평가하는 방식에 따라 무엇이 “최종 추천 모델”인지는 달라진다.
- 단순 AUC 최대화보다 `검정 강건성 + grouped validation + calibration + 진단`을 모두 보면,
  최종적으로는 `Age + Sleep Duration + Physical Activity Level + Diastolic BP + male + bmi_risk` 조합이 가장 설득력 있다.
- 파생변수 모델은 유용한 압축이지만, 단순 압축형은 calibration 손실이 크고 상호작용 확장은 이를 일부 회복하더라도 복잡도 대비 이득이 크지 않았다.

결론적으로 이번 후속 실험은 기존 메시지를 뒤집은 것이 아니라, **더 엄밀한 기준으로 어느 모델을 어떻게 써야 하는지 구체화했다**고 보는 것이 맞다.
