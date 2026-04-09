# 데이터셋 선택 근거, 다항 로지스틱 회귀, 민감도 분석 보고서

## 1. 왜 이 데이터셋을 선택했는가

`dataset` 폴더에는 두 종류의 데이터가 있었다.

1. `Sleep_health_and_lifestyle_dataset`
2. `FitBit Fitness Tracker Data`

통계분석 관점에서 본 분석 데이터로 `Sleep_health_and_lifestyle_dataset`를 선택한 이유는 아래와 같다.

1. **명시적 종속변수 존재**
   - `Sleep Disorder`가 직접 포함되어 있어 분류모형과 집단 비교가 가능하다.
   - 반면 Fitbit 데이터는 활동/수면 로그는 풍부하지만 수면장애 라벨이 없고, 첫 번째 데이터와 개인 단위 연결도 없다.
2. **예측변수 완전성**
   - 주요 예측변수에는 결측이 없고, `Sleep Disorder` 결측은 실제로 `None` 의미로 쓰여 재코딩 가능했다.
3. **연속형 + 범주형 변수가 함께 존재**
   - ANOVA, 카이제곱, 상관분석, 로지스틱 회귀를 한 데이터셋 안에서 일관되게 수행하기 좋다.
4. **이진/다항 분석 모두 가능한 타깃 구조**
   - `None / Insomnia / Sleep Apnea` 3범주가 있어 이진 로지스틱과 다항 로지스틱을 모두 적용할 수 있다.
5. **클래스 불균형이 극단적이지 않음**
   - 전체 데이터 기준 `None=219`, `Insomnia=77`, `Sleep Apnea=78`로, 다항 분류를 시도할 최소 구조를 갖춘다.

데이터 요약:

| dataset | n_rows | n_none | n_insomnia | n_sleep_apnea | positive_share |
| --- | --- | --- | --- | --- | --- |
| Full | 374 | 219 | 77 | 78 | 0.414 |
| Deduplicated | 132 | 73 | 29 | 30 | 0.447 |

## 2. 왜 이 데이터셋이 로지스틱 회귀에 적합한가

로지스틱 회귀의 특성에 비추어 보면 이 데이터셋은 아래 이유로 적합하다.

1. **종속변수가 범주형**
   - 로지스틱 회귀는 연속형이 아니라 범주형 결과를 설명하는 모델이며, 이 데이터는 `수면장애 여부`와 `수면장애 subtype`을 모두 제공한다.
2. **설명변수가 해석 가능한 임상/생활습관 변수**
   - 나이, 심박수, 혈압, 수면시간, 수면질, 스트레스, BMI는 회귀계수 해석이 직관적이다.
3. **표본 수 대비 변수 수가 과도하지 않음**
   - 특히 압축형 파생변수를 사용하면 공선성을 줄이면서도 모델 규모를 관리할 수 있다.
4. **클래스 균형이 완전히 무너지지 않음**
   - 극단적 희소 클래스가 아니어서 softmax 기반 다항 로지스틱을 적용할 수 있다.

다만 완전한 이상적 조건은 아니다.

- `person_id`를 제외하면 동일 프로파일이 많이 반복되어 독립성 가정이 약해질 수 있다.
- 혈압과 수면 관련 변수들 사이 공선성이 존재한다.

이 한계를 보완하기 위해 이번 분석에서는

1. 혈압군과 수면-스트레스군을 압축한 파생변수 사용
2. 중복 프로파일 제거 전후 민감도 분석

을 함께 수행했다.

## 3. 분석 흐름

1. 다항 로지스틱 회귀에 투입할 **압축형 변수 세트**를 사용했다.
   - `Age`, `Heart Rate`, `Mean Arterial Pressure`, `Pulse Pressure`, `Sleep Deficit (vs 7h)`, `Sleep-Stress Balance`, `Male`, `BMI Risk`
2. 같은 구조를 사용해
   - `Full data`
   - `Deduplicated data`
   를 각각 적합했다.
3. 결과는 두 층위에서 해석했다.
   - `MNLogit`: subtype별 오즈비와 p-value
   - `Multinomial logistic classifier`: 예측 성능과 계수 안정성

## 4. 다항 로지스틱 회귀 결과

### 4.1 Full data에서 유의했던 변수

| comparison | feature_label | odds_ratio | ci_low | ci_high | p_value |
| --- | --- | --- | --- | --- | --- |
| Insomnia vs None | BMI Risk | 34.639 | 8.333 | 143.986 | <0.001 |
| Insomnia vs None | Sleep-Stress Balance | 0.372 | 0.207 | 0.666 | <0.001 |
| Insomnia vs None | Pulse Pressure | 2.528 | 1.455 | 4.392 | 0.001 |
| Insomnia vs None | Heart Rate | 0.728 | 0.600 | 0.882 | 0.001 |
| Insomnia vs None | Sleep Deficit (vs 7h) | 0.025 | 0.001 | 0.609 | 0.024 |
| Sleep Apnea vs None | Mean Arterial Pressure | 1.538 | 1.239 | 1.910 | <0.001 |
| Sleep Apnea vs None | Heart Rate | 1.247 | 1.053 | 1.476 | 0.011 |
| Sleep Apnea vs None | Pulse Pressure | 0.618 | 0.387 | 0.987 | 0.044 |

해석:

- `Insomnia vs None`에서는 `Pulse Pressure` 증가, `BMI Risk` 증가, `Sleep-Stress Balance` 저하가 주요 신호였다.
- `Sleep Apnea vs None`에서는 `Mean Arterial Pressure` 증가와 `Heart Rate` 증가가 더 강하게 나타났다.
- 즉, **불면증은 수면-스트레스 균형과 상대적 압력차**, **수면무호흡은 평균 혈압부담과 심혈관 부하** 쪽에 더 가깝게 연결된다.

### 4.2 Deduplicated data에서 유의했던 변수

| comparison | feature_label | odds_ratio | ci_low | ci_high | p_value |
| --- | --- | --- | --- | --- | --- |
| Insomnia vs None | Pulse Pressure | 2.355 | 1.358 | 4.087 | 0.002 |
| Insomnia vs None | Sleep-Stress Balance | 0.502 | 0.278 | 0.908 | 0.023 |
| Insomnia vs None | BMI Risk | 5.120 | 1.021 | 25.672 | 0.047 |
| Sleep Apnea vs None | Mean Arterial Pressure | 1.322 | 1.070 | 1.633 | 0.010 |

해석:

- 표본 수가 줄어 p-value는 전반적으로 약해졌지만,
- `Pulse Pressure`, `Sleep-Stress Balance`, `BMI Risk`의 불면증 방향성,
- `Mean Arterial Pressure`의 수면무호흡 방향성은 유지되었다.

## 5. subtype별 원자료 평균

| sleep_disorder | age | heart_rate | map_bp | pulse_pressure | sleep_deficit_7h | sleep_stress_balance |
| --- | --- | --- | --- | --- | --- | --- |
| None | 39.040 | 69.020 | 95.350 | 43.050 | 0.170 | 2.510 |
| Insomnia | 43.520 | 70.470 | 101.920 | 45.180 | 0.470 | 0.660 |
| Sleep Apnea | 49.710 | 73.090 | 107.740 | 45.050 | 0.450 | 1.540 |

원자료 수준 해석:

- `Sleep Apnea`는 평균적으로 가장 나이가 많고, 심박수와 평균동맥압이 가장 높았다.
- `Insomnia`는 `None` 대비 수면-스트레스 균형이 더 나빴다.
- `Sleep Deficit`는 `Insomnia`와 `Sleep Apnea` 모두 `None`보다 높았고, 특히 `Insomnia`에서 더 나쁜 패턴이 보였다.

## 6. 민감도 분석

이번 민감도 분석은 **중복 프로파일 제거 전후** 결과 비교다.

### 6.1 효과크기 민감도

| dataset | feature_label | eta_squared | p_value |
| --- | --- | --- | --- |
| Full | Age | 0.239 | <0.001 |
| Full | Heart Rate | 0.151 | <0.001 |
| Full | Mean Arterial Pressure | 0.576 | <0.001 |
| Full | Pulse Pressure | 0.205 | <0.001 |
| Full | Sleep Deficit (vs 7h) | 0.140 | <0.001 |
| Full | Sleep-Stress Balance | 0.067 | <0.001 |
| Deduplicated | Age | 0.059 | 0.020 |
| Deduplicated | Heart Rate | 0.165 | <0.001 |
| Deduplicated | Mean Arterial Pressure | 0.348 | <0.001 |
| Deduplicated | Pulse Pressure | 0.221 | <0.001 |
| Deduplicated | Sleep Deficit (vs 7h) | 0.102 | <0.001 |
| Deduplicated | Sleep-Stress Balance | 0.080 | 0.005 |

해석:

- `Mean Arterial Pressure`, `Pulse Pressure`, `Heart Rate`, `Sleep Deficit`, `Sleep-Stress Balance`는 dedup 후에도 유의했다.
- 즉, 예측 성능은 줄어들더라도 **주요 변수와 수면장애 subtype 간의 통계적 분리 자체는 유지**되었다.

### 6.2 예측 성능 민감도

| dataset | cv_macro_f1_mean | cv_macro_f1_std | test_accuracy | test_macro_f1 | test_auc_ovr_macro | best_c |
| --- | --- | --- | --- | --- | --- | --- |
| Full | 0.858 | 0.058 | 0.920 | 0.898 | 0.933 | 1.000 |
| Deduplicated | 0.612 | 0.158 | 0.650 | 0.581 | 0.703 | 1.000 |

해석:

- `Full data`에서는 macro-F1과 accuracy가 높게 나왔다.
- 하지만 `Deduplicated data`에서는 성능이 뚜렷하게 하락했다.
- 따라서 **중복 프로파일이 모델 성능을 낙관적으로 보이게 했을 가능성**이 크다.

### 6.3 계수 방향 안정성

클래스별 계수 부호 일치 비율:

| class | sign_same |
| --- | --- |
| Insomnia | 0.875 |
| None | 0.875 |
| Sleep Apnea | 0.875 |

해석:

- dedup 후에도 핵심 방향성은 꽤 유지되었다.
- 특히 `Sleep Apnea` 쪽의 `MAP` 양의 방향, `Insomnia` 쪽의 `Pulse Pressure` 양의 방향과 `Sleep-Stress Balance` 음의 방향은 비교적 안정적이었다.

## 7. 최종 정리

### 7.1 데이터셋 선택에 대한 결론

이 데이터셋은

1. 명시적 범주형 타깃이 있고
2. 예측변수 결측이 거의 없으며
3. 집단비교와 로지스틱 회귀를 함께 수행할 수 있다는 점에서

통계분석용으로 적합했다.

### 7.2 로지스틱 회귀 적합성에 대한 결론

특히 이 데이터는

1. `이진 로지스틱`으로는 수면장애 유무
2. `다항 로지스틱`으로는 `Insomnia / Sleep Apnea / None`

를 모두 다룰 수 있어 로지스틱 회귀 교육용/해석용 예제로도 적합하다.

### 7.3 다항 로지스틱과 민감도 분석을 통해 얻은 결론

- subtype 차이는 단순히 “수면장애가 있다/없다”보다 더 구체적이다.
- `Sleep Apnea`는 혈압부담(MAP)과 심혈관 패턴이 더 강하고,
- `Insomnia`는 수면-스트레스 불균형과 상대적 압력차(Pulse Pressure)와 더 가깝다.
- 다만 중복 제거 후 성능이 크게 낮아졌으므로, **예측 성능 수치는 보수적으로 해석해야 한다.**
- 반대로 주요 변수의 방향성과 효과크기는 상당 부분 유지되어, **어떤 feature가 중요한가**에 대한 결론은 비교적 견고하다.

## 8. 생성된 결과물

### 8.1 보고서

- `results/multinomial_sensitivity_study/multinomial_sensitivity_report_ko.md`

### 8.2 시각화

1. `results/multinomial_sensitivity_study/figures/01_class_balance_full_vs_dedup.png`
2. `results/multinomial_sensitivity_study/figures/02_selected_features_by_class.png`
3. `results/multinomial_sensitivity_study/figures/03_multinomial_odds_ratios_full.png`
4. `results/multinomial_sensitivity_study/figures/04_effect_size_sensitivity.png`
5. `results/multinomial_sensitivity_study/figures/05_multinomial_performance_sensitivity.png`
6. `results/multinomial_sensitivity_study/figures/06_coefficient_stability_heatmap.png`
7. `results/multinomial_sensitivity_study/figures/07_confusion_matrices_full_vs_dedup.png`
