# 수면장애 통계분석 요약

## 1. 어떤 데이터로 분석했는가

- `dataset/Sleep_health_and_lifestyle_dataset/Sleep_health_and_lifestyle_dataset.csv`를 본 분석 데이터로 사용했다.
- `dataset/FitBit Fitness Tracker Data`도 함께 확인했지만, 이 데이터는 활동량/수면분 단위 로그 중심이며 `Sleep Disorder` 라벨이 없고 첫 번째 데이터와 개인 단위로 연결되지 않아 수면장애 분류모델의 직접 학습 데이터로 쓰기에는 부적절했다.

## 2. 왜 이런 순서로 분석했는가

`reference` 폴더의 강의노트 흐름을 기준으로 분석 순서를 설계했다.

1. 3주차 자료의 흐름에 맞춰 기술통계와 집단 간 비교를 먼저 수행했다.
   - 이유: 모델을 만들기 전에 어떤 변수가 `None`, `Insomnia`, `Sleep Apnea` 집단을 실제로 구분하는지 확인해야 한다.
2. 4주차 자료의 흐름에 맞춰 상관분석과 다중공선성(VIF) 점검을 수행했다.
   - 이유: 유의한 변수라도 서로 너무 비슷한 정보를 담으면 회귀계수가 불안정해질 수 있기 때문이다.
3. 5주차 자료의 흐름에 맞춰 로지스틱 회귀를 수행했다.
   - 이유: 최종 목표가 `수면장애 여부`를 분류하고, 어떤 변수 조합이 가장 해석 가능하고 실용적인지 제시하는 것이기 때문이다.

즉, 이번 분석은 `기술통계 -> 집단 간 차이 검정 -> 상관분석 -> 다중공선성 점검 -> 로지스틱 회귀` 순서로 진행했다.

## 3. 데이터 전처리와 해석 시 주의점

- 총 374행이며, `Sleep Disorder` 결측치는 실제로 `None` 의미로 사용되어 `비수면장애군`으로 처리했다.
- 혈압은 `Blood Pressure` 문자열을 `Systolic BP`, `Diastolic BP`로 분리했다.
- `BMI Category`의 `Normal Weight`는 `Normal`로 통합했다.
- `person_id`를 제외하면 동일 프로파일이 242건 반복되어 있어, 이 데이터는 실제 임상 코호트라기보다 템플릿형 또는 합성형 구조의 성격이 강하다.
- 따라서 방향성 해석은 유효하지만, p-value가 실제 임상자료보다 더 강하게 보일 가능성은 있다.

## 4. 수면장애와 어떤 feature가 관련이 있었는가

### 4.1 집단 간 차이 분석(ANOVA)

`None`, `Insomnia`, `Sleep Apnea` 3집단 간 평균 차이를 비교한 결과, 가장 큰 분리력을 보인 변수는 다음과 같았다.

1. `Diastolic BP`: eta squared = 0.591
2. `Systolic BP`: eta squared = 0.536
3. `Age`: eta squared = 0.239
4. `Physical Activity Level`: eta squared = 0.192
5. `Heart Rate`: eta squared = 0.151

추가로 `Sleep Duration`, `Quality of Sleep`, `Stress Level`도 모두 유의했다.

의미:
- 수면장애 집단은 혈압, 나이, 심박수처럼 심혈관/생리학적 변수에서 뚜렷한 차이를 보였다.
- 동시에 수면시간과 수면의 질, 스트레스 수준 역시 장애 여부와 함께 변하는 핵심 변수였다.

### 4.2 범주형 변수 연관성(카이제곱 검정)

범주형 변수의 연관성은 다음과 같았다.

1. `Occupation`: Cramer's V = 0.751
2. `BMI Category`: Cramer's V = 0.573
3. `Gender`: Cramer's V = 0.381

해석:
- `Occupation`은 통계적으로는 매우 강했지만, 일부 직업군 표본 수가 너무 작아 최종 회귀모형 변수로 쓰기에는 불안정했다.
- `BMI Category`와 `Gender`는 수면장애와 유의한 연관성을 보였고, 후보 변수로 고려할 만했다.

### 4.3 상관분석(point-biserial)

`Has Sleep Disorder`와의 상관계수 기준으로 강한 관련성을 보인 변수는 다음과 같았다.

1. `Diastolic BP`: r = 0.705
2. `Systolic BP`: r = 0.692
3. `Age`: r = 0.432
4. `Sleep Duration`: r = -0.339
5. `Heart Rate`: r = 0.330
6. `Quality of Sleep`: r = -0.311
7. `Stress Level`: r = 0.182

해석:
- 양의 상관: 나이, 혈압, 심박수, 스트레스가 높을수록 수면장애 가능성이 커지는 방향
- 음의 상관: 수면시간과 수면의 질이 낮을수록 수면장애 가능성이 커지는 방향

## 5. 로지스틱 회귀로 분류할 때 어떤 변수를 써야 하는가

### 5.1 변수 선택 기준

최종 로지스틱 회귀 변수는 다음 기준으로 골랐다.

1. 앞선 ANOVA, 카이제곱, 상관분석에서 유의하거나 효과크기가 충분할 것
2. 해석이 쉬울 것
3. 서로 지나치게 겹치는 정보가 아닐 것
4. 범주 수가 너무 작아 불안정한 변수는 제외할 것

### 5.2 최종 선택 변수

최종 분류모형에 남은 변수는 아래 3개였다.

1. `Age`
2. `Diastolic BP`
3. `Quality of Sleep`

선정 이유:
- `Age`: 단순 상관과 집단 비교 모두에서 유의했고, 다른 변수들을 함께 넣어도 독립적인 설명력이 남았다.
- `Diastolic BP`: 가장 강한 연관성을 보인 혈압 변수였고, 수면장애 위험 증가 방향이 분명했다.
- `Quality of Sleep`: 보호요인으로 작용했으며, 수면 관련 핵심 지표 중 가장 안정적으로 남았다.

### 5.3 다중공선성 점검

- `Age` VIF = 2.832
- `Diastolic BP` VIF = 2.223
- `Quality of Sleep` VIF = 1.856

모두 5 미만이므로 최종 모형은 공선성 문제 없이 해석 가능한 수준이다.

### 5.4 로지스틱 회귀 계수 해석

- `Quality of Sleep`: OR = 0.211
  - 수면의 질이 1단위 높아질수록 수면장애 오즈는 약 79% 감소하는 방향으로 해석된다.
- `Diastolic BP`: OR = 1.379
  - 이완기혈압이 1단위 높아질수록 수면장애 오즈는 약 37.9% 증가하는 방향이다.
- `Age`: OR = 1.175
  - 나이가 1세 증가할수록 수면장애 오즈는 약 17.5% 증가하는 방향이다.

## 6. 모델 성능은 어땠는가

- 5-fold CV ROC-AUC = 0.945 +/- 0.020
- Test ROC-AUC = 0.971
- Accuracy = 0.903
- Precision = 0.875
- Recall = 0.894
- F1-score = 0.884

해석:
- 해석 가능한 3개 변수만으로도 수면장애 여부를 상당히 잘 구분했다.
- 다만 데이터 반복 패턴이 많아 실제 외부 데이터에서는 성능이 다소 낮아질 수 있으므로, 현재 성능은 내부 기준의 참고치로 보는 것이 안전하다.

## 7. 최종 인사이트

### 7.1 무엇을 알 수 있었는가

이번 통계분석을 통해 다음을 확인할 수 있었다.

1. 수면장애는 단순히 `잠을 적게 자는 것`만의 문제가 아니라, 혈압·심박수·스트레스·BMI 같은 전반적 생리/생활습관 요인과 함께 움직인다.
2. 단변량 수준에서는 `수면시간`, `수면의 질`, `스트레스`, `혈압`, `심박수`, `BMI`가 모두 중요했다.
3. 그러나 여러 변수를 동시에 고려하는 로지스틱 회귀에서는 정보가 겹치는 변수들이 정리되면서, 최종적으로 `나이`, `이완기혈압`, `수면의 질`이 가장 안정적인 핵심 변수로 남았다.

### 7.2 이 결과를 어떻게 활용할 수 있는가

이 결과를 기반으로 다음과 같은 도움을 줄 수 있다.

1. 간단한 설문과 활력징후만으로 수면장애 고위험군을 1차 선별할 수 있다.
2. 수면의 질이 낮고 혈압이 높으며 연령이 높은 사람을 우선적으로 정밀검사 대상으로 추천할 수 있다.
3. 생활습관 개선 프로그램에서는 `수면의 질 향상`, `스트레스 관리`, `혈압 관리`를 우선 intervention 포인트로 설계할 수 있다.
4. 의료현장이나 헬스케어 서비스에서 전문 검사 이전의 사전 스크리닝 모델로 활용할 수 있다.

## 8. 생성된 결과물

### 8.1 보고서

- `results/sleep_disorder_statistical_report.md`: 전체 영문 상세 보고서
- `results/sleep_disorder_statistical_summary_ko.md`: 한국어 요약본

### 8.2 테이블

- `results/tables/anova_results.csv`
- `results/tables/anova_tukey_posthoc.csv`
- `results/tables/chi_square_results.csv`
- `results/tables/pointbiserial_results.csv`
- `results/tables/logistic_odds_ratios.csv`
- `results/tables/vif_results.csv`
- `results/tables/classification_report.csv`

### 8.3 시각화

- `results/figures/01_sleep_disorder_distribution.png`
- `results/figures/02_numeric_correlation_heatmap.png`
- `results/figures/03_anova_effect_sizes.png`
- `results/figures/04_group_boxplots.png`
- `results/figures/05_categorical_effect_sizes.png`
- `results/figures/06_categorical_panels.png`
- `results/figures/07_vif_plot.png`
- `results/figures/08_logistic_odds_ratios.png`
- `results/figures/09_model_performance.png`
