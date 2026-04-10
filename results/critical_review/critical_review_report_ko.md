# 수면장애 분석 결과에 대한 비판적 검토 보고서

## 1. 총평

이번 분석은 `reference`의 흐름을 따라 기술통계, 집단 비교, 상관분석, 공선성 진단, 로지스틱 회귀, Ridge 로지스틱 회귀, 다항 로지스틱 회귀, 민감도 분석까지 확장했다는 점에서 교육용 분석으로는 충분히 체계적이다.

하지만 통계 전문가 관점에서 보면, 현재 결과를 **의학적 근거**나 **실제 예측모형 성능**으로 강하게 해석하기에는 중요한 제약이 여럿 존재한다. 가장 큰 문제는 다음 네 가지다.

1. 데이터 자체가 `synthetic`이며 실제 임상자료가 아니다.
2. `Sleep Disorder`의 219개 결측을 모두 `None`으로 재코딩했다.
3. `person_id`를 제외하면 동일 프로파일 반복이 매우 많아 독립성 가정과 검증 성능이 심각하게 흔들린다.
4. ANOVA, Tukey, 카이제곱 등 일부 검정은 현재 데이터 구조에서 가정 위반 가능성이 크다.

따라서 현재 분석은 **설명 가능한 탐색적 데모 분석**으로는 타당하지만, **엄밀한 통계 추론**과 **실사용 예측모형 검증** 관점에서는 보완이 반드시 필요하다.

## 2. 가장 중요한 의문점과 비판 포인트

### 2.1 데이터가 합성 데이터라는 점

- `dataset/Sleep_health_and_lifestyle_dataset/about_data.rtf`의 설명문에는 이 데이터가 `synthetic and created by me for illustrative purposes`라고 명시되어 있다.
- 즉, 현재 분석은 실제 환자군을 반영한 의학적 추론이라기보다, **구조가 설계된 예시 데이터에 대한 통계 데모**에 가깝다.

의미:

- p-value, 오즈비, AUC가 좋아 보여도 그것이 실제 임상적 재현성을 뜻하지 않는다.
- 변수 관계가 데이터 생성 규칙에 의해 인위적으로 형성되었을 수 있다.

### 2.2 종속변수 결측을 전부 `None`으로 처리한 점

- 원본 CSV에서 `Sleep Disorder`의 실제 비결측 값은 `Sleep Apnea`, `Insomnia`뿐이며, `None` 문자열은 존재하지 않았다.
- 원본 기준 분포는 `NaN=219`, `Sleep Apnea=78`, `Insomnia=77`이다.
- 현재 스크립트는 이 결측을 모두 `None`으로 바꾼다.
  - [sleep_disorder_statistical_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/sleep_disorder_statistical_analysis.py#L148)
  - [ridge_logistic_feature_engineering_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/ridge_logistic_feature_engineering_analysis.py#L189)
  - [multinomial_sensitivity_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/multinomial_sensitivity_analysis.py#L98)

문제점:

- 데이터 설명문에는 `None` 범주가 존재한다고 되어 있지만, 실제 CSV에서는 `None`이 결측으로만 표현돼 있다.
- 결측이 곧 무질환이라는 해석은 **합리적일 수는 있어도 자동으로 보장되지는 않는다**.
- 이 재코딩은 결과 전체를 좌우하는 강한 가정이다.

권고:

- 본문에 “결측을 무질환으로 해석한 분석”임을 명시해야 한다.
- `Missing=Unknown`으로 두는 대안 분석, 결측 제외 분석, 현재 재코딩 분석의 3-way 민감도 분석이 필요하다.

### 2.3 중복 프로파일과 데이터 누수 가능성

- 전체 374행 중 `person_id`를 제외한 고유 프로파일은 132개뿐이다.
- 즉, 242행이 중복이며, 일부 프로파일은 최대 13회 반복된다.
- 랜덤 분할 기준으로 test 113행 중 90행이 train에도 동일 프로파일로 존재했고, test의 고유 프로파일 73개 중 54개가 train에서 이미 관측됐다.

현재 스크립트에서도 일반 랜덤 분할을 사용한다.

- [sleep_disorder_statistical_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/sleep_disorder_statistical_analysis.py#L344)
- [ridge_logistic_feature_engineering_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/ridge_logistic_feature_engineering_analysis.py#L300)
- [multinomial_sensitivity_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/multinomial_sensitivity_analysis.py#L226)

문제점:

- 이 구조에서는 test 성능이 “새로운 개인에 대한 일반화 성능”이 아니라, “이미 본 프로파일을 다시 맞추는 능력”에 가까워질 수 있다.
- 실제로 deduplicated 분석에서 성능이 크게 하락한 것은 이 문제를 강하게 시사한다.

권고:

- 단순 train/test split 대신 **profile-grouped split** 또는 **grouped cross-validation**을 써야 한다.
- 현재 성능지표는 본문에서 “낙관적 추정치”로 명확히 낮춰 표현하는 것이 맞다.

### 2.4 고전적 검정의 가정 위반 가능성

현재 수치형 비교는 ANOVA와 Tukey를 사용했다.

- [sleep_disorder_statistical_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/sleep_disorder_statistical_analysis.py#L192)
- [multinomial_sensitivity_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/multinomial_sensitivity_analysis.py#L159)

범주형 비교는 카이제곱 검정을 사용했다.

- [sleep_disorder_statistical_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/sleep_disorder_statistical_analysis.py#L222)

이번 재검토에서 확인된 점:

- Shapiro-Wilk는 점검한 모든 수치형 변수에서 정규성 기각.
- Levene 검정은 9개 중 8개 변수에서 등분산성 기각.
- `Occupation x Sleep Disorder` 분할표는 기대도수 5 미만 셀이 33개 중 12개였고 최소 기대도수는 0.206이었다.

문제점:

- 일반 ANOVA + Tukey는 현재 데이터에서는 지나치게 낙관적일 수 있다.
- `Occupation`의 카이제곱 p-value 역시 희소 셀 때문에 신뢰도가 낮다.

권고:

- 수치형은 Welch ANOVA 또는 Kruskal-Wallis로 대체하는 것이 더 적절하다.
- 사후비교는 Games-Howell 또는 Dunn test를 고려해야 한다.
- 범주형은 희소 범주 통합 후 카이제곱, 또는 exact / Monte Carlo 기반 검정을 고려해야 한다.

## 3. 중간 수준의 비판 포인트

### 3.1 다중검정 보정이 없다

- ANOVA 9개, point-biserial 9개, 카이제곱 3개, 다수의 post-hoc 비교를 수행했지만 다중검정 보정이 없다.
- 현재 p-value는 거짓양성 위험을 과소평가할 수 있다.

권고:

- 최소한 Benjamini-Hochberg FDR 또는 Holm 보정을 같이 제시해야 한다.

### 3.2 파생변수 일부는 통계적으로는 편리하지만 구성타당도가 약하다

- `map_bp`, `pulse_pressure`는 임상 해석성이 높아 비교적 타당하다.
- 하지만 아래 파생변수는 더 조심해야 한다.
  - `sleep_deficit_7h = max(0, 7 - sleep_duration)`
  - `sleep_stress_balance = quality_of_sleep - stress_level`
  - `bmi_risk = bmi_category != Normal`

문제점:

- `7시간` 기준은 합리적이지만 데이터 기반으로 검증된 절단점이 아니다.
- `quality_of_sleep`와 `stress_level`을 동일 간격 척도로 보고 단순 차를 취한 것은 심리측정학적으로 강한 가정이다.
- BMI를 `Normal vs Non-Normal`로 단순화하면 `Overweight`와 `Obese`의 차이를 잃는다.

권고:

- 파생변수는 “해석 가능한 요약지표”로 제시하되, 검증된 임상 척도처럼 표현하면 안 된다.
- spline, restricted cubic spline, ordinal coding, target-free PCA/factor analysis로 대체 후보를 검토할 수 있다.

### 3.3 feature selection과 성능평가가 완전히 분리되어 있지 않다

- 파생변수 스크리닝은 전체 데이터를 이용해 수행됐다.
  - [ridge_logistic_feature_engineering_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/ridge_logistic_feature_engineering_analysis.py#L247)
- 그 후 같은 데이터에서 모델 비교를 한다.
  - [ridge_logistic_feature_engineering_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/ridge_logistic_feature_engineering_analysis.py#L300)

문제점:

- 이 절차는 작은 데이터에서 선택 편향을 만들 수 있다.
- 특히 파생변수의 효용이 다소 과장될 가능성이 있다.

권고:

- nested cross-validation 안에서 파생변수 선택 또는 후보 비교를 수행해야 한다.
- 최소한 “탐색 후 검증이 완전히 분리되지 않았다”는 한계가 보고서에 들어가야 한다.

### 3.4 `class_weight="balanced"`와 0.5 threshold 사용의 타당성

- 현재 로지스틱/다항 로지스틱 분류기는 모두 `class_weight="balanced"`를 사용한다.
  - [sleep_disorder_statistical_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/sleep_disorder_statistical_analysis.py#L364)
  - [ridge_logistic_feature_engineering_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/ridge_logistic_feature_engineering_analysis.py#L277)
  - [multinomial_sensitivity_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/multinomial_sensitivity_analysis.py#L210)

문제점:

- 현재 클래스 불균형은 극단적이지 않다.
- `balanced`는 분류 기준을 바꾸는 대신 확률 calibration을 손상시킬 수 있다.
- 그럼에도 calibration curve, Brier score, decision threshold 최적화가 수행되지 않았다.

권고:

- weighted / unweighted 모델을 함께 비교해야 한다.
- calibration curve, Brier score, net benefit를 추가하는 것이 좋다.

## 4. 통계 전문가 수준에서 특히 부족한 부분

아래 항목은 “기초 분석”을 넘어서 “전문가 수준의 엄밀성”을 요구할 때 특히 아쉬운 지점이다.

1. **데이터 생성 과정과 측정 수준에 대한 비판적 검토가 보고서 본문에서 충분히 전면화되지 않았다.**
2. **추론과 예측을 분리해 보고하지 않았다.**
   - 오즈비 해석과 분류 성능 보고는 목적이 다르므로 별도 기준이 필요하다.
3. **가정 진단 결과에 따라 검정을 교체하지 않았다.**
4. **중복 구조를 반영한 grouped validation, cluster-robust inference가 없다.**
5. **다중검정, calibration, 불확실성 구간(bootstrap CI), 외적 타당도 논의가 없다.**
6. **합성 데이터의 한계 때문에 인과적·임상적 문장 표현을 더 절제했어야 한다.**

즉, 분석 흐름은 좋았지만, 전문가 수준으로 끌어올리려면 **검정 선택의 보수성**, **가정 위반 처리**, **불확실성 정량화**, **일반화 성능 검증 설계**가 더 필요하다.

## 5. 개선 우선순위

### 1순위

- 결과 해석 수준을 “탐색적 분석 / synthetic data demonstration”로 낮춰 표현
- `Sleep Disorder` 결측 처리 민감도 분석 수행
- grouped CV 또는 unique-profile split으로 예측성능 재평가

### 2순위

- Welch ANOVA / Kruskal-Wallis / Games-Howell / exact or Monte Carlo chi-square로 재분석
- FDR 또는 Holm 보정 적용
- weighted vs unweighted logistic 비교

### 3순위

- calibration curve, Brier score, bootstrap CI 추가
- nested CV 기반 feature engineering 검증
- profile cluster 또는 latent factor 기반 구조 분석 추가

## 6. reference 외에 추가로 수행해볼 수 있는 통계 방법

다음 방법들은 현재 `reference` 흐름을 넘어서 더 엄밀한 평가에 도움된다.

1. **Welch ANOVA / Games-Howell**
   - 등분산 가정이 무너진 현재 데이터에 더 적합하다.
2. **Kruskal-Wallis / Dunn test**
   - 반복값이 많고 비정규적인 현재 자료 구조에 강건하다.
3. **FDR 보정**
   - 다중검정으로 인한 과잉 해석을 줄인다.
4. **Bootstrap confidence intervals**
   - 오즈비, 상관계수, AUC에 대한 불확실성을 수치로 보여준다.
5. **Grouped cross-validation**
   - 동일 프로파일 누수를 막고 일반화 성능을 더 정직하게 평가한다.
6. **Calibration analysis**
   - ROC-AUC 외에 예측확률 자체가 믿을 만한지 검증한다.
7. **Decision curve analysis**
   - 실제 스크리닝 도구로 쓸 가치가 있는지 평가한다.
8. **Generalized additive model (GAM)**
   - 나이, 혈압, 수면시간의 비선형 효과를 포착한다.
9. **Ordinal / partial proportional odds 모델**
   - 무질환-불면증-무호흡을 단순 명목형 대신 위험 연속선상으로 볼 수 있는지 점검한다.
10. **Latent class analysis / factor analysis**
   - 혈압축, 수면-스트레스축, 활동축 같은 잠재 구조를 분리할 수 있다.
11. **Permutation test**
   - 합성 데이터나 작은 샘플에서 분포가정 의존도를 줄인다.
12. **Cluster-robust or profile-level inference**
   - 반복 프로파일 구조를 반영한 더 보수적인 표준오차 추정이 가능하다.

## 7. 최종 결론

현재 분석은 교육용·탐색용으로는 충분히 설득력 있고, 특히 공선성 진단과 민감도 분석을 추가한 점은 분명 장점이다.

그러나 엄밀한 통계 비판 관점에서는 다음 문장으로 요약하는 것이 가장 정확하다.

> 이 분석은 synthetic 데이터에 대한 탐색적 예시 분석으로는 유용하지만, 실제 의료적 의사결정이나 일반화 가능한 예측모형 성능의 근거로 사용하기에는 데이터 구조와 검정 가정 측면에서 중요한 제약이 남아 있다.

따라서 앞으로의 개선 방향은 “모델을 더 복잡하게 만드는 것”보다 먼저,

1. 데이터 해석 가정 명시
2. 검정 가정 위반에 맞는 방법 교체
3. 누수 없는 검증 설계
4. 불확실성 정량화

를 우선하는 것이 맞다.
