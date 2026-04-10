# 현재 데이터를 최대한 잘 분석하기 위한 방법론 개선 계획

## 1. 전제

이 문서는 다음 전제를 둔다.

- 현재 사용 중인 `Sleep_health_and_lifestyle_dataset`는 분석 가능한 신뢰할 만한 데이터다.
- 따라서 여기서의 비판은 **데이터 자체의 진위나 외적 타당도**가 아니라, **현재 분석 설계가 이 데이터를 얼마나 엄밀하게 활용했는가**에 초점을 둔다.

즉, 질문은 이것이다.

> “지금 가진 데이터를 믿는다고 할 때, 현재 분석은 어디까지 잘 되어 있고, 어디서부터는 아직 통계적으로 더 다듬어야 하는가?”

## 2. 현재 분석의 장점

현재 분석은 이미 아래 강점을 갖고 있다.

1. 기술통계 -> 집단 간 차이 -> 상관/공선성 -> 회귀모형으로 흐름이 자연스럽다.
2. 단순 이진 로지스틱뿐 아니라 Ridge 로지스틱, 다항 로지스틱, 민감도 분석까지 확장했다.
3. 공선성을 실제로 확인하고 파생변수를 설계했다.
4. 시각화와 표를 함께 남겨 재현성과 설명력을 확보했다.

즉, **분석의 뼈대는 좋다.**

문제는 그 다음 단계다. 지금부터 필요한 것은 “새 모델을 더 많이 추가하는 것”이 아니라, **현재 결론이 얼마나 견고한지 검증하는 절차**다.

## 3. 데이터 신뢰성 문제를 제외해도 남는 핵심 부족점

### 3.1 검정 선택이 데이터 구조에 완전히 맞춰져 있지 않다

현재 수치형 집단 비교는 ANOVA와 Tukey 중심이다.

- [sleep_disorder_statistical_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/sleep_disorder_statistical_analysis.py#L192)
- [multinomial_sensitivity_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/multinomial_sensitivity_analysis.py#L159)

범주형 비교는 카이제곱 검정을 사용한다.

- [sleep_disorder_statistical_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/sleep_disorder_statistical_analysis.py#L222)

부족한 이유:

- 현재 변수들은 반복값이 많고, 분포가 이산적이며, 분산도 집단별로 다를 가능성이 높다.
- 이 경우 일반 ANOVA/Tukey는 결과가 너무 낙관적일 수 있다.
- 특히 `Occupation`처럼 범주 수가 많은 변수는 희소 셀 문제가 생길 수 있다.

정리하면:

- 현재 분석은 “차이가 있는가”는 보여주지만,
- “그 차이가 검정 가정이 바뀌어도 유지되는가”는 아직 충분히 보여주지 못했다.

### 3.2 다중검정 보정이 없다

현재 분석에서는 다음이 반복된다.

- 여러 수치형 변수에 대한 ANOVA
- 여러 변수에 대한 point-biserial correlation
- 여러 범주형 변수에 대한 카이제곱 검정
- post-hoc pairwise 비교

하지만 FDR/Holm 같은 보정이 없다.

부족한 이유:

- 개별 p-value는 유의해 보여도, 여러 검정을 동시에 수행하면 거짓양성이 자연스럽게 늘어난다.
- 따라서 현재 유의하다고 제시된 변수 중 일부는 보정 후에는 약해질 수 있다.

### 3.3 예측모형 검증이 아직 “가장 엄밀한 방식”은 아니다

현재 분류 평가는 일반적인 stratified split과 cross-validation을 사용한다.

- [sleep_disorder_statistical_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/sleep_disorder_statistical_analysis.py#L344)
- [ridge_logistic_feature_engineering_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/ridge_logistic_feature_engineering_analysis.py#L300)
- [multinomial_sensitivity_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/multinomial_sensitivity_analysis.py#L226)

부족한 이유:

- 현재 데이터는 동일하거나 매우 유사한 프로파일이 여러 번 반복되는 구조를 갖고 있다.
- 이런 데이터에서는 일반 랜덤 분할이 모델 성능을 실제보다 좋게 보이게 만들 수 있다.
- 이미 deduplicated 민감도 분석에서 성능이 의미 있게 떨어졌기 때문에, 이 문제는 실제로 영향을 준다고 봐야 한다.

즉:

- 현재 모델은 “패턴을 포착했다”는 근거는 있다.
- 하지만 “새로운 샘플에도 이 정도로 잘 맞는다”는 표현은 아직 조심해야 한다.

### 3.4 로지스틱 회귀의 핵심 가정 점검이 덜 되어 있다

현재는 공선성은 충분히 보았지만, 로지스틱 회귀에서 더 중요한 아래 항목은 아직 직접 점검하지 않았다.

1. 연속형 변수의 logit 선형성
2. 영향점과 고레버리지 샘플
3. calibration
4. 분류 임계값의 타당성
5. 다항 로지스틱의 IIA 가정

부족한 이유:

- 오즈비를 해석하려면 선형성 가정이 어느 정도 맞는지 봐야 한다.
- ROC-AUC가 높아도 확률이 실제 위험을 잘 반영하는지는 calibration을 봐야 알 수 있다.
- 0.5 threshold를 그대로 쓰는 것이 실제 목적에 맞는지도 검토가 필요하다.

### 3.5 feature engineering은 합리적이지만 검증은 더 필요하다

현재 파생변수는 방향이 좋다.

- `Mean Arterial Pressure`
- `Pulse Pressure`
- `Sleep Deficit`
- `Sleep-Stress Balance`

하지만 현재는 전체 데이터를 보고 후보를 설계하고, 그 성능을 같은 데이터 내에서 비교했다.

- [ridge_logistic_feature_engineering_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/ridge_logistic_feature_engineering_analysis.py#L247)

부족한 이유:

- 이는 탐색적 feature engineering으로는 충분히 괜찮지만,
- “이 파생변수가 정말 일반적으로 더 낫다”는 결론을 내리기에는 선택 편향이 남는다.

즉:

- 현재 파생변수는 “유망하다”까지는 말할 수 있다.
- “검증되었다”고 말하려면 nested CV나 bootstrap stability가 더 필요하다.

### 3.6 추론과 예측이 아직 완전히 분리되어 있지 않다

현재 보고서에서는 다음 두 목적이 함께 다뤄진다.

1. 어떤 변수가 수면장애와 관련이 있는가
2. 어떤 모델이 분류를 잘 하는가

이 두 질문은 다르다.

부족한 이유:

- 추론은 오즈비, 신뢰구간, p-value, 효과크기, 가정 진단이 중요하다.
- 예측은 CV 설계, calibration, threshold, net benefit가 중요하다.
- 둘을 함께 보면 설명은 쉬워지지만, 엄밀성은 조금 흐려질 수 있다.

## 4. 개선 실험계획

아래 실험계획은 “현재 데이터를 최대한 잘 분석한다”는 목적에 맞춰 우선순위를 둔 것이다.

## 4.1 1단계: 단변량 결과의 강건성 재검증

### 실험 내용

1. 수치형 변수에 대해 `ANOVA`와 함께 `Welch ANOVA`를 병행
2. 비정규/반복값이 많은 변수에 대해 `Kruskal-Wallis` 추가
3. 사후비교는 `Games-Howell` 또는 `Dunn test`로 대체/병행
4. 범주형 변수는 희소 범주를 통합한 뒤 카이제곱 재수행
5. 모든 단변량 검정에 `FDR` 또는 `Holm` 보정 적용

### 왜 필요한가

- 현재 결론이 특정 검정 선택에만 의존하는지 확인하기 위해서다.
- 정말 중요한 변수는 검정 방식을 조금 바꿔도 계속 살아남아야 한다.

### 기대 산출물

- “기존 결과와 강건하게 일치하는 변수”
- “검정 방식에 따라 유의성이 흔들리는 변수”
- “보정 후에도 남는 핵심 변수”

### 해석 가치

이 단계가 끝나면, 이후 회귀모형에 넣을 후보 변수의 신뢰도가 훨씬 높아진다.

## 4.2 2단계: 로지스틱 회귀 가정 진단 실험

### 실험 내용

1. 연속형 변수에 대해 `Box-Tidwell` 또는 spline 기반 logit 선형성 점검
2. binary logistic에서 영향점, high leverage, Cook’s distance 확인
3. 다항 로지스틱에서 IIA 민감도 점검
4. separation 또는 quasi-separation 여부 확인

### 왜 필요한가

- 로지스틱 회귀는 단순히 돌아가기만 하면 끝이 아니다.
- 오즈비를 해석하려면 “모델 가정이 얼마나 맞는가”를 봐야 한다.
- 가정이 틀리면 유의한 변수처럼 보여도 해석이 왜곡될 수 있다.

### 기대 산출물

- 선형항으로 충분한 변수
- spline이나 구간화가 필요한 변수
- 특정 샘플에 과도하게 좌우되는 계수 여부

### 해석 가치

이 단계는 “어떤 변수가 중요하다”를 넘어서 “왜 그렇게 보이는가”를 더 정확히 설명하게 해준다.

## 4.3 3단계: 검증 설계 재구성

### 실험 내용

1. 기존 stratified CV 유지
2. 추가로 `unique profile 기반 grouped CV` 수행
3. `deduplicated data` 성능과 grouped CV 성능 비교
4. bootstrap으로 AUC, F1, OR 신뢰구간 계산

### 왜 필요한가

- 지금 성능이 좋은 것이 진짜 구조를 학습한 결과인지, 중복 패턴에 의존한 결과인지 더 엄밀히 가려내기 위해서다.
- grouped CV는 현재 데이터 구조에서 가장 중요한 검증 실험 중 하나다.

### 기대 산출물

- 낙관적 추정치와 보수적 추정치의 차이
- 모델 성능의 신뢰구간
- “실사용 기대 성능”에 가까운 범위

### 해석 가치

이 단계가 있어야 AUC나 F1을 더 자신 있게 말할 수 있다.

## 4.4 4단계: feature engineering의 엄밀한 검증

### 실험 내용

1. 현재 파생변수 세트 유지
2. 원 변수 세트, 압축 파생변수 세트, 상호작용 포함 세트 비교
3. nested CV 안에서 feature set 비교
4. bootstrap stability로 파생변수 선택 일관성 확인

### 왜 필요한가

- 지금은 파생변수가 좋아 보이지만, 그 우월성이 얼마나 안정적인지는 아직 불명확하다.
- 단순히 이번 데이터에서만 좋아 보이는지, 아니면 분할이 바뀌어도 계속 좋은지 봐야 한다.

### 기대 산출물

- 가장 안정적인 변수 세트
- 성능 대비 해석가능성이 가장 좋은 모델
- 분할이 바뀌어도 자주 선택되는 핵심 변수

### 해석 가치

이 단계가 끝나면 “왜 Compressed Derived Ridge를 추천하는가”를 훨씬 강하게 말할 수 있다.

## 4.5 5단계: 예측모형 평가를 ROC-AUC 중심에서 확률 품질 중심으로 확장

### 실험 내용

1. calibration curve
2. Brier score
3. calibration intercept / slope
4. threshold sweep
5. 필요시 decision curve analysis

### 왜 필요한가

- 현재는 ROC-AUC와 accuracy 중심이어서 “잘 구분하느냐”는 알 수 있다.
- 하지만 스크리닝 모델이라면 “예측확률이 실제 위험을 얼마나 잘 반영하느냐”가 더 중요할 수 있다.

### 기대 산출물

- 확률 예측의 신뢰성
- 목적에 맞는 임계값
- false positive와 false negative 균형에 대한 근거

### 해석 가치

이 단계는 모델을 실제 의사결정 보조 형태로 연결하는 데 필요하다.

## 4.6 6단계: 구조 확장 실험

### 실험 내용

1. `Age`, `MAP`, `Sleep Duration`에 spline 적용
2. 사전 지정 상호작용 실험
   - `Age x MAP`
   - `Sleep Duration x Stress Level`
   - `BMI x Blood Pressure`
3. multinomial Ridge / Elastic Net 비교

### 왜 필요한가

- 현재 모델은 대부분 선형 주효과에 의존한다.
- 하지만 수면장애 위험은 특정 구간부터 급격히 증가하거나, 변수 조합에서만 강해질 수 있다.

### 기대 산출물

- 비선형 임계 구간
- 특정 조건에서만 위험이 커지는 상호작용
- subtype 분류에서 더 안정적인 정규화 모델

### 해석 가치

이 단계는 설명력과 실무 활용성을 동시에 높인다.

## 5. 우선순위 제안

모든 실험을 한 번에 하는 것보다 아래 순서가 가장 효율적이다.

### 우선순위 A

1. `Welch/Kruskal + FDR 보정`
2. `grouped CV + bootstrap CI`
3. `calibration 분석`

이 세 가지는 지금 결과를 가장 빠르게 “더 신뢰 가능한 결과”로 바꿔준다.

### 우선순위 B

4. `logit 선형성 + 영향점 진단`
5. `nested CV 기반 파생변수 검증`

이 단계는 현재 모델 해석과 추천 변수를 더 정교하게 만든다.

### 우선순위 C

6. `spline + interaction + multinomial regularization`

이 단계는 고급 확장 단계다.

## 6. 최종 정리

현재 데이터를 믿는다는 전제하에서는, 분석의 가장 큰 부족함은 **데이터가 나빠서**가 아니라 **검증과 강건성 확인이 아직 충분하지 않아서**다.

정확히 말하면 지금 필요한 것은 다음이다.

1. 단변량 결과가 검정 선택이 바뀌어도 유지되는지 확인
2. 회귀모형 가정이 맞는지 확인
3. 성능이 누수 없이도 유지되는지 확인
4. 예측확률이 실제로도 해석 가능한지 확인
5. 파생변수 추천이 안정적인지 확인

즉, 다음 단계의 핵심은 “새로운 방법을 더 많이 쓰는 것”이 아니라,

> **현재 결론이 얼마나 흔들리지 않는지 검증하는 것**

이다.

이 기준으로 보면 다음 실행 순서가 가장 적절하다.

1. 강건한 단변량 재검정
2. grouped CV와 bootstrap
3. calibration
4. 로지스틱 가정 진단
5. nested CV 기반 파생변수 검증

이 순서가 통계적으로도, 보고서 설득력 측면에서도 가장 타당하다.
