# 수면장애 통합 메인 보고서

## 1. 이 보고서가 무엇을 통합했는가

이 문서는 지금까지 수행한 분석을 하나의 메인 보고서로 다시 정리한 것이다.

통합한 분석 묶음은 아래 7단계다.

여기서 `7단계`는 보고서 구조상 큰 묶음이고, 실제 실험 실행 순서는 3장에서 더 세분한 `8개 단계`로 다시 설명한다.

1. 기초 통계분석과 이진 로지스틱 회귀
2. Ridge 로지스틱 회귀와 파생변수 탐색
3. 다항 로지스틱 회귀와 민감도 분석
4. 후속 엄밀성 검증
   - 강건한 단변량 재검정
   - grouped CV
   - bootstrap
   - calibration
   - 로지스틱 가정 진단
   - feature set 최종 비교
5. 비판 포인트 해소를 위한 추가 검증
   - repeated grouped model comparison
   - tuning objective 비교
   - profile-cluster GEE sensitivity analysis
   - repeated grouped multinomial validation
6. grouped bootstrap 모델 비교와 변수 안정성 후속 검증
   - grouped bootstrap OOB model comparison
   - pairwise superiority analysis
   - stability selection
   - final-model coefficient bootstrap
7. 최종 엄밀성 보강
   - exact-row dedup sensitivity
   - nested grouped bootstrap 재비교

즉, 이 보고서는 “처음 얻은 결과”가 아니라, **후속 검증까지 반영한 최신 결론**을 메인 결과로 제시한다.

## 2. 어떤 데이터로 분석했는가

- 본 분석 데이터는 [Sleep_health_and_lifestyle_dataset.csv](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/dataset/Sleep_health_and_lifestyle_dataset/Sleep_health_and_lifestyle_dataset.csv) 이다.
- [FitBit Fitness Tracker Data](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/dataset/FitBit%20Fitness%20Tracker%20Data) 도 함께 확인했지만, 이 데이터는 `Sleep Disorder` 라벨이 없고 현재 데이터와 개인 단위 연결도 되지 않아 분류모형의 직접 학습에는 쓰지 않았다.

최신 검증 기준 데이터 구조는 다음처럼 이해하는 것이 맞다.

- 전체 행 수: `374`
- predictor profile 기준 고유 패턴 수: `109`
- 반복 행 수: `265`
- 가장 많이 반복된 패턴의 반복 횟수: `13`

주의할 점은, 여기의 `109`는 **predictor profile 기준 고유 패턴 수**이고, 후속 exact-row dedup sensitivity에서 나오는 `132`는 **person_id만 제거한 뒤 정확히 같은 row를 하나로 접은 row 수**라는 점이다. 즉 같은 predictor profile 안에서도 라벨이 다른 행은 dedup 후에도 별도 row로 남을 수 있으므로, 두 숫자는 서로 다른 개념이라 함께 존재해도 모순이 아니다.

이 수치는 왜 중요하냐면, 같은 profile이 여러 번 반복되는 구조에서는 일반 랜덤 분할 성능이 쉽게 낙관적으로 보일 수 있기 때문이다. 그래서 최신 결론은 모두 `grouped validation`까지 통과한 결과를 기준으로 정리했다.

## 3. 왜 이런 순서로 실험했는가

최신 분석 흐름은 다음 순서로 설계했다.

1. **강건한 단변량 재검정**
   - 이유: 어떤 변수가 정말 중요한지 먼저 확인해야 모델 설계가 흔들리지 않는다.
   - 단순 ANOVA가 아니라 `Welch ANOVA`, `Kruskal-Wallis`, `FDR 보정`, permutation chi-square까지 함께 썼다.
2. **grouped CV와 bootstrap**
   - 이유: 성능이 실제로 안정적인지, 검증 설계가 엄격해져도 유지되는지 확인해야 한다.
3. **calibration과 threshold 안정성**
   - 이유: AUC만 높다고 좋은 모델이 아니라, 예측확률이 믿을 만하고 0.5 기준에서 바로 쓸 수 있어야 실제 활용성이 있다.
4. **로지스틱 가정 진단**
   - 이유: 최종적으로 추천할 모델이 오즈비 해석까지 가능한지 확인해야 한다.
5. **feature set 최종 비교**
   - 이유: 원 변수 모델, 압축 파생변수 모델, 상호작용 확장 모델 중 무엇을 메인 모델로 추천할지 결정해야 한다.
6. **추가 비판 해소 실험**
   - 이유: tuning objective, threshold 안정성, clustered inference, subtype 압축 문제를 마지막으로 다시 확인해야 한다.
7. **grouped bootstrap과 변수 안정성 검증**
   - 이유: 상위 모델 간 차이가 실제로 큰지, 그리고 최종 변수들이 표본 재추출에도 안정적으로 남는지를 확인해야 한다.
8. **exact-row dedup sensitivity와 nested grouped bootstrap**
   - 이유: 단변량 검정과 예측 검증의 독립성 기준을 더 잘 맞추고, fixed-hyperparameter bootstrap이라는 마지막 비판 포인트까지 해소해야 한다.

즉, 이번 메인 결론은 `유의성 -> 성능 -> 확률 품질 -> 해석 가능성`을 순서대로 확인한 뒤 도출한 결과다.

## 4. 수면장애와 어떤 feature가 강건하게 관련되었는가

### 4.1 수치형 변수

검정 방식을 바꾸고 exact-row dedup sensitivity를 거쳐도 계속 살아남은 핵심 변수는 아래와 같다.

1. `Diastolic BP`
2. `Systolic BP`
3. `Mean Arterial Pressure`
4. `Age`
5. `Physical Activity Level`
6. `Sleep Duration`
7. `Quality of Sleep`
8. `Heart Rate`
9. `Pulse Pressure`
10. `Sleep Deficit`

핵심 해석:

- 가장 강한 축은 여전히 `혈압`이다.
- 그 다음은 `나이`, `수면시간`, `활동량`, `심박수`, `수면의 질`이다.
- 즉, 수면장애는 단순히 “잠을 적게 자느냐”만이 아니라, 심혈관 축과 생활습관 축이 함께 움직이는 구조로 이해하는 것이 맞다.

### 4.2 범주형 변수

강건성 검정 기준으로 안정적으로 남은 범주형 변수는 아래와 같다.

1. `Gender`
2. `BMI Category`
3. `BMI Category (Collapsed)`

`Occupation`도 매우 강한 연관성을 보였지만, 희소 셀이 남아 있어 **설명용 참고 변수**로는 유용하나 최종 회귀모형 입력 변수로는 보수적으로 다루는 것이 적절했다.

## 5. 최종적으로 어떤 모델을 써야 하는가

최신 검증에서는 아래 세 부류의 모델을 비교했고, 추가로 repeated grouped `roc_auc`/`neg_log_loss` 기준 비교까지 수행했다.

1. 원 변수 대표 모델
2. 압축 파생변수 모델
3. 압축 파생변수 + 상호작용 모델

중요한 점은, **무엇을 최종 추천 모델로 볼지는 평가 목적에 따라 달라진다**는 것이다.

### 5.1 예측 우선 최상위 모델군

최신 결론은 “단일 절대 승자 모델”보다 아래의 `quality-based family`가 최상위권이라는 것이다.

1. `Original: Quality + PA`
2. `Original: Quality + HR`

이 두 모델은 repeated grouped, grouped bootstrap, nested grouped bootstrap을 함께 봤을 때 사실상 같은 최상위 prediction-first 모델군으로 남았다.

#### `Original: Quality + PA`

사용 변수:

1. `Age`
2. `Quality of Sleep`
3. `Physical Activity Level`
4. `Diastolic BP`
5. `male`
6. `bmi_risk`

이 모델을 prediction-first 기본 추천으로 두는 이유는 다음과 같다.

- repeated grouped `neg_log_loss` 기준 winner share가 가장 높았다 (`0.300`).
- 같은 기준에서 `Brier = 0.075`, `ECE = 0.085`로 확률 품질이 가장 좋았다.
- `ROC-AUC = 0.936`, `F1 = 0.895` 수준으로 분리력도 충분했다.
- 추가 profile-cluster GEE sensitivity analysis에서 `Quality of Sleep`과 `Diastolic BP`가 모두 유의한 신호로 유지됐다.
- nested grouped bootstrap에서도 `Brier = 0.076`으로 가장 좋은 평균 확률 품질을 유지했다.

즉, **예측확률의 품질을 중시할 때 `Quality + PA`는 가장 자연스러운 operational default지만, 이는 tied top family 안에서의 실용적 선택으로 읽는 편이 더 정확하다.**

### 5.2 근접한 대안 모델

#### `Original: Quality + HR`

- repeated grouped `neg_log_loss` 기준 `ROC-AUC = 0.938`, `F1 = 0.898`로 가장 높았다.
- 다만 `Brier`와 `winner share`에서는 `Quality + PA`와 거의 차이가 없었다.
- profile-cluster GEE sensitivity analysis에서는 `Quality of Sleep`과 `Diastolic BP`가 유의했고, `Heart Rate`는 독립 효과가 강하게 남지 않았다.
- grouped bootstrap OOB 평균에서는 `Quality + PA`와 사실상 동급이었고, 오히려 아주 근소하게 앞서는 평균값을 보였다.
- nested grouped bootstrap에서도 `F1 = 0.889`, winner share `0.400`으로 `Quality + PA`와 사실상 같은 최상위권에 남았다.

#### `Original: Sleep + PA`

- repeated grouped `neg_log_loss` 기준 `ROC-AUC = 0.937`, `F1 = 0.892`, `Brier = 0.081`이었다.
- `Quality of Sleep` 같은 self-reported 변수를 덜 쓰고, 더 보수적인 생활습관형 baseline을 원할 때는 여전히 매우 실용적이다.
- 즉, **예측 성능은 약간 양보하더라도 측정 보수성과 설명 단순성을 선호할 때의 기본형**으로 적합하다.

### 5.3 왜 이전 결론이 바뀌었는가

이전 메인 보고서에서는 `Original: Sleep + PA`가 가장 설득력 있는 기본형처럼 보였고, 그다음에는 `Quality + PA`가 prediction-first 기본형으로 정리됐다.

하지만 bootstrap 후속 검증까지 포함하면 결론이 더 정교해진다.

1. `roc_auc`를 기준으로 튜닝하면 일부 모델은 threshold 0.5에서 F1과 Brier가 크게 흔들린다.
2. `neg_log_loss`처럼 확률 품질에 더 가까운 목적함수로 튜닝하면, quality 기반 원 변수 모델이 근소하게 앞선다.
3. grouped bootstrap OOB 기준으로는 `Quality + PA`와 `Quality + HR`의 차이가 매우 작아, 단일 승자보다 `quality-based family`라는 표현이 더 정직하다.
4. exact-row dedup sensitivity와 nested grouped bootstrap까지 추가해도 이 결론은 유지됐다.

즉, **모델 추천은 데이터만이 아니라 무엇을 최적화하느냐에 따라 달라진다.**

### 5.4 파생변수 모델은 어떻게 봐야 하는가

- `Compressed Derived`는 `neg_log_loss` 기준에서는 꽤 경쟁적이었지만, 원 변수 모델을 명확히 이기지는 못했다.
- `Compressed + Interaction`도 충분히 좋은 성능을 보였지만, 복잡도 증가 대비 이득은 제한적이었다.

따라서 현재 단계에서는 **파생변수 모델을 1순위 기본형으로 올리기보다, 보조 대안으로 두는 것이 더 타당**하다.

## 6. 최종 모델에서 어떤 변수가 실제로 안정적으로 남았는가

이번 프로젝트 전체를 변수군 수준에서 보면 가장 안정적으로 남은 축은 `혈압축 + BMI Risk + Quality of Sleep`이었다.

근거는 네 층위에서 반복 확인됐다.

1. 강건한 단변량 검정에서 혈압 관련 변수들이 가장 강한 축으로 남았다.
2. profile-cluster GEE sensitivity analysis에서 `Diastolic BP`와 `Quality of Sleep`가 모두 안정적으로 유지됐다.
3. stability selection에서 `BMI Risk 0.979`, `Quality of Sleep 0.843`, `Systolic BP 0.757`, `Diastolic BP 0.736`으로 높은 selection frequency를 보였다.
4. 최종 `Quality + PA` 고정 모델 coefficient bootstrap에서 `Diastolic BP`, `BMI Risk`, `Quality of Sleep`, `Age`의 부호 안정성이 모두 `1.0`이었다.
5. exact-row dedup 단변량 재검정에서도 `Age`, `Quality of Sleep`, `Physical Activity Level`, `Sleep Duration`, `Diastolic BP`를 포함한 핵심 축이 그대로 유지됐다.

다만 “단일 변수 하나”로 압축하면, 최종 고정 모델 안에서 가장 안정적으로 남은 핵심 변수는 여전히 `Diastolic BP`였다.

- `Original: Quality + HR` profile-cluster GEE sensitivity analysis: `Quality of Sleep OR = 0.344`, 95% CI = `0.167–0.711`
- `Original: Quality + PA` profile-cluster GEE sensitivity analysis: `Quality of Sleep OR = 0.365`, 95% CI = `0.172–0.773`
- `Quality + PA` coefficient bootstrap: `Diastolic BP` standardized coef median = `0.941`, `OR per 1 SD = 2.563`
- `Quality + PA` coefficient bootstrap: `Quality of Sleep` standardized coef median = `-0.648`, `OR per 1 SD = 0.523`

반면 아래 변수들은 모델에 기여하더라도 독립 효과는 더 조심해서 해석하는 편이 맞았다.

- `Heart Rate`
- `Physical Activity Level`
- `Sleep Duration`
- `male`

즉, 최신 추가 검증까지 반영하면 가장 단단하게 남는 메시지는 아래 세 문장이다.

> 변수군 수준에서는 `혈압축 + BMI Risk + Quality of Sleep`가 가장 안정적으로 남는다.

> 최종 `Quality + PA` 고정 모델 안에서는 `Diastolic BP`가 가장 안정적인 양(+) 방향 신호다.

> quality 기반 screening 모델에서는 `Quality of Sleep`이 반복적으로 보호 방향 신호를 보인다.

## 7. 최신 결과를 기반으로 얻는 인사이트

### 7.1 통계적으로 무엇이 달라졌는가

초기 분석에서는 하나의 “최종 모델”을 찾는 쪽으로 해석이 흘렀다.

하지만 최신 추가 검증까지 포함하면 결론이 더 정교해진다.

1. 혈압과 수면 관련 축이 중요하다는 큰 방향은 유지된다.
2. 다만 “최종 추천 모델”은 단순 AUC 최대화가 아니라, `tuning objective`, `확률 품질`, `profile-cluster sensitivity inference`까지 함께 봐야 한다.
3. 그 기준을 모두 적용하면, prediction-first 추천은 `Quality + PA / Quality + HR`의 quality-based family, 보수적 baseline은 `Sleep + PA`처럼 **두 층위의 권고**로 정리하는 것이 더 정확하다.

### 7.2 실무적으로 어떻게 활용할 수 있는가

이 결과는 아래처럼 활용하는 것이 가장 자연스럽다.

1. **1차 스크리닝 모델**
   - self-reported quality score를 활용할 수 있다면 `Quality + PA`와 `Quality + HR`가 같은 최상위 quality-based prediction-first family로 보는 것이 가장 정확하다.
   - 이 중 operational default 하나를 고르면 `Age + Quality of Sleep + Physical Activity Level + Diastolic BP + male + bmi_risk` 조합이 가장 자연스럽다.
2. **개입 우선순위**
   - 혈압 관리가 가장 핵심이고, `BMI Risk`가 높거나 quality score가 낮은 군은 우선 관리 대상으로 볼 수 있다.
3. **설명 가능한 보수적 baseline**
   - quality score를 덜 쓰고 싶다면 `Sleep + PA` 조합을 baseline screening 도구로 둘 수 있다.
4. **subtype 후속 평가**
   - 고위험군으로 선별된 뒤에는 다항 로지스틱 결과를 함께 봐서 `Insomnia`와 `Sleep Apnea` 가능성을 구분하는 흐름이 자연스럽다.

## 8. 최신 결과를 반영한 비판적 검토

데이터 자체를 신뢰할 수 있다고 가정하더라도, 방법론 관점에서 아직 남는 주의점은 있다.

### 8.1 반복 프로파일 구조는 여전히 중요하다

- grouped CV로 직접적인 train-test profile 중복 문제는 줄였지만,
- effective sample size가 줄어드는 구조 자체는 그대로다.

의미:

- 성능 추정은 이전보다 정직해졌지만, 표본 수 해석은 여전히 보수적이어야 한다.

### 8.2 tuning objective가 확률 품질에 직접적인 영향을 줬다

- `roc_auc` 기준으로 튜닝한 raw probability는 일부 모델에서 F1, Brier, ECE가 크게 흔들렸다.
- 반면 repeated grouped `neg_log_loss` 기준으로 다시 튜닝하면 같은 원 변수 모델들도 `Brier 0.075~0.081`, `ECE 0.082~0.093` 수준으로 훨씬 안정적이었다.

의미:

- 현재 데이터에서는 “추가 calibration을 붙일 것인가”보다 **무엇을 최적화하며 튜닝할 것인가**가 더 중요했다.
- 따라서 절대위험 확률을 쓸 때는 calibration만 볼 것이 아니라, tuning objective 자체를 확률 품질에 맞추는 것이 먼저다.

### 8.3 혈압 외 변수의 독립성은 모델별로 차이가 있다

- `Diastolic BP`는 어떤 모델에서도 가장 안정적이었다.
- `Quality of Sleep`은 quality 기반 상위 모델에서는 유의하게 남았지만,
- `Sleep Duration`, `Physical Activity Level`, `Heart Rate`는 예측에는 기여해도 독립 효과는 더 약하거나 모델 의존적이었다.

의미:

- “혈압은 핵심 변수”라고 말하는 것은 꽤 안정적이다.
- 반면 혈압 외 변수는 `예측 보조 변수`와 `독립 위험/보호 인자`를 구분해서 해석해야 한다.

### 8.4 Occupation은 여전히 해석용 변수에 가깝다

- 강한 차이는 보였지만 셀 희소성이 남아 있어, 최종 모델 입력 변수로 쓰기에는 여전히 불안정하다.

의미:

- 직업군 결과는 설명적 인사이트로는 유지하되, 최종 추천 모델에는 넣지 않는 것이 맞다.

### 8.5 상호작용 모델은 성능 이득보다 복잡도 증가가 더 컸다

- `Compressed + Interaction`은 확률 품질을 일부 회복했지만,
- 기본 추천 모델을 압도할 만큼 일관된 우위는 없었다.

의미:

- 지금 단계에서는 “더 복잡한 모델”보다 “더 안정적인 모델”을 우선 추천하는 것이 타당하다.

### 8.6 predictor profile과 라벨 사이에 모호성이 남아 있다

- predictor profile만 기준으로 보면 `109`개 profile 중 `22`개가 둘 이상의 `Sleep Disorder` 라벨을 가졌다.
- 행 기준으로 보면 전체의 약 `34.2%`가 이런 ambiguous profile에 속했다.

의미:

- 이건 단순 train-test 누수 문제와는 다른 층위의 한계다.
- 즉, 현재 입력 변수만으로는 같은 profile에 대해 서로 다른 결과가 관측되는 구조가 이미 존재한다는 뜻이다.
- 따라서 어떤 모델이든 완전한 결정 규칙처럼 해석하면 안 되고, **확률적 경향 모델**로 보는 것이 맞다.

### 8.7 단일 holdout 결과는 생각보다 split-sensitive하다

- grouped nested CV 기준으로는 현재 추천 모델이 가장 균형이 좋았지만,
- 단일 grouped holdout에서는 더 단순한 축소 모델이 일시적으로 더 좋아 보이는 경우도 확인됐다.

의미:

- 최종 모델 선택은 한 번의 holdout 결과가 아니라, 여러 fold를 평균한 grouped nested CV를 우선 기준으로 둬야 한다.
- 따라서 현재 메인 권고는 “절대적으로 유일한 최적 모델”이라기보다, **반복 검증 기준 가장 안정적인 모델**로 이해하는 것이 맞다.

### 8.8 threshold 해석은 아직 기술적(descriptive) 수준이다

- 추가 실험에서 repeated grouped split 기준 threshold 분포까지 다시 봤지만,
- 이것도 여전히 내부 재표본화 검증이지 외부 검증은 아니다.

의미:

- 현재 threshold 결과는 모델 간 상대 비교와 내부 안정성 점검에는 유용하다.
- 하지만 실제 배치용 cutoff를 정하려면 별도 calibration set 또는 외부 검증에서 threshold를 다시 고정해야 한다.

### 8.9 예측 성능과 회귀 해석은 서로 다른 데이터 층위에서 평가됐다

- 성능 평가는 전체 데이터의 grouped CV / grouped holdout 기준이다.
- 반면 Box-Tidwell, 영향점, bootstrap OR는 `person_id` 제거 후 중복 행을 줄인 `deduplicated` 데이터에서 수행했다.

의미:

- 이 구성은 “예측 모델 평가”와 “계수 해석 안정성 평가”를 분리했다는 점에서 합리적이었다.
- 추가로 clustered GEE를 수행하면서 이 간극은 이전보다 줄었다.
- 다만 여전히 `성능상 추천`과 `독립 효과 해석`을 완전히 같은 근거로 본 것처럼 표현하지 않는 편이 더 정확하다.

### 8.10 이진 모델 결론은 subtype 차이를 압축한 결과다

- 메인 보고서의 최종 추천 모델은 `has_sleep_disorder` 이진 타깃 기준이다.
- 하지만 다항 로지스틱 결과에서는 `Insomnia`와 `Sleep Apnea`가 서로 다른 feature 축을 보였다.

의미:

- 따라서 “수면장애와 관련된 feature”라는 표현은 정확히는 `어떤 수면장애든 하나라도 있을 확률`에 대한 요약이다.
- subtype별 기전이나 차이를 설명하려면 다항 로지스틱 결과를 함께 봐야 한다.

### 8.11 최종 추천은 목적 의존적이다

- repeated grouped `neg_log_loss` 기준 prediction-first 모델은 `Quality + PA`였다.
- grouped bootstrap 기준으로는 `Quality + PA`와 `Quality + HR`가 사실상 같은 최상위 family로 남았다.
- 하지만 `Sleep + PA`도 매우 근접했고, self-reported quality score를 덜 쓰는 장점이 있다.

의미:

- 현재 데이터에서는 “절대적 1위 모델” 하나를 고집하기보다,
- `prediction-first`와 `보수적 baseline`을 나눠 제시하는 편이 더 정직하다.

### 8.12 exact-row dedup과 nested grouped bootstrap 이후에도 결론은 유지됐다

- exact-row dedup 후에도 full-row 단변량에서 보이던 핵심 방향은 대부분 유지됐다.
- 약해진 것은 `Sleep Quality per Hour`, `Steps per Activity` 같은 비율형 파생변수였고, 메인 해석 축은 흔들리지 않았다.
- nested grouped bootstrap 안에서 다시 튜닝해도 `Quality + PA`와 `Quality + HR`는 사실상 같은 최상위 family였다.
- 다만 `Quality + PA`는 `Brier 0.076`으로 가장 좋은 확률 품질을, `Quality + HR`는 winner share `0.400`으로 가장 많은 split 승리를 보여 단일 압승 모델은 아니었다.

의미:

- 현재 operational default는 `실용적 선택`이지 `압도적 유일 승자`가 아니다.
- 대신 `quality-based family가 최상위권이고, 그 안에서 확률 품질을 우선하면 Quality + PA를 쓴다`는 식의 권고는 꽤 단단해졌다.

## 9. 최종 결론

최신 결과를 모두 반영하면, 이번 프로젝트의 메인 결론은 아래처럼 정리하는 것이 가장 정확하다.

1. exact-row dedup sensitivity까지 포함해도 수면장애는 혈압, 나이, 수면시간, 활동량, 수면의 질과 강하게 관련된다.
2. 변수군 수준에서는 `혈압축 + BMI Risk + Quality of Sleep`가 가장 안정적으로 남고, 최종 고정 모델 안에서는 `Diastolic BP`가 가장 강건한 양(+) 방향 신호다.
3. prediction-first 관점의 최상위 모델군은 `Quality + PA`와 `Quality + HR`의 quality-based family다.
4. 실무적으로 하나의 default가 필요하면 `Age + Quality of Sleep + Physical Activity Level + Diastolic BP + male + bmi_risk` 조합을 쓰는 것이 가장 자연스럽지만, 이는 tied top family 안에서의 실용적 선택으로 읽는 편이 더 정확하다.
5. `Age + Sleep Duration + Physical Activity Level + Diastolic BP + male + bmi_risk` 조합은 quality score를 덜 쓰는 보수적 baseline으로 여전히 매우 근접한 성능을 유지한다.
6. 파생변수 모델은 보조 대안으로는 유용하지만, 현재 데이터에서는 원 변수 기본형보다 명확히 우위라고 보기 어렵다.

즉, 메인 보고서 기준 최종 권고는 다음 한 문장으로 압축할 수 있다.

> 현재 데이터에서 `수면장애 유무` screening의 prediction-first 최상위 모델군은 `Quality + PA`와 `Quality + HR`이며, operational default 하나를 고르면 `Age + Quality of Sleep + Physical Activity Level + Diastolic BP + male + bmi_risk` 조합이 가장 자연스럽다. 다만 이는 tied top family 안에서 확률 품질을 더 중시한 실용적 선택으로 읽는 편이 더 정확하다. 변수 수준에서는 `혈압축 + BMI Risk + Quality of Sleep`가 가장 안정적으로 남고, 그중 `이완기혈압`은 최종 고정 모델 안에서 가장 강건한 핵심 위험 신호다.

## 10. 관련 결과물

### 10.1 메인 문서

- [수면장애 통합 메인 보고서](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/sleep_disorder_statistical_summary_ko.md)
- [최종 엄밀성 보강 보고서](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/final_rigor_upgrade/final_rigor_upgrade_report_ko.md)
- [Grouped Bootstrap 모델 비교 및 변수 안정성 후속 보고서](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/bootstrap_stability_followup/bootstrap_stability_followup_report_ko.md)
- [비판 포인트 해소를 위한 추가 검증 보고서](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/critical_resolution/critical_resolution_report_ko.md)
- [후속 엄밀성 검증 보고서](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/rigorous_validation/rigorous_validation_report_ko.md)

### 10.2 재현 스크립트

- [sleep_disorder_statistical_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/sleep_disorder_statistical_analysis.py)
- [ridge_logistic_feature_engineering_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/ridge_logistic_feature_engineering_analysis.py)
- [multinomial_sensitivity_analysis.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/multinomial_sensitivity_analysis.py)
- [rigorous_followup_validation.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/rigorous_followup_validation.py)
- [critical_resolution_experiments.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/critical_resolution_experiments.py)
- [bootstrap_stability_followup.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/bootstrap_stability_followup.py)
- [final_rigor_upgrade.py](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/analysis/final_rigor_upgrade.py)

### 10.3 최신 시각화

- [rigorous_validation figures](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/rigorous_validation/figures)
- [rigorous_validation tables](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/rigorous_validation/tables)
- [critical_resolution figures](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/critical_resolution/figures)
- [critical_resolution tables](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/critical_resolution/tables)
- [bootstrap_stability_followup figures](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/bootstrap_stability_followup/figures)
- [bootstrap_stability_followup tables](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/bootstrap_stability_followup/tables)
- [final_rigor_upgrade figures](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/final_rigor_upgrade/figures)
- [final_rigor_upgrade tables](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/final_rigor_upgrade/tables)
