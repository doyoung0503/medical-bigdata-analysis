# 수면장애 분석 흐름과 실험 당위성 정리

## 1. 왜 이 문서를 따로 만들었는가

이번 프로젝트는 한 번에 끝난 단일 분석이 아니라, 초기 통계분석에서 시작해 공선성 대응, subtype 분석, 민감도 분석, 엄밀성 검증, 비판 포인트 해소 실험까지 단계적으로 확장된 분석이었다.

따라서 최종 결과를 이해하려면 “무슨 분석을 했는가”보다도,

1. 왜 그 실험을 먼저 했는가
2. 그 결과를 어떻게 해석했는가
3. 그래서 왜 다음 실험이 필요하다고 판단했는가

를 순서대로 보는 것이 중요하다.

이 문서는 바로 그 흐름을 정리한 요약본이다.

## 2. 전체 분석 흐름 한눈에 보기

1. 데이터셋 선택과 기초 통계분석
2. 상관관계와 다중공선성 확인
3. 이진 로지스틱 회귀 초기 모델
4. Ridge 로지스틱 회귀와 파생변수 탐색
5. 다항 로지스틱 회귀와 민감도 분석
6. 후속 엄밀성 검증
7. 비판 포인트 해소를 위한 추가 검증
8. grouped bootstrap 모델 비교와 변수 안정성 후속 검증
9. exact-row dedup sensitivity와 nested grouped bootstrap 최종 보강

핵심 원칙은 항상 같았다.

- 먼저 `관련성이 있는가`를 본다.
- 그다음 `공선성과 모델 안정성`을 본다.
- 그다음 `결과가 검증 설계와 가정 변화에도 유지되는가`를 본다.
- 마지막으로 `비판 포인트가 실제로 해소되는가`를 확인한다.

## 3. 1단계: 데이터셋 선택과 기초 통계분석

### 왜 이 실험을 먼저 했는가

통계분석은 먼저 “어떤 데이터가 종속변수를 가지고 있고, 집단 비교와 회귀분석을 수행할 수 있는가”를 확인해야 시작할 수 있다.

그래서 `dataset` 폴더의 두 데이터셋을 비교했고, 최종적으로 `Sleep_health_and_lifestyle_dataset`을 선택했다.

선택 이유는 명확했다.

1. `Sleep Disorder`라는 직접적인 종속변수가 있었다.
2. 연속형 변수와 범주형 변수가 함께 있었다.
3. `None / Insomnia / Sleep Apnea` 구조라 이진/다항 분석이 모두 가능했다.
4. 결측과 클래스 불균형이 극단적이지 않았다.

### 무엇을 했는가

- 기술통계
- 집단별 평균 비교
- 범주형 분포 비교
- 시각화 기반 분포 확인

### 무엇이 나왔는가

초기 단계에서 이미 혈압, 나이, 수면시간, 수면의 질, 활동량, 심박수 축이 수면장애와 관련될 가능성이 높다는 신호가 나타났다.

### 어떻게 해석했는가

이 단계에서는 아직 “최종 모델”을 말할 수는 없었지만, 적어도 수면장애가 단일 변수 하나가 아니라

- 혈압을 포함한 심혈관 축
- 수면시간/수면의 질 축
- 생활습관 축

이 함께 움직이는 구조라는 점은 확인할 수 있었다.

### 그래서 다음에 무엇을 해야 한다고 판단했는가

단순 평균 차이만으로는 충분하지 않았다. 어떤 변수들이 서로 겹치고 있는지, 즉 공선성 구조를 먼저 확인해야 회귀모형을 안전하게 설계할 수 있었다.

그래서 다음 단계는 `상관관계와 다중공선성 분석`이었다.

## 4. 2단계: 상관관계와 다중공선성 확인

### 왜 이 실험이 필요했는가

로지스틱 회귀는 해석이 강점이지만, 설명변수들이 서로 매우 강하게 상관되어 있으면 계수 해석이 불안정해진다.

그래서 본격적인 회귀에 앞서 다음을 확인했다.

- 상관행렬
- VIF
- 변수군 간 정보 중복

### 무엇이 나왔는가

특히 다음 관계가 강했다.

- `Systolic BP` 와 `Diastolic BP`
- `Sleep Duration` 과 `Quality of Sleep`
- `Quality of Sleep` 과 `Stress Level`

즉, 원 변수들을 그대로 다 넣으면 계수 해석과 변수 선택이 흔들릴 수 있는 구조였다.

### 어떻게 해석했는가

이 결과는 두 가지를 의미했다.

1. 원 변수 기반 해석형 모델은 대표 변수만 선택하는 방식이 필요하다.
2. 파생변수나 규제 회귀로 정보 압축을 시도할 가치가 있다.

### 그래서 다음에 무엇을 해야 한다고 판단했는가

우선은 가장 기본적인 해석형 기준점을 만들 필요가 있어서 `이진 로지스틱 회귀 초기 모델`을 먼저 적합했고, 이후에는 공선성을 줄이기 위한 `Ridge + 파생변수`로 넘어가기로 했다.

## 5. 3단계: 이진 로지스틱 회귀 초기 모델

### 왜 먼저 이진 모델을 만들었는가

프로젝트의 가장 직접적인 질문은 “수면장애가 있는가 없는가”였다.

따라서 subtype을 나누기 전에, 먼저 `has_sleep_disorder` 이진 타깃으로 기본 해석형 모델을 만드는 것이 자연스러웠다.

### 무엇을 했는가

- 수면장애 유무를 종속변수로 한 로지스틱 회귀
- 원 변수 기반 후보 모델 비교
- 오즈비 해석

### 무엇이 나왔는가

초기에는 `Age`, `Diastolic BP`, `Quality of Sleep` 같은 조합이 해석상 설득력 있어 보였다.

### 어떻게 해석했는가

이 단계에서 얻은 가장 중요한 메시지는 “혈압 축이 강하다”는 것이었다.

다만 공선성 구조를 이미 확인했기 때문에, 이 결과를 최종 결론으로 두기엔 아직 이르다고 판단했다.

### 그래서 다음에 무엇을 해야 한다고 판단했는가

공선성 문제를 완화하면서 더 안정적인 예측 모델을 만들기 위해 `Ridge 로지스틱 회귀와 파생변수 탐색`이 필요했다.

## 6. 4단계: Ridge 로지스틱 회귀와 파생변수 탐색

### 왜 Ridge와 파생변수를 썼는가

공선성이 강한 데이터에서는 계수를 강하게 수축하는 Ridge가 유리할 수 있다. 또 혈압군, 수면-스트레스군처럼 서로 얽힌 변수들을 요약한 파생변수를 만들면 해석과 안정성을 함께 개선할 수 있다.

### 무엇을 했는가

- Ridge 로지스틱 회귀
- 파생변수 설계
- 원 변수 세트 vs 압축형 세트 비교
- VIF 재점검

### 어떤 feature engineering을 했는가

핵심 파생변수는 아래와 같았다.

1. `Mean Arterial Pressure`
2. `Pulse Pressure`
3. `Sleep Deficit (vs 7h)`
4. `Sleep-Stress Balance`

### 무엇이 나왔는가

압축형 변수 세트는 공선성을 낮추는 데 효과가 있었고, 예측 성능도 유지했다. 특히 `Compressed Derived Ridge`는 성능과 공선성 완화의 균형이 좋아 보였다.

### 어떻게 해석했는가

이 단계에서 얻은 결론은 “원 변수만이 답은 아니다”였다. 혈압군과 수면군을 적절히 압축하면 예측에는 충분히 쓸 만한 모델이 나올 수 있었다.

다만 이것만으로는 아직 부족했다.

- 이 모델이 정말 해석 가능한가
- subtype 차이를 눌러버린 것은 아닌가
- 중복 profile 구조 때문에 성능이 낙관적이지 않은가

라는 질문이 남았다.

### 그래서 다음에 무엇을 해야 한다고 판단했는가

다음 단계는 `다항 로지스틱 회귀와 민감도 분석`이었다.

## 7. 5단계: 다항 로지스틱 회귀와 민감도 분석

### 왜 이 실험이 필요했는가

이진 로지스틱은 `Insomnia`와 `Sleep Apnea`를 모두 하나의 positive class로 묶는다. 하지만 실제로 두 subtype은 서로 다른 변수 축을 가질 수 있다.

또한 같은 predictor profile이 반복되는 구조가 있어, 성능이 과대평가될 가능성도 점검해야 했다.

### 무엇을 했는가

- 다항 로지스틱 회귀
- subtype별 계수와 오즈비 해석
- 중복 제거 전후 성능 비교
- 효과크기 민감도 분석

### 무엇이 나왔는가

subtype 차이는 분명했다.

- `Insomnia`는 `Sleep-Stress Balance`, `Pulse Pressure`, `BMI Risk` 쪽 신호가 더 강했다.
- `Sleep Apnea`는 `Mean Arterial Pressure`, `Heart Rate` 쪽 신호가 더 강했다.

또한 dedup 후에는 성능이 내려갔지만, 중요한 변수의 방향성과 효과크기는 상당 부분 유지됐다.

### 어떻게 해석했는가

이 단계는 매우 중요했다.

1. 이진 screening 모델은 유용하지만 subtype 차이를 모두 설명하진 못한다.
2. 성능 수치만 보면 안 되고, 중복 구조에 대한 민감도를 반드시 봐야 한다.

### 그래서 다음에 무엇을 해야 한다고 판단했는가

이제 남은 질문은 “이 결과가 검정 방식과 검증 설계가 더 엄격해져도 유지되는가”였다.

그래서 다음 단계는 `후속 엄밀성 검증`이었다.

## 8. 6단계: 후속 엄밀성 검증

### 왜 이 실험이 필요했는가

기존 분석은 전반적으로 타당했지만, 아직 다음 문제들이 남아 있었다.

1. 단변량 결론이 검정 방식에 민감한가
2. random split 성능이 과대평가된 것은 아닌가
3. ROC-AUC는 높지만 확률은 불안정한가
4. 로지스틱 가정이 크게 깨지지는 않는가

### 무엇을 했는가

1. `Welch ANOVA`, `Kruskal-Wallis`, `FDR`, permutation chi-square
2. grouped CV와 grouped holdout bootstrap
3. calibration intercept/slope, Brier, ECE, threshold sweep
4. Box-Tidwell, 영향점, bootstrap OR

### 무엇이 나왔는가

- 혈압, 나이, 수면시간, 활동량, 수면의 질 축은 robust test에서도 유지됐다.
- grouped CV를 적용해도 핵심 모델들의 분리력은 유지됐다.
- 이 단계에서는 `Original: Sleep + PA`가 가장 균형 좋은 기본형처럼 보였다.
- `Diastolic BP`는 bootstrap OR에서도 가장 안정적이었다.

### 어떻게 해석했는가

이 단계에서 “혈압이 핵심”이라는 메시지는 꽤 단단해졌다.

하지만 동시에 새로운 비판 포인트도 드러났다.

1. threshold 해석이 single holdout에 너무 의존했다.
2. 성능 평가는 full data, 오즈비 해석은 deduplicated data로 나뉘어 있었다.
3. tuning objective에 따라 추천 모델이 달라질 가능성이 남아 있었다.

### 그래서 다음에 무엇을 해야 한다고 판단했는가

이제는 단순히 또 다른 모델을 추가하는 게 아니라, 남아 있는 비판 포인트를 직접 해소하는 실험이 필요했다.

그래서 마지막 단계로 `비판 포인트 해소를 위한 추가 검증`을 수행했다.

## 9. 7단계: 비판 포인트 해소를 위한 추가 검증

### 왜 이 실험이 필요했는가

앞 단계까지 오면서 핵심 질문은 아래 세 가지로 좁혀졌다.

1. `threshold`와 `calibration` 결론이 single holdout에 과도하게 의존하는가
2. 예측과 해석의 데이터 층위 차이를 줄일 수 있는가
3. 이진 screening 결론이 subtype 차이를 지나치게 압축하는가

### 무엇을 했는가

1. repeated grouped model comparison
2. `roc_auc` vs `neg_log_loss` tuning objective 비교
3. repeated grouped threshold / calibration 재검증
4. profile-cluster GEE sensitivity analysis
5. repeated grouped multinomial validation

### 무엇이 나왔는가

가장 중요한 결과는 아래와 같았다.

1. repeated grouped 기준에서는 `Original: Quality + PA`가 prediction-first 기본형으로 가장 설득력 있었다.
   - repeated grouped `neg_log_loss` 기준 winner share `0.300`
   - `ROC-AUC 0.936`
   - `F1 0.895`
   - `Brier 0.075`
2. `Original: Quality + HR`도 매우 근접했고, 이후 grouped bootstrap까지 포함하면 두 모델을 같은 quality-based top family로 보는 편이 더 정확해졌다.
3. `Original: Sleep + PA`는 self-reported quality를 덜 쓰는 보수적 baseline으로 여전히 경쟁력이 있었다.
4. threshold 문제는 repeated grouped 재검증으로 크게 완화됐고,
   추가 calibration보다 `neg_log_loss` 기준 튜닝이 더 직접적으로 확률 품질을 개선했다.
5. profile-cluster GEE sensitivity analysis에서도 `Diastolic BP`는 안정적이었다.
6. quality 기반 상위 모델에서는 `Quality of Sleep`도 유의한 보호 방향 신호로 유지됐다.
7. repeated grouped multinomial validation에서도 `accuracy 0.880`, `macro-F1 0.849`, `macro-AUC 0.911`로 subtype 구조가 유지됐다.

### 어떻게 해석했는가

이 단계는 최종 결론을 더 정교하게 만들었다.

1. “절대적 단일 최적 모델”보다는 `목적과 재표본화 기준에 따라 권고가 달라진다`고 보는 것이 맞다.
2. repeated grouped 기준에서는 `Quality + PA`가 prediction-first 기본형으로 적합했다.
3. grouped bootstrap까지 포함하면 `Quality + PA / Quality + HR`를 같은 최상위 quality-based family로 읽는 편이 더 엄밀하다.
4. 측정 보수성과 설명 단순성을 중시하면 `Sleep + PA`도 충분히 좋은 baseline이다.
5. 어떤 경우에도 `Diastolic BP`는 최종 고정 모델 안에서 가장 안정적인 핵심 축이다.

## 10. 8단계: grouped bootstrap 모델 비교와 변수 안정성 후속 검증

### 왜 이 실험이 추가로 필요했는가

비판 포인트 해소 실험까지 끝난 뒤에도 마지막으로 두 질문이 남았다.

1. 상위 모델 간 차이가 실제로도 충분히 크다고 볼 수 있는가
2. 최종 변수들이 표본 재추출에도 안정적으로 남는가

즉, 이 단계의 목적은 “최종 권고를 정말 하나로 좁힐 수 있는가”와 “핵심 변수라는 표현이 실제로 타당한가”를 마지막으로 검증하는 데 있었다.

### 무엇을 했는가

1. grouped bootstrap OOB model comparison
2. pairwise superiority analysis
3. L1 stability selection
4. final `Quality + PA` model coefficient bootstrap

### 무엇이 나왔는가

1. grouped bootstrap 기준으로는 `Quality + PA`와 `Quality + HR`가 사실상 같은 최상위 quality-based model family로 남았다.
2. `Quality + PA`는 `Sleep + PA`보다 Brier 측면에서 더 자주 좋았지만, `Quality + HR`를 압도하는 수준은 아니었다.
3. stability selection에서는 `BMI Risk`, `Quality of Sleep`, `Systolic BP`, `Diastolic BP`가 가장 자주 살아남았다.
4. 최종 `Quality + PA` 고정 모델 coefficient bootstrap에서는 `Diastolic BP`, `BMI Risk`, `Quality of Sleep`, `Age`의 부호 안정성이 모두 매우 높았다.

### 어떻게 해석했는가

이 단계는 최종 메시지를 한 단계 더 정교하게 만들었다.

1. 예측 모델 추천은 더 이상 `Quality + PA` 단독이라기보다 `Quality + PA / Quality + HR` quality-based family로 보는 것이 맞다.
2. 변수 수준에서는 개별 `Diastolic BP`만이 아니라 `혈압축 + BMI Risk + Quality of Sleep`가 가장 강건한 축이라고 해석하는 편이 더 정확하다.
3. 다만 최종 `Quality + PA` 고정 모델 안에서는 `Diastolic BP`가 가장 안정적인 양(+) 방향 신호라는 결론이 유지됐다.

## 11. 9단계: exact-row dedup sensitivity와 nested grouped bootstrap 최종 보강

### 왜 이 실험이 마지막으로 더 필요했는가

앞 단계까지 오면서 결론은 상당히 단단해졌지만, 두 비판 포인트가 남아 있었다.

1. 단변량 검정은 full-row 기준인데 예측 검증은 grouped 기준이라 독립성 기준이 완전히 일치하지 않았다.
2. grouped bootstrap 모델 비교는 이전 CV에서 고른 하이퍼파라미터를 고정해 비교했기 때문에, 엄밀히는 fixed-hyperparameter bootstrap이었다.

### 무엇을 했는가

1. exact-row deduplicated 단변량 재검정
2. nested grouped bootstrap 모델 재비교
   - replicate마다 grouped inner CV로 `C`와 class weight를 다시 튜닝
   - OOB test로 성능 재평가

### 무엇이 나왔는가

1. exact-row dedup 후에도 `Age`, `Quality of Sleep`, `Sleep Duration`, `Physical Activity Level`, `Diastolic BP`를 포함한 핵심 축은 그대로 유지됐다.
2. 약해진 것은 `Sleep Quality per Hour`, `Steps per Activity`처럼 비율형 파생변수뿐이었다.
3. nested grouped bootstrap 평균에서는 `Quality + PA`가 `Brier 0.076`으로 가장 좋았고, `Quality + HR`는 `F1 0.889`, winner share `0.400`으로 사실상 동급 최상위였다.
4. 따라서 quality-based family가 최상위권이라는 해석은 fixed-hyperparameter bootstrap에만 의존한 결론이 아니게 됐다.

### 어떻게 해석했는가

이 마지막 보강 단계는 두 가지를 더 분명하게 만들었다.

1. 메인 단변량 인사이트는 반복 row 때문에만 생긴 결과가 아니었다.
2. `Quality + PA` operational default는 유지되지만, 이것을 통계적으로 압도적 유일 승자라고 표현하는 것은 과하다.

즉, 가장 엄밀한 문장은 “`Quality + PA / Quality + HR`가 같은 최상위 family이고, `Quality + PA`는 그 안에서 확률 품질을 더 중시할 때의 practical default”다.

## 12. 최종적으로 남은 핵심 메시지

지금까지의 모든 실험을 순서대로 통합하면, 최종 메시지는 아래처럼 정리된다.

### 11.1 변수 수준 결론

변수군 수준에서는 `혈압축 + BMI Risk + Quality of Sleep`가 가장 안정적으로 남았다.

이 결론은

- robust univariate test
- exact-row dedup sensitivity
- deduplicated GLM
- profile-cluster GEE sensitivity analysis

를 거쳐도 유지됐다.

추가로 stability selection까지 포함하면 `BMI Risk`도 매우 안정적인 축으로 확인됐다.

### 11.2 모델 수준 결론

세 층위의 권고가 가장 정직하다.

1. `prediction-first family`
   - `Quality + PA`
   - `Quality + HR`
2. `operational default`
   - `Age + Quality of Sleep + Physical Activity Level + Diastolic BP + male + bmi_risk`
3. `보수적 baseline`
   - `Age + Sleep Duration + Physical Activity Level + Diastolic BP + male + bmi_risk`

### 11.3 해석 수준 결론

이 분석은 단순히 “수면을 적게 자면 위험하다” 수준이 아니었다.

오히려 더 안정적인 해석은 아래에 가깝다.

- 수면장애는 `혈압 축 + 수면/회복 축 + 생활습관 축`이 함께 움직이는 구조다.
- 그중에서도 최종 고정 모델 안에서 가장 단단한 핵심 신호는 `이완기혈압`이다.
- subtype 수준에서는 `Insomnia`와 `Sleep Apnea`가 동일한 메커니즘으로 설명되지 않는다.

## 13. 그래서 앞으로 또 어떤 실험을 하면 좋은가

이번 분석으로 많은 비판 포인트를 해소했지만, 완전히 끝난 것은 아니다.

다음 실험 후보는 아래 순서가 타당하다.

1. 외부 검증 또는 시간 분리 검증
   - 현재 threshold와 확률 품질은 내부 재표본화 기준까지는 확인했지만, 외부 검증까지는 아니다.
2. decision-curve / cost-sensitive threshold analysis
   - 실제 screening에서 어떤 cutoff가 유리한지 비용 구조까지 반영할 필요가 있다.
3. quality score 대체 변수 또는 quality score 관측 신뢰도 분석
   - quality 기반 모델이 prediction-first로 앞섰기 때문에, 이 변수의 측정 안정성을 추가 확인할 가치가 있다.
4. subtype별 후속 개입 모델
   - `Insomnia`와 `Sleep Apnea`가 다르게 나타났으므로, 이후 개입 설계는 subtype 분리형이 더 자연스럽다.

## 14. 관련 문서

- [통합 메인 보고서](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/sleep_disorder_statistical_summary_ko.md)
- [최종 엄밀성 보강 보고서](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/final_rigor_upgrade/final_rigor_upgrade_report_ko.md)
- [Grouped Bootstrap 모델 비교 및 변수 안정성 후속 보고서](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/bootstrap_stability_followup/bootstrap_stability_followup_report_ko.md)
- [비판 포인트 해소를 위한 추가 검증 보고서](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/critical_resolution/critical_resolution_report_ko.md)
- [인사이트 및 활용 요약](/Users/doyoung/Documents/Bigdata_analysis/sleep_health/results/final_synthesis/insights_and_applications_ko.md)
