# 수면장애 분석 인사이트 및 활용 요약

## 1. 핵심 인사이트

### 1.1 가장 안정적인 핵심 신호는 이완기혈압이다

이번 프로젝트 전체에서 가장 일관되게 남은 변수는 `Diastolic BP`였다.

- robust test에서 강했다.
- deduplicated GLM에서도 유지됐다.
- profile-cluster GEE sensitivity analysis에서도 유지됐다.

즉, 어떤 모델을 쓰더라도 **이완기혈압은 수면장애 위험과 가장 안정적으로 연결되는 축**이라고 볼 수 있다.

### 1.2 수면의 질은 예측 우선 모델에서 중요한 보호 신호다

quality 기반 상위 모델에서는 `Quality of Sleep`이 유의한 보호 방향 신호로 반복 확인됐다.

따라서 수면장애 screening에서 단순 수면시간만 보는 것보다, **주관적 회복감과 수면의 질 정보까지 함께 보는 것이 더 유리**했다.

### 1.3 단일 모델 하나만 정답이라고 보기는 어렵다

추가 검증까지 포함하면 최종 추천은 두 층위로 나뉜다.

1. 예측 성능 우선:
   - `Age + Quality of Sleep + Physical Activity Level + Diastolic BP + male + bmi_risk`
2. 보수적 baseline:
   - `Age + Sleep Duration + Physical Activity Level + Diastolic BP + male + bmi_risk`

즉, 실제 활용 목적에 따라 모델 선택 기준이 달라진다.

### 1.4 수면장애는 하나의 subtype으로만 해석하면 안 된다

다항 로지스틱과 repeated grouped multinomial validation 결과를 보면,

- `Insomnia`
- `Sleep Apnea`

는 서로 다른 feature 축을 가진다.

따라서 초기 screening은 이진 모델로 하더라도, 이후 해석과 개입은 subtype 분리를 고려하는 것이 더 타당하다.

## 2. 어떻게 활용할 수 있는가

### 2.1 1차 screening 도구

병원, 건강검진, 웰니스 서비스, 디지털 헬스 환경에서는 아래 흐름으로 활용할 수 있다.

1. `나이`, `혈압`, `BMI`, `수면 관련 지표`, `활동량`을 입력한다.
2. 수면장애 고위험군을 선별한다.
3. 고위험군에 대해 추가 검사 또는 정밀 평가를 권고한다.

prediction-first로는 quality 기반 모델이 적합하고, 보다 단순한 baseline이 필요하면 `Sleep + PA` 조합을 쓸 수 있다.

### 2.2 개입 우선순위 설계

이번 결과를 바탕으로 우선순위를 잡으면 다음이 자연스럽다.

1. 혈압 관리
2. 수면의 질 개선
3. 수면시간 부족 교정
4. 생활습관 조정

즉, 단순히 “더 오래 자라”가 아니라, **혈압과 회복감까지 포함한 다축 개입**이 더 적절하다.

### 2.3 subtype 후속 분류

고위험군으로 선별된 뒤에는 subtype별로 후속 해석을 다르게 가져갈 수 있다.

- `Insomnia` 가능성이 높으면 수면-스트레스 불균형 중심 평가
- `Sleep Apnea` 가능성이 높으면 혈압부담과 심혈관 축 중심 평가

이렇게 하면 screening 이후의 후속 조치도 더 정교해질 수 있다.

### 2.4 설명 가능한 의사결정 지원

이번 분석은 black-box 모델이 아니라 해석 가능한 로지스틱 기반 결과를 중심으로 정리되어 있다.

그래서 현업에서는 아래처럼 설명하기 쉽다.

- 왜 이 사람이 고위험으로 분류됐는가
- 어떤 feature가 위험을 올렸는가
- 어떤 생활습관 지표가 보호 방향이었는가

이는 의료 현장, 보건정책, 서비스 설계 측면에서 큰 장점이다.

## 3. 실무 적용 시 권장 방식

### 3.1 예측 우선 적용

quality score를 확보할 수 있다면 아래 모델이 가장 적합하다.

- `Age + Quality of Sleep + Physical Activity Level + Diastolic BP + male + bmi_risk`

이 모델은 확률 품질과 screening 성능을 함께 볼 때 가장 설득력이 있었다.

### 3.2 보수적 baseline 적용

quality score를 덜 쓰거나 측정 보수성을 더 중시한다면 아래 모델이 적합하다.

- `Age + Sleep Duration + Physical Activity Level + Diastolic BP + male + bmi_risk`

이 모델은 quality 기반 모델보다 약간 뒤처졌지만 여전히 매우 근접한 성능을 유지했다.

## 4. 주의해서 활용해야 할 점

1. threshold는 내부 반복 검증까지는 했지만 외부 검증으로 확정된 값은 아니다.
2. 상위 모델 간 차이는 크지 않아서 목적에 맞는 선택이 중요하다.
3. `Diastolic BP`는 핵심 변수라고 말할 수 있지만, 일부 변수는 예측 기여와 독립 효과를 구분해서 해석해야 한다.
4. subtype 해석이 필요할 때는 이진 screening 결과만으로 결론을 내리면 안 된다.

## 5. 한 문장 요약

이번 분석이 주는 가장 실용적인 메시지는 아래와 같다.

> 수면장애 screening에서는 `이완기혈압`이 가장 안정적인 핵심 신호이며, 예측 성능을 높이려면 `수면의 질`을 함께 쓰는 것이 유리하고, 이후 해석과 개입은 `Insomnia`와 `Sleep Apnea`를 분리해서 접근하는 것이 더 효과적이다.
