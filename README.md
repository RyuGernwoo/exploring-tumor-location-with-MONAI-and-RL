# 강화 학습 기반 CXR ROI 탐색 및 진단 모델 구현 계획

고해상도 흉부 X-ray(CXR) 원본 데이터에서 강화학습(RL) 에이전트를 통해 질병 의심 구역(ROI)을 우선적으로 찾고, 해당 패치만을 정밀 분석하여 연산 리소스를 절감하는 진단 추천 시스템 구축 목적입니다.

1024x1024 고해상도 이미지를 저해상도 글로벌 뷰와 고해상도 로컬 패치로 이원화하여 연산 효율을 극대화하고, NIH 데이터셋의 Multi-label 정보를 활용 및 BBox 데이터를 초기 학습에 사용하여 전체적인 아키텍처를 구성합니다.

---

## 1. 강화 학습 환경 (RL Environment) 설계

### 1.1 상태 (State) 및 공간 구조
*   **물리적 정보**: 현재 에이전트가 바라보고 있는 윈도우(Window)의 위치와 크기 좌표 정보.
*   **시각적 정보**: 전체 이미지의 저해상도 뷰(Global Context, 예: 256x256) 및 현재 윈도우의 특징 데이터.
*   **시간/이력 문맥 (Temporal & History Context)**: 에이전트가 과거 탐색 궤적을 맴도는 무한 루프(Looping)에 빠지는 것을 방지하기 위해 **LSTM/GRU 모듈**을 백본에 결합하여 직전 $k$ 스텝의 이동 이력(Action History) 및 시계열 변화량을 State로 기억하도록 합니다. 또는 각 스텝의 ROI 마스크(Mask) 채널을 Global Context에 차례대로 쌓아서 누적 방문 상태 맵을 입력으로 활용합니다.

### 1.2 행동 (Action) 공간 (Continuous Control)
*   **탐색 제어 (Scale & Shift)**: DQN 기반의 고정된 보폭 이동(상/하/좌/우 이산 제어)이 아닌, **PPO (Proximal Policy Optimization)** 알고리즘을 활용한 **연속 행동 공간(Continuous Action Space)** 제어를 설계합니다.
    *   에이전트는 $[-1, 1]$ 사이의 벡터 $[dx, dy, dw, dh]$ 차원을 출력합니다. 중심 좌표($dx, dy$)의 미세 이동과 윈도우 크기 변화 비율($dw, dh$) 파라미터로 사용하여, 환경 내에서 동적이고 정밀한 ROI Stride/Zoom 궤적을 산출합니다.
*   **확정 (Terminal Action)**: "여기가 질병 의심 구역이다"라고 선언하고 프레임워크에 정밀 진단 분석을 요청하는 중단 시그널.

### 1.3 보상 함수 (Reward Shaping)
에이전트의 신속하고 안정적인 수렴을 위해 단일 이진 보상(+) 대신 세분화된 보상 체계를 결합합니다.
*   **$\Delta$ IoU 점진적 보상 (Dense Reward)**: 이전 탐색 윈도우 대비 현재 스텝에서의 실제 BBox(Ground Truth)와의 **IoU (Intersection over Union)** 증가량에 비례하여 양(+)의 보상을 지급하여, 정답 지점을 향한 자연스러운 흐름을 유도합니다. (멀어질 경우 패널티 부여).
*   **탐색 효율성 패널티 (Efficiency & Coverage Penalty)**: 
    *   스텝 길이에 비례한 기본 감가(음(-)의 패널티).
    *   이미 방문했던 영역과 높은 IoU로 겹치는 구역을 재탐색할 경우 주어지는 순환 패널티로, 새로운 이미지 영역을 탐색하도록 강제합니다.
*   **최종 진단 보상**: 확정(Terminal) 액션 후, 선별된 최종 ROI 내에서 분류 모델(Classifier)이 도출한 예측 클래스의 정답 확률과 실제 라벨 간의 BCE Loss 혹은 일치도를 최종 Reward로 사용합니다.

---

## 2. 모델 아키텍처 (Model Architecture)

*   **탐색 에이전트 (RL Agent)**:
    *   **PPO (Proximal Policy Optimization)** 알고리즘을 기반으로, ResNet-18 등의 CNN 백본을 통해 현재 윈도우 피처를 추출하고, Global 뷰와 LSTM 모듈에서 생성된 은닉 층(Hidden State)을 결합하여 Actor-Critic 네트워크의 입력으로 사용합니다.
*   **정밀 진단 분류 모델 (Classifier)**:
    *   MONAI 프레임워크의 `DenseNet121` 또는 `EfficientNet`을 활용하여 최종 14종 병변 (Atelectasis, Cardiomegaly 등)에 대한 Multi-label 분류를 수행합니다.

---

## 3. 학습 전략 (Step-by-Step Training)

에이전트가 탐색 과정을 수행할 때 일관된 목적지침을 갖기 위해서는 보상으로 활용될 분류기 모델이 먼저 일정 수준 이상의 정확도를 갖추어야 합니다.

*   **Phase 0 (Classifier Pre-training)**:
    *   본격적인 RL 강화학습 전, 제공되는 BBox 데이터 영역으로 크롭(Crop) 및 리사이즈된 이미지 패치들을 이용하여 DenseNet121 분류 모델을 먼저 지도 학습(Supervised Learning)시킵니다. 에이전트가 질병 위치를 정확히 짚어냈을 때 신뢰성 있는 보상(정답 확률)을 모델이 출력할 수 있도록 가중치를 초기화하는 과정입니다.
*   **Phase 1 (Agent Supervised Pre-training)**:
    *   BBox 정답이 존재하는 일부 훈련 이미지를 활용하여, 에이전트가 정답 위치로 빠르게 이동하는 궤적을 모방하도록(Imitation Learning/Behavior Cloning) 지도 학습시킵니다.
*   **Phase 2 (RL Fine-tuning in Frozen Classifier)**:
    *   Phase 0 에서 학습된 Classifier의 가중치를 고정(Freeze)하고, 전체 데이터셋 환경으로 확장하여 환경-에이전트 간의 상호작용 속에서 PPO 정책 모델을 최적화(조밀 보상 수집)합니다.
*   **Phase 3 (Joint E2E Optimization)**:
    *   탐색(Agent)과 진단(Classifier) 모듈을 조인하여, 에이전트의 ROI가 진단 정확도를 극대화시키는 방향으로 양쪽 네트워크를 동시에 미세 조정(Fine-tuning) 합니다.

---

## 4. Verification Plan

### 시각적/정량적 검증 (Automated & Manual)
*   **탐색 궤적 시각화**: 테스트 셋에 대해 $t=0$ 에서 $t=terminal$ 까지 에이전트의 윈도우 이동 궤적을 애니메이션(GIF/Video) 형태로 확인하여 무한 루프나 발산 여부 모니터링.
*   **지표 측정**:
    *   Agent가 BBox를 찾아낸 평균 탐색 스텝 수 및 BBox와의 최종 평균 IoU 확률.
    *   기존 원본 고해상도(1024x1024) 이미지를 통째로 사용할 때 대비, 256x256 View + 고해상도 ROI 패치 시스템의 **연산 속도 증가율 및 FLOPs 감소폭** 비교 검증.
