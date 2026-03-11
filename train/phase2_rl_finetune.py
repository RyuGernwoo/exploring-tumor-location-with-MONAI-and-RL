import torch

def train_phase2_ppo():
    """
    Phase 2: RL Fine-Tuning 
    
    1. Phase 0/1을 거쳐 어느 정도 위치로 가는 법을 배운 PPO Agent 및 
       가중치가 동결(Freeze)된 Classifier(DenseNet) 준비.
    2. data/env.py 환경 루프(Episode) 속에서 Step(동작) 샘플 수집.
    3. IoU 증가량에 대한 즉각 보상, 발자국/루핑 패널티, 최종 터미널 
       예측(Classifier Score) 기반 누적 합산 리턴 스칼라 사용.
    4. Actor-Critic 오차(Surrogate Advantage) 산출을 통한 정책 업데이트.
    """
    pass

if __name__ == "__main__":
    train_phase2_ppo()
