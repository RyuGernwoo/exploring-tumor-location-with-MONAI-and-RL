import torch

def train_phase0():
    """
    Phase 0: Classifier Pre-training (Supervised)
    
    1. BBox_List_2017.csv에 나열된 약 1,000개의 이미지 리스트를 로드.
    2. Ground Truth BBox 영역만 Crop한 부분 이미지 또는 GT 위치의 패치 샘플 추출.
    3. 이를 DenseNet121 정밀 진단 모듈(models/classifier.py)에 투입.
    4. 실제 병변 클래스(Multi-label)에 대해 BCE Loss 방식으로 모델 사전 학습.
    에이전트에게 확정시 정확도 보상을 반환해주기 위한 핵심 신뢰성(Reliability) 확보 목적.
    """
    pass

if __name__ == "__main__":
    train_phase0()
