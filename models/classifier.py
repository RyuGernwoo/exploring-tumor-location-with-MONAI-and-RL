import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121

class CXRClassifier(nn.Module):
    """
    고해상도 패치 및 원본 리사이즈 이미지에서 패턴을 예측하는
    Phase 0 및 Phase 3 조인트 학습용 정밀 진단 분류 모델 (DenseNet121 기반).
    """
    def __init__(self, spatial_dims=2, in_channels=1, out_classes=14):
        super(CXRClassifier, self).__init__()
        # MONAI의 내장 DenseNet121 모델 활용
        self.model = DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_classes
        )
        # Multi-label 의사결정을 위한 마지막 Sigmoid 레이어 (BCE Loss와 결합)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.model(x)
        return self.sigmoid(logits)

if __name__ == '__main__':
    # 모델 정의 테스트
    model = CXRClassifier()
    dummy_input = torch.randn(2, 1, 224, 224)
    out = model(dummy_input)
    print(f"Output shape (Batch, Classes): {out.shape}")
