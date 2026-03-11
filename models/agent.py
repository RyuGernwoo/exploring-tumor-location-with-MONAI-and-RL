import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class RPPOAgent(nn.Module):
    """
    Recurrent PPO Agent for continuous bounding box exploration.
    내부적으로 현재의 로컬 패치, 전체 글로벌 컨텍스트, 윈도우 좌표를 받아
    LSTM(시계열 상태)을 통해 다음 동작(Shift/Scale)을 결정합니다.
    """
    def __init__(self, in_channels=1, hidden_size=256, action_dim=4):
        super(RPPOAgent, self).__init__()
        
        # 1. Local Patch Feature Extractor (ResNet18)
        self.local_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        if in_channels != 3:
            # Grayscale 1채널 엑스레이 처리를 위한 Conv1 계층 수정
            self.local_extractor.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        feature_dim = self.local_extractor.fc.in_features
        self.local_extractor.fc = nn.Identity()  # 기존 분류기 Head 제거 (Feature만 반환)
        
        # 2. Global Context Extractor (CNN 기반으로 Global View(256x256 등) 압축)
        self.global_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=4), # 64x64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=4), # 16x16
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU()
        )
        
        # 3. Spatial/Coordinate Feature Embedder
        self.coord_embed = nn.Sequential(
            nn.Linear(4, 32), # [cx, cy, w, h] 정규화 비율
            nn.ReLU()
        )
        
        # 합쳐진 특성 벡터의 차원
        combined_dim = feature_dim + 128 + 32
        
        # 4. LSTM for Temporal/Action History
        # 과거 탐색 기록의 문맥(Context)을 기억하기 위한 순환 신경망 층
        self.lstm = nn.LSTM(input_size=combined_dim, hidden_size=hidden_size, batch_first=True)
        
        # 5. Actor Head (Continuous Actions: 평균(mu) 및 표준편차 성분)
        self.actor_mu = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() # 출력을 [-1, 1] 범위로 제한하여 이동/크기 배율 제어
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # 확정 상태 (Terminal Action) 판단 - 멈춤/선언 여부 결정 확률
        self.actor_terminal = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 6. Critic Head (해당 상태 변화의 가치 (Value Estimate) 평가)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, local_patch, global_view, coords, hidden_state=None):
        """
        입력 파라미터 (Batch, Sequence 지원형):
            local_patch: (B, Seq, C, H, W) 
            global_view: (B, Seq, C, H, W)
            coords: (B, Seq, 4) - 현재 진행 중인 윈도우 [cx, cy, w, h] 값
            hidden_state: (h_0, c_0) Tensor Tuple
        """
        batch_size, seq_len = local_patch.shape[:2]
        
        # CNN 특성 추출을 위해 시계열 및 배치 축 임시 평탄화
        local_patch_flat = local_patch.view(batch_size * seq_len, *local_patch.shape[2:])
        global_view_flat = global_view.view(batch_size * seq_len, *global_view.shape[2:])
        coords_flat = coords.view(batch_size * seq_len, -1)
        
        local_feat = self.local_extractor(local_patch_flat)
        global_feat = self.global_extractor(global_view_flat)
        coord_feat = self.coord_embed(coords_flat)
        
        # State 통합
        combined = torch.cat([local_feat, global_feat, coord_feat], dim=-1) 
        combined = combined.view(batch_size, seq_len, -1) # -> (B, Seq, Dim)
        
        # LSTM 상태 전이
        lstm_out, new_hidden = self.lstm(combined, hidden_state)
        
        # 각 액션에 대한 모델 결과 연산
        mu = self.actor_mu(lstm_out)
        logstd = self.actor_logstd.expand_as(mu)
        terminal_prob = self.actor_terminal(lstm_out)
        value = self.critic(lstm_out)
        
        return mu, logstd, terminal_prob, value, new_hidden

if __name__ == '__main__':
    # 모델 텐서 테스트
    agent = RPPOAgent()
    dummy_local = torch.randn(2, 5, 1, 224, 224) # Batch=2, Seq=5
    dummy_global = torch.randn(2, 5, 1, 256, 256)
    dummy_coords = torch.rand(2, 5, 4)
    
    mu, std, term, val, h_state = agent(dummy_local, dummy_global, dummy_coords)
    print(f"Action Mu Shape: {mu.shape}")
    print(f"Value Shape: {val.shape}")
