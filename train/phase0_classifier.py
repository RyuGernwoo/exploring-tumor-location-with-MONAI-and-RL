import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.functional as TF

# 프로젝트 루트 경로 추가 (모듈 import 위함)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import CXR8Dataset, get_cxr_transforms
from models.classifier import CXRClassifier

def crop_and_resize(image_tensor, bbox, size=(224, 224)):
    """
    배치별로 (x, y, w, h) BBox를 기준으로 이미지를 Crop한 후 
    정밀 진단 모델의 입력 사이즈(예: 224x224) 패치(Patch)로 Resize 합니다.
    이 과정을 통해 향후 성능 평가시(Reward) '에이전트가 BBox를 잘 탐색했을 때의 이상적인 모델 시야'를 
    사전 학습하게 됩니다.
    """
    cropped_batch = []
    for i in range(image_tensor.size(0)):
        img = image_tensor[i]
        x, y, w, h = bbox[i]
        
        if w.item() == 0 or h.item() == 0:
            # BBox가 비정상적인 경우 (No Finding 또는 파싱 실패) 원본 전체를 Resize하여 사용
            crop = img
        else:
            x, y, w, h = int(x.item()), int(y.item()), int(w.item()), int(h.item())
            _, H, W = img.shape
            
            # Crop 바운더리 보호 처리
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))
            
            crop = TF.crop(img, top=y, left=x, height=h, width=w)
            
        resized_crop = TF.resize(crop, size, antialias=True)
        cropped_batch.append(resized_crop)
        
    return torch.stack(cropped_batch)

def train_phase0(epochs=5, batch_size=16, lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Starting Phase 0 Training on device: {device}")
    
    # 1. BBox가 존재하는 데이터만 활용하여 Dataset 준비
    # 경로는 tutorials/CXR8 기준 (tumor_aing 프로젝트 외부의 공용 경로를 참조)
    root_dir = "C:/Users/qesad/Desktop/tutorials/CXR8"
    bbox_csv = os.path.join(root_dir, "BBox_List_2017.csv")
    entry_csv = os.path.join(root_dir, "Data_Entry_2017_v2020.csv")
    
    transforms = get_cxr_transforms()
    dataset = CXR8Dataset(
        root_dir=root_dir, 
        bbox_csv=bbox_csv, 
        entry_csv=entry_csv, 
        transform=transforms, 
        use_bbox_only=True
    )
    
    print(f"Loaded {len(dataset)} BBox labeled images for Supervised Pre-training.")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 2. 분류 모델(DenseNet121 기반) 생성
    model = CXRClassifier(in_channels=1, out_classes=14).to(device)
    
    # 3. 손실 함수 및 옵티마이저 (Multi-label 분류이므로 BCELoss 사용하며, 내부 Sigmoid는 모델에 결합되어 있음)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. 학습 루프 
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for batch in loop:
            # Transform을 통해 [B, 1, H, W] 텐서로 넘어옴
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"] # [B, 4]
            
            # Ground Truth 병변 영역으로 강제 Crop 후 224x224 사이즈로 변형 수행
            patches = crop_and_resize(images, bboxes, size=(224, 224)).to(device)
            
            optimizer.zero_grad()
            outputs = model(patches) # 출력: 0~1 (Sigmoid Probability, Shape: [B, 14])
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
            
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {epoch_loss:.4f}")
        
    # 5. 가중치 저장 (이후 Phase 2, 3 에이전트 탐색 과정에서 Freeze하여 보상 산정에 사용됨)
    save_dir = "C:/Users/qesad/Desktop/tutorials/tumor_aing/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "phase0_classifier.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Phase 0 Pre-training finished. Weights saved at {save_path}")

if __name__ == "__main__":
    train_phase0()
