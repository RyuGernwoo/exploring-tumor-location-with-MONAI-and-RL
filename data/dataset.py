import os
import pandas as pd
import numpy as np
from PIL import Image
from monai.data import Dataset
import monai.transforms as mt

class CXR8Dataset(Dataset):
    """
    MONAI Dataset을 상속한 CXR8 이미지, BBox, 라벨 로드 모듈.
    Data_Entry_2017_v2020.csv 및 BBox_List_2017.csv를 DataFrame으로 합쳐서 
    경로 매핑과 리사이즈 처리를 원활하게 연결해줍니다.
    """
    def __init__(self, root_dir, bbox_csv, entry_csv, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # 14개의 NIH Multi-label 
        self.classes = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        
        # 향후 실제 CSV 파일 읽기 로직 주입 (Pandas)
        # self.df = pd.read_csv(entry_csv)
        # self.bbox_df = pd.read_csv(bbox_csv)

        # Placeholder 길이 설정
        self.data_entries = np.zeros(10) 

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        # TODO: 실제 이미지 IO (cv2.imread / PIL Image 로드)
        # TODO: Multi-hot 라벨 Array 파싱
        # TODO: BBox 좌표 존재여부에 따른 반환
        
        data_dict = {
            "image": np.zeros((1, 1024, 1024)), # Placeholder
            "label": np.zeros(14),
            "bbox": np.array([0, 0, 0, 0])
        }
        
        # MONAI Transform 적용
        if self.transform:
            data_dict = self.transform(data_dict)
            
        return data_dict

def get_cxr_transforms():
    """
    원본 (보통 1024x1024 해상도) 1채널 이미지를 에이전트와 분산할 수 있도록
    MONAI Dictionary기반의 스케일 정리 포맷팅 Transform 
    """
    transforms = mt.Compose([
        # 실제 적용 시 LoadImaged 사용
        # mt.LoadImaged(keys=["image"]),
        mt.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        mt.ScaleIntensityd(keys=["image"]), # 0~1 정규화 
        # mt.Resized(keys=["image"], spatial_size=(1024, 1024))
    ])
    return transforms
