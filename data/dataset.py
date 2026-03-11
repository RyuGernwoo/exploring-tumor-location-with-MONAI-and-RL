import os
import pandas as pd
import numpy as np
from PIL import Image
from monai.data import Dataset
import monai.transforms as mt
import torch

class CXR8Dataset(Dataset):
    """
    MONAI Dataset을 상속한 CXR8 이미지, BBox, 라벨 로드 모듈.
    Data_Entry_2017_v2020.csv 및 BBox_List_2017.csv를 DataFrame으로 합쳐서 
    경로 매핑과 리사이즈 처리를 원활하게 연결해줍니다.
    """
    def __init__(self, root_dir, bbox_csv, entry_csv, transform=None, use_bbox_only=False):
        self.root_dir = root_dir
        self.transform = transform
        
        # 14개의 NIH Multi-label 
        self.classes = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        
        # 1. Entry CSV 로드
        self.entry_df = pd.read_csv(entry_csv)
        
        # 2. BBox CSV 로드 (헤더 파싱 문제 회피를 위해 수동으로 컬럼 지정)
        self.bbox_df = pd.read_csv(bbox_csv, skiprows=1, header=None)
        self.bbox_df = self.bbox_df.iloc[:, :6] # 첫 6개 컬럼만 사용
        self.bbox_df.columns = ['Image Index', 'Finding Label', 'x', 'y', 'w', 'h']

        # 3. 데이터 필터링 (Phase 0, 1을 위한 BBox 존재 데이터만 유지 옵션)
        if use_bbox_only:
            valid_images = self.bbox_df['Image Index'].unique()
            self.entry_df = self.entry_df[self.entry_df['Image Index'].isin(valid_images)].reset_index(drop=True)
            
        self.data_entries = self.entry_df.to_dict('records')

    def __len__(self):
        return len(self.data_entries)

    def _parse_labels(self, label_str):
        """ 'Atelectasis|Cardiomegaly' 문자열을 Multi-hot 형태의 Vector로 변환합니다. """
        labels = np.zeros(len(self.classes), dtype=np.float32)
        if pd.isna(label_str) or label_str == 'No Finding':
            return labels
        for label in label_str.split('|'):
            if label in self.classes:
                labels[self.classes.index(label)] = 1.0
        return labels

    def _get_bbox(self, image_index):
        """ 
        특정 이미지의 BBox 파싱 로직.
        다수의 BBox가 존재할 수 있으나 본 환경에서는 에이전트 탐색을 위해 가장 첫 번째 BBox로 간략화합니다.
        반환 포맷: [x, y, w, h] (원 x_min, y_min 좌표와 너비/높이)
        """
        bbox_rows = self.bbox_df[self.bbox_df['Image Index'] == image_index]
        if len(bbox_rows) > 0:
            row = bbox_rows.iloc[0]
            return np.array([row['x'], row['y'], row['w'], row['h']], dtype=np.float32)
        return np.array([0, 0, 0, 0], dtype=np.float32)

    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        image_name = entry['Image Index']
        label_str = entry['Finding Labels']
        
        # 이미지 전체 경로 산출 (데이터셋 구조에 따라 images/ 하위 또는 root 에 위치할 수 있음)
        img_path = os.path.join(self.root_dir, 'images', image_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.root_dir, image_name)
            
        try:
            # 원본 해상도를 유지한 채 Grayscale 1채널 모드로 이미지 로드
            img = Image.open(img_path).convert('L')
            img_array = np.array(img, dtype=np.float32)            
            
            # MONAI 처리를 위한 (C, H, W) 포맷 적용
            if len(img_array.shape) == 2:
                img_array = np.expand_dims(img_array, axis=0)
        except Exception as e:
            # IO 로드 실패 시 빈 이미지 텐서 제공
            img_array = np.zeros((1, 1024, 1024), dtype=np.float32)
            
        labels = self._parse_labels(label_str)
        bbox = self._get_bbox(image_name)

        data_dict = {
            "image": img_array,
            "label": labels,
            "bbox": bbox,
            "image_name": image_name
        }
        
        # MONAI Transform 변환 및 텐서 캐스팅
        if self.transform:
            data_dict = self.transform(data_dict)
            
        return data_dict

def get_cxr_transforms():
    """
    RL 모델 및 Classifier에 전달하기 전 numpy Array를 Torch Tensor로 변환하고
    스케일 정규화 (Intensity Scale) 묶음을 제공합니다.
    """
    transforms = mt.Compose([
        mt.ScaleIntensityd(keys=["image"]), # 픽셀 값을 0~1 사이로 정규화
        mt.ToTensord(keys=["image", "label", "bbox"], dtype=torch.float32)
    ])
    return transforms
