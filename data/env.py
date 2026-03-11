import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import torch
from utils.metrics import calculate_iou, cxcywh_to_xyxy

class CXRExplorationEnv(gym.Env):
    """
    Gymnasium 환경 인터페이스를 준수하는 CXR Bounding Box 탐색 환경.
    Global View, Local Patch의 이원화된 정보와 동적인 연속 보상을 제공합니다.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, data_list, max_steps=10, global_res=(256, 256), patch_res=(224, 224)):
        super().__init__()
        
        self.data_list = data_list
        self.max_steps = max_steps
        self.global_res = global_res
        self.patch_res = patch_res
        
        # PPO 연속 제어: [dx, dy, dw, dh, terminal_signal]
        # dx, dy: 중심점 이동 범위 [-1, 1]
        # dw, dh: 크기 변경 배율 범위 [-1, 1]
        # terminal_signal: 0.5 이상일시 종결 확정 선언
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0]), 
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            "local_patch": spaces.Box(low=0, high=255, shape=(1, *patch_res), dtype=np.uint8),
            "global_view": spaces.Box(low=0, high=255, shape=(1, *global_res), dtype=np.uint8),
            "coords": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32) 
        })

        self.current_step = 0
        self.image = None
        self.bbox_gt = None 
        self.current_window = [0.5, 0.5, 1.0, 1.0] # cx, cy, w, h
        
    def _get_obs(self):
        # 1. Global view 특징 (저해상도 전체 컨텍스트)
        global_view = cv2.resize(self.image, self.global_res)
        global_view = np.expand_dims(global_view, axis=0) # (1, H, W) 포맷
        
        # 2. Local patch 특징 (동적 크기의 정밀 윈도우)
        ih, iw = self.image.shape
        cx, cy, w, h = self.current_window
        
        x_min = int(max(0, (cx - w/2) * iw))
        x_max = int(min(iw, (cx + w/2) * iw))
        y_min = int(max(0, (cy - h/2) * ih))
        y_max = int(min(ih, (cy + h/2) * ih))
        
        if x_max <= x_min or y_max <= y_min:
            crop = np.zeros(self.patch_res, dtype=np.uint8)
        else:
            crop = self.image[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                crop = np.zeros(self.patch_res, dtype=np.uint8)
            else:
                crop = cv2.resize(crop, self.patch_res)
            
        local_patch = np.expand_dims(crop, axis=0)
        
        return {
            "local_patch": local_patch,
            "global_view": global_view,
            "coords": np.array(self.current_window, dtype=np.float32)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # TODO: self.data_list에서 실제 이미지 로드 연동
        # 현재는 Dummy(가짜 데이터)로 형태만 정의합니다.
        self.image = np.zeros((1024, 1024), dtype=np.uint8) 
        self.bbox_gt = [0.4, 0.4, 0.6, 0.6] # 임시 정답 박스 좌표
        
        # 이미지 전체를 포괄하는 초기 윈도우 셋팅 (1.0 비율)
        self.current_window = [0.5, 0.5, 1.0, 1.0] 
        self.current_step = 0
        self.last_iou = calculate_iou(self.bbox_gt, cxcywh_to_xyxy(self.current_window))
        
        return self._get_obs(), {}

    def step(self, action):
        dx, dy, dw, dh, terminal = action
        
        cx, cy, w, h = self.current_window
        scale_pos_stride = 0.2  # 최대 중심점 이동 비율 제어
        scale_size_stride = 0.2 # 최대 크기 변환 비율 제어
        
        # [-1, 1] 액션 상에서의 실제 윈도우 좌표 배율 연산
        cx = np.clip(cx + dx * scale_pos_stride, 0.0, 1.0)
        cy = np.clip(cy + dy * scale_pos_stride, 0.0, 1.0)
        w = np.clip(w * (1.0 + dw * scale_size_stride), 0.1, 1.0)
        h = np.clip(h * (1.0 + dh * scale_size_stride), 0.1, 1.0)
        
        self.current_window = [cx, cy, w, h]
        self.current_step += 1
        
        terminated = False
        truncated = False
        reward = 0.0
        
        # $\Delta$ IoU에 근거한 밀도 있는(Dense) 탐색 보상 지급
        curr_xyxy = cxcywh_to_xyxy(self.current_window)
        current_iou = calculate_iou(self.bbox_gt, curr_xyxy)
        iou_delta = current_iou - self.last_iou
        self.last_iou = current_iou
        
        # 보상 체계 연산
        reward += (iou_delta * 10.0) 
        reward -= 0.05  # 매 스텝마다 탐색 지연 시간(Efficiency) 삭감 패널티
        
        # 확정 여부 타진 시 (0.5 이상의 시그널)
        if terminal > 0.5:
            terminated = True
            # 확정 시 보너스 또는 실제 Classifier Joint 보상 연결부
            if current_iou > 0.5:
                reward += 5.0 
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}
