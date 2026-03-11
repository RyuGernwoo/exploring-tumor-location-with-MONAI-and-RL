import numpy as np

def calculate_iou(boxA, boxB):
    """
    두 Bounding Box [x_min, y_min, x_max, y_max] 구조 간의 교집합 비율 (IoU) 연산
    """
    # 좌표 정렬 교집합 계산
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 겹치지 않는 경우 교집합 면적 0
    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Zero Division 회피
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def cxcywh_to_xyxy(cxcywh_box):
    """
    [Center X, Center Y, Width, Height] 형식을 [X_min, Y_min, X_max, Y_max] 구조로 변경합니다.
    """
    cx, cy, w, h = cxcywh_box
    x_min = cx - (w / 2)
    y_min = cy - (h / 2)
    x_max = cx + (w / 2)
    y_max = cy + (h / 2)
    return [x_min, y_min, x_max, y_max]
