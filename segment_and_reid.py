import cv2
import torch
import numpy as np
import time
from torchvision import transforms
from ultralytics import YOLO
from torchreid.reid.models import build_model
from torchreid.reid.utils import load_pretrained_weights
from scipy.spatial.distance import cosine

# Configuration
SIMILARITY_THRESHOLD = 0.3  # 값 ↓ → 더 엄격한 매칭
FEATURE_UPDATE_ALPHA = 0.7  # 0.5~0.9 권장 (높을수록 기존 특징 유지) 조도 변화가 심한 환경에서는 0.5 정도로 설정
MAX_INACTIVE_FRAMES = 300    # 해당 프레임 동안 미검시 시 ID 삭제

# 모델 초기화
model = YOLO("yolo11n-seg.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reid_model = build_model(name='osnet_x1_0', num_classes=1000, pretrained=True)
load_pretrained_weights(reid_model, '~/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth')
reid_model = reid_model.to(device).eval()

# 전처리 파이프라인
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ID 관리 시스템
class IDManager:
    def __init__(self):
        self.next_id = 1
        self.known_ids = {}  # {id: {'feature': np.array, 'last_seen': int}}
        self.frame_count = 0

    def update_features(self, id, new_feature):
        """기존 특징 벡터를 가중 평균으로 갱신"""
        old_feature = self.known_ids[id]['feature']
        self.known_ids[id]['feature'] = FEATURE_UPDATE_ALPHA * old_feature + (1-FEATURE_UPDATE_ALPHA)*new_feature
        self.known_ids[id]['last_seen'] = self.frame_count

    def add_new_id(self, feature):
        """새 ID 생성"""
        self.known_ids[self.next_id] = {
            'feature': feature,
            'last_seen': self.frame_count
        }
        self.next_id += 1

    def cleanup_old_ids(self):
        """비활성 ID 제거"""
        to_delete = [id for id, data in self.known_ids.items() 
                    if (self.frame_count - data['last_seen']) > MAX_INACTIVE_FRAMES]
        for id in to_delete:
            del self.known_ids[id]

# Initialize systems
id_manager = IDManager()
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret: break
    
    # YOLO 추적 (Retina-Mask 활성화)
    results = model.track(
        frame, 
        classes=[0],          # 사람만 검출
        retina_masks=True,   # 고품질 마스크
        persist=True,         # 프레임 간 ID 유지
        verbose=False         # 출력 비활성화
    )
    
    id_manager.frame_count += 1
    id_manager.cleanup_old_ids()

    # 객체가 없는 경우 건너뛰기
    if results[0].boxes is None or results[0].masks is None:
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) == ord('q'): break
        continue

    # 마스크 & BBox 처리
    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    for mask, box in zip(masks, boxes):
        x1, y1, x2, y2 = map(int, box)
        
        # 마스크 영역 추출 (BBox 크기로 크롭)
        cropped_mask = mask[y1:y2, x1:x2]
        if cropped_mask.size == 0: continue  # 유효하지 않은 영역 건너뛰기
        
        # 마스크 적용 영역 추출
        masked_region = frame[y1:y2, x1:x2] * cropped_mask[..., None]
        if masked_region.size == 0: continue
        
        # 특징 벡터 추출
        with torch.no_grad():
            obj_tensor = preprocess(masked_region).unsqueeze(0).to(device)
            feature = reid_model(obj_tensor).cpu().numpy().flatten()

        # ID 매칭 로직
        best_id = None
        min_distance = float('inf')
        
        # 기존 ID와 비교
        for id, data in id_manager.known_ids.items():
            distance = cosine(feature, data['feature'])
            if distance < SIMILARITY_THRESHOLD and distance < min_distance:
                min_distance = distance
                best_id = id
                
        # 매칭된 ID 처리
        if best_id is not None:
            id_manager.update_features(best_id, feature)
            assigned_id = best_id
        else:
            assigned_id = id_manager.next_id
            id_manager.add_new_id(feature)
            
        # 시각화
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID:{assigned_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    # 결과 출력
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
