import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')

# 이미지 경로 및 결과 저장 경로 설정
image_path = r'처리할 이미지 경로' 
output_path = r'결과를 저장할 경로'

# 라벨링된 좌표 정보 (x_center, y_center, width, height)
coordinates = [
    (0.724404, 0.720395, 0.339411, 0.297932),  # desk
    (0.421809, 0.650376, 0.223703, 0.417293),  # chair
    (0.272791, 0.498590, 0.193548, 0.615602)   # person
]

# 각 객체에 대한 신뢰도 설정
confidences = [0.7, 0.85, 0.8]  # desk, chair, person에 대한 신뢰도

# 이미지 읽기
image = cv2.imread(image_path)
height, width, _ = image.shape  # 이미지 크기 가져오기

# 좌표 정보를 사용하여 바운딩 박스 그리기
for idx, (x_center, y_center, w, h) in enumerate(coordinates):
    # 상대 좌표를 절대 좌표로 변환
    x1 = int((x_center - w / 2) * width)
    y1 = int((y_center - h / 2) * height)
    x2 = int((x_center + w / 2) * width)
    y2 = int((y_center + h / 2) * height)
    
    # 라벨에 따라 색상 설정
    color = (255, 0, 0) if idx == 0 else (0, 255, 0) if idx == 1 else (0, 0, 255)
    label = ['desk', 'chair', 'person'][idx]
    
    # 바운딩 박스 그리기
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, f'{label} {confidences[idx]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# YOLO 모델을 사용하여 객체 탐지
results = model(image)

# 결과 이미지에 YOLO 탐지 결과 그리기
if results:  # 결과가 있는 경우에만 처리
    detections = results[0].boxes  # 첫 번째 요소에서 박스 정보 가져오기
    for idx, box in enumerate(detections):  # 각 박스에 대해 반복
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌표를 정수로 변환
        cls = int(box.cls[0])  # 클래스 인덱스 가져오기
        label = model.names[cls]  # 클래스 이름 가져오기
        confidence = box.conf[0]  # YOLO 모델에서 제공하는 신뢰도 가져오기
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 탐지된 객체에 대해 노란색 박스
        cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

# 결과 이미지 저장
cv2.imwrite(output_path, image)

print(f"결과 이미지가 {output_path}에 저장되었습니다.")
