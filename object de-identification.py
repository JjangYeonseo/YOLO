import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')  # 또는 다른 사전 학습된 모델 경로를 사용

# 비디오 파일 로드
video_path = 'yolovid.mp4'  # 분석할 비디오 파일 경로
cap = cv2.VideoCapture(video_path)

# 결과 비디오 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video_2.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv8로 객체 탐지 수행
    results = model(frame)
    
    for result in results:
        # 사람 객체만 필터링
        for box in result.boxes:
            if box.cls == 0:  # 0은 COCO dataset에서 'person' 클래스에 해당
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 특정 성질을 가진 사람을 필터링하는 조건 예시
                if box.conf > 0.5:  # 예: 신뢰도가 0.5 이상인 사람
                    # 비식별화 처리 (모자이크)
                    person = frame[y1:y2, x1:x2]
                    person = cv2.resize(person, (10, 10), interpolation=cv2.INTER_LINEAR)
                    person = cv2.resize(person, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                    frame[y1:y2, x1:x2] = person
    
    # 결과 프레임 저장
    out.write(frame)
    
    # 결과 프레임을 화면에 표시 (선택 사항)
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
