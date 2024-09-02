import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')  # 'yolov8n.pt' 또는 다른 모델을 사용할 수 있습니다.

# 비디오 파일 로드
video_path = 'yolovid.mp4'  # 분석할 비디오 파일 경로
cap = cv2.VideoCapture(video_path)

# 결과 비디오 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video_5.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# 클릭된 객체 정보 저장
clicked_box = None
clicked_hsv_color = None

# 마우스 클릭 이벤트 처리 함수
def click_event(event, x, y, flags, param):
    global clicked_box, clicked_hsv_color
    
    if event == cv2.EVENT_LBUTTONDOWN:
        for box in current_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked_box = box
                # 클릭한 객체의 중앙 위치에서 색상 추출
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                clicked_hsv_color = hsv_frame[center_y, center_x]
                print(f"클릭된 객체의 색상: {clicked_hsv_color}")
                break

# 비디오 프레임 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv8로 객체 탐지 수행
    results = model(frame)
    current_boxes = [box for result in results for box in result.boxes if box.cls == 0]  # 사람만 추출
    
    # 현재 프레임의 HSV 변환
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 마우스 이벤트 처리 연결
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', click_event)
    
    if clicked_box is not None:
        for box in current_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 현재 객체의 중앙 색상 추출
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            current_hsv_color = hsv_frame[center_y, center_x]
            
            # 클릭된 객체와 색상 비교
            color_difference = np.linalg.norm(clicked_hsv_color - current_hsv_color)
            print(f"현재 객체 색상: {current_hsv_color}, 차이: {color_difference}")

            if color_difference < 40:  # 색상 차이 기준을 40으로 조정
                # 유사한 객체 비식별화 (모자이크 처리)
                person_roi = frame[y1:y2, x1:x2]
                person_roi = cv2.resize(person_roi, (10, 10), interpolation=cv2.INTER_LINEAR)
                person_roi = cv2.resize(person_roi, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                frame[y1:y2, x1:x2] = person_roi
    
    # 결과 프레임 화면에 표시 및 비디오 파일에 저장
    cv2.imshow('frame', frame)
    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
