from ultralytics import YOLO

#cpu만으로 학습 오래 걸리면 gpu로 돌리는 방식 추가 필요함

# 기존 COCO 모델을 불러옴 (yolov8n.pt 또는 yolov8s.pt, yolov8m.pt 등)
model = YOLO('yolov8n.pt')  # 필요한 경우 다른 모델을 선택하세요

# 새로운 데이터셋으로 학습 (epochs는 학습 반복 횟수, data.yaml 파일 경로 제공)
model.train(data='C:\\Users\\jys20\\Desktop\\cartimage\\data.yaml', epochs=100, imgsz=640)

# 새로운 데이터셋으로 학습 (epochs는 학습 반복 횟수, data.yaml 파일 경로 제공)
model.train(data='C:\\Users\\jys20\\Desktop\\shelvesimage\\data.yaml', epochs=100, imgsz=640)

# 새로운 데이터셋으로 학습 (epochs는 학습 반복 횟수, data.yaml 파일 경로 제공)
model.train(data='C:\\Users\\jys20\\Desktop\\indoorimage\\data.yaml', epochs=100, imgsz=640)

#개별적으로 학습시키지 않고 한 번에 묶어서 처리하는 코드 추가 필요함

# 학습이 끝난 후 모델 저장
model.save('yolov8_custom.pt')

import cv2

# 학습된 YOLO 모델 불러오기
model = YOLO('yolov8_custom.pt')

# 비디오 파일 경로
video_path = 'mart1.mp4'

# 결과 비디오 저장 경로
output_path = 'output_video_with_custom_detections.mp4'

# 비디오 읽기
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# 비디오 속성 가져오기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO로 객체 탐지
    results = model(frame)

    # 탐지 결과 처리 및 표시
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            name = model.names[class_id]

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{name} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # 장애물 회피 처리 (예시로 x좌표 기준 왼쪽, 오른쪽 이동을 처리함)
            center_x = (x1 + x2) // 2
            if center_x < width // 2:
                print(f'{name} detected on the left side, moving right.')
            else:
                print(f'{name} detected on the right side, moving left.')

    # 결과 프레임 저장
    out.write(frame)

    # 결과 프레임 표시
    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()



