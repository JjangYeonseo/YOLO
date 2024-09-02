import cv2
from ultralytics import YOLO

# YOLOv8n 모델 로드
model = YOLO('yolov8n.pt') 

# 비디오 파일 경로
video_path = 'yolovid.mp4'  # 동영상 파일 경로를 지정

# 비디오 캡처 열기
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# 비디오 저장을 위한 설정
# 원본 비디오의 프레임 크기 및 FPS 가져오기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# VideoWriter 객체 생성
output_path = 'output_vid.mp4'  # 결과 비디오를 저장할 경로
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 (예: mp4v for .mp4)
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("Video has ended or cannot read the frame.")
        break

    # YOLOv8n 모델을 사용하여 객체 탐지
    results = model(frame)

    # 탐지 결과를 프레임에 그리기
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls_id)]
            confidence = conf

            # 경계 상자 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 프레임을 화면에 표시
    cv2.imshow('YOLOv8n Detection', frame)

    # 결과 프레임을 비디오 파일에 저장
    out.write(frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 및 저장 해제, 모든 창 닫기
cap.release()
out.release()
cv2.destroyAllWindows()
