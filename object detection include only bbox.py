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
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_path = 'output_video_1.mp4'  # 결과 비디오를 저장할 경로
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("Video has ended or cannot read the frame.")
        break

    # YOLOv8n 모델을 사용하여 객체 탐지
    results = model(frame)

    # 'person' 클래스 ID (COCO dataset에서 'person'은 클래스 ID 0)
    person_class_id = 0

    # 탐지 결과를 필터링하여 'person' 클래스만 남기기
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for box, cls_id in zip(boxes, class_ids):
            if int(cls_id) == person_class_id:
                x1, y1, x2, y2 = map(int, box)

                # 경계 상자 그리기 (Bounding Box만 표시)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 결과 프레임을 화면에 표시
    cv2.imshow('YOLOv8n Person Detection', frame)

    # 결과 프레임을 비디오 파일에 저장
    out.write(frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 및 저장 해제, 모든 창 닫기
cap.release()
out.release()
cv2.destroyAllWindows()
