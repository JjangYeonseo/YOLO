from ultralytics import YOLO
import cv2

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')

# 분석할 이미지 경로
image_path = 'test.jpg'

# 이미지에 대해 예측 수행
results = model(image_path)

# 차량 클래스를 나타내는 클래스 ID 리스트 (COCO 데이터셋 기준)
vehicle_classes = [2, 3, 5, 7]

# 차량을 제외한 객체만 남기기
for r in results:
    boxes = r.boxes.xyxy  # Bounding box 좌표
    classes = r.boxes.cls  # 클래스 ID
    confidences = r.boxes.conf  # 신뢰도 점수

    # 필터링된 결과를 위한 빈 리스트
    filtered_boxes = []
    filtered_classes = []
    filtered_confidences = []

    # 차량이 아닌 객체 필터링
    for i, cls in enumerate(classes):
        if cls not in vehicle_classes:
            filtered_boxes.append(boxes[i])
            filtered_classes.append(cls)
            filtered_confidences.append(confidences[i])

    # 원본 이미지를 로드
    image = cv2.imread(image_path)

    # 필터링된 결과를 이미지에 그리기
    for box, cls, conf in zip(filtered_boxes, filtered_classes, filtered_confidences):
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]}: {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 이미지 저장
    cv2.imwrite('result_no_vehicles.jpg', image)

print("차량을 제외한 분석이 완료되었고 결과 이미지가 저장되었습니다.")
