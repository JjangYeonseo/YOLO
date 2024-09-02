from ultralytics import YOLO
import cv2
from collections import Counter

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')

# 분석할 이미지 경로
image_path = 'test.jpg'

# 이미지에 대해 예측 수행
results = model(image_path)

# 클래스별 개수를 저장할 Counter 객체
class_counter = Counter()

# 검출된 객체의 개수 출력
for r in results:
    # 각 검출된 객체의 클래스 정보를 추출
    classes = r.boxes.cls  # 각 객체의 클래스 리스트
    class_counter.update(classes)  # 클래스별로 개수 세기

    # 결과 이미지 저장
    im_array = r.plot()  # 박스가 그려진 이미지 배열
    im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result.jpg', im)

# 클래스별 개수 출력
for cls, count in class_counter.items():
    print(f"클래스 {cls}: {count}개")

print("분석이 완료되었고 결과 이미지가 저장되었습니다.")
