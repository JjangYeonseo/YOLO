from ultralytics import YOLO
import cv2
# ultralytics는 YOLO (You Only Look Once) 객체 검출 모델을 위한 라이브러리로 실시간 객체 검출, 분류, 세그멘테이션 등의 작업에 사용
# cv2(opencv) 광범위한 컴퓨터 비전 작업을 위한 오픈소스 라이브러리로 이미지 및 비디오 처리, 분석을 위한 다양한 함수 제공
# ultralytics로 객체를 검출한 후 cv2를 사용하여 검출된 객체를 이미지에 표시하거나 추가적인 이미지 처리 수행

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')  # 'n'은 nano 버전, 다른 버전 사용하려면 변경하면 됨
# n은 YOLOv8의 가장 작고 빠른 버전이고 '.pt'는 PyTorch 모델 파일 확장자
# 해당 모델은 사전 훈련된 가중치를 가지고 있어, 별도의 훈련 없이 바로 객체 검출에 사용 가능

# 분석할 이미지 경로
image_path = 'test.jpg'
# YOLO.py 파일과 해당 이미지 파일이 같은 경로에 있으므로 경로를 직접 입력할 필요 없이 파일명만 쓰면 자동으로 파일 찾음

# 이미지에 대해 예측 수행
results = model(image_path)
# yolov8n 모델에서 해당 이미지를 분석하여 추론을 진행하고 객체 검출하여 results에 결과 저장
# 검출된 객체의 목록, 객체의 bbox 좌표, 객체의 클래스, 각 검출의 신뢰도 점수가 저장됨

# 결과 이미지 저장
for r in results:
    im_array = r.plot()  # 박스가 그려진 이미지 배열
    im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result.jpg', im) 
# r.plot()은 검출된 객체에 bbox)를 그린 이미지 생성하여 넘파이 배열로 반환 (해당 배열은 RGB 형식)
# cv2.cvtColor는 이미지의 색상 형식을 변환하는 함수
# cv2.COLOR_RGB2BGR은 RGB 형식을 BGR 형식으로 변환
# OpenCV는 기본적으로 BGR 형식을 사용하므로, 이 변환이 필요
# cv2.imwrite는 이미지를 파일로 저장하는 함수
# 이 이미지에는 검출된 객체에 대한 경계 상자가 그려져 있음
# 이 코드는 루프 안에 있지만, 단일 이미지 처리의 경우 한 번만 실행딤
# 여러 이미지를 처리하는 경우, 파일 이름을 동적으로 생성해야 각 결과가 별도로 저장됨
# RGB는 대부분의 이미지 포맷과 디스플레이 장치에서 표준으로 사용됨
# BGR은 OpenCV 라이브러리의 내부 표현 방식
# OpenCV로 이미지를 처리할 때, 다른 라이브러리나 디스플레이와의 호환을 위해 BGR에서 RGB로, 또는 그 반대로 변환이 필요한 경우 많음
# 같은 숫자 값이라도 RGB와 BGR에서는 다른 색상나타냄

print("분석이 완료되었고 결과 이미지가 저장되었습니다.")

#파이썬 최신 버전을 사용했을 때는 객체 검출이 제대로 이루어지지 않았는데, 버전을 3.8.19로 바꾸니 제대로 되는 듯함 (확실히는 모르겠음..)

for r in results:
    print(f"검출된 객체 수: {len(r.boxes)}")

    # 결과 이미지 저장
    im_array = r.plot()  # 박스가 그려진 이미지 배열
    im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite('resultlen.jpg', im)

print("분석이 완료되었고 결과 이미지가 저장되었습니다.")
