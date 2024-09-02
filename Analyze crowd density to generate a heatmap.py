import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO("yolov8n.pt")

# 동영상 파일 로드
cap = cv2.VideoCapture("yolovid.mp4")

# 첫 번째 프레임의 크기와 FPS 얻기
ret, frame = cap.read()
height, width = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)

# 결과 저장을 위한 VideoWriter 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 ('XVID', 'mp4v', 'X264' 등 사용 가능)
out = cv2.VideoWriter('output_video_6.mp4', fourcc, fps, (width, height))

# 히트맵을 저장할 빈 이미지 생성
heatmap = np.zeros((height, width), dtype=np.float32)

# 동영상 다시 처음부터 읽기
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv8로 객체 감지
    results = model(frame)
    
    # YOLOv8 결과에서 bounding box 가져오기
    for result in results[0].boxes:  # 'results[0]'은 프레임 당 결과
        x1, y1, x2, y2 = result.xyxy[0]  # bbox 좌표 추출
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # 정수형 변환

        # 감지된 영역을 히트맵에 추가
        heatmap[y1:y2, x1:x2] += 1

    # 히트맵 이미지로 변환 및 크기 조정
    heatmap_img = cv2.applyColorMap(cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1), cv2.COLORMAP_JET)
    heatmap_img = cv2.resize(heatmap_img, (width, height))

    # 최종 히트맵을 원본 프레임에 오버레이
    overlay = cv2.addWeighted(heatmap_img, 0.6, frame, 0.4, 0)

    # 결과를 동영상 파일로 저장
    out.write(overlay)

    # 화면에 출력
    cv2.imshow("Heatmap", overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 작업이 끝나면 리소스 해제
cap.release()
out.release()  # VideoWriter 해제
cv2.destroyAllWindows()
