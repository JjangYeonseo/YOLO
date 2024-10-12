import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드 (기본 모델 사용 또는 사용자 정의 모델 경로 입력)
model = YOLO('yolov8n.pt')  # yolov8n.pt는 YOLOv8 기본 모델

# 영상에서 객체 탐지 및 자율 주행 시뮬레이션을 수행하는 함수
def autonomous_cart_simulation(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # 영상의 가로, 세로 크기 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 결과를 저장할 VideoWriter 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 코덱 설정
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8로 현재 프레임에서 객체 탐지 수행
        results = model(frame)
        obstacles = []

        # 객체 탐지 결과에서 장애물 좌표 저장
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = box.cls[0]
                name = model.names[int(class_id)]

                # 관심 있는 객체 필터링 (예: 사람, 카트, 선반)
                if name in ['person', 'cart', 'shelf']:  # 필요에 따라 객체 이름 수정 가능
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    obstacles.append((center_x, center_y, name))

                    # 탐지된 객체를 프레임에 표시 (바운딩 박스 및 레이블)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'{name} {confidence:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # 장애물에 따른 자율 주행 회피 로직 (간단한 좌/우 회피)
        if obstacles:
            steer_direction = 0  # 0: 직진, -1: 왼쪽, 1: 오른쪽

            for (x, y, obj) in obstacles:
                if x < width // 2:
                    steer_direction = 1  # 장애물이 왼쪽에 있으므로 오른쪽으로 회피
                else:
                    steer_direction = -1  # 장애물이 오른쪽에 있으므로 왼쪽으로 회피

            # 회피 방향 출력
            if steer_direction == 1:
                print("장애물을 피해 오른쪽으로 회전합니다.")
            elif steer_direction == -1:
                print("장애물을 피해 왼쪽으로 회전합니다.")
            else:
                print("직진합니다.")
        else:
            print("장애물이 없어 직진합니다.")

        # 탐지 결과가 표시된 프레임을 저장
        out.write(frame)

        # 탐지 결과를 화면에 표시
        cv2.imshow('Autonomous Cart Simulation', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 사용 예시
video_path = 'mart1.mp4'  # 처리할 영상 경로
output_path = 'output_video_with_detections_final.mp4'  # 결과 영상 저장 경로
autonomous_cart_simulation(video_path, output_path)
