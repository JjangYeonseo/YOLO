from ultralytics import YOLO
import cv2
import random

# Load the learned YOLO model (load from local path)
model = YOLO('C:\\Users\\jys20\\Desktop\\martimg\\yolov8_custom_withoutside.pt') # Change to your own yolov8_custom.pt path

# Video file path (use local path)
video_path = 'C:\\Users\\jys20\\Desktop\\martimg\\mart1.mp4' # Modify the video file path to suit your environment
output_video_path = 'C:\\Users\\jys20\\Desktop\\martimg\\outputvid_withoutdisplaystand.mp4' # Video path to save

# Read video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video save settings (codec settings and video settings to save)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4 codec settings
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Set color according to class ID (random color generation)
colors = {}

# Define the classes to keep (사람, 카트, 기둥의 클래스 ID를 저장, 정확한 ID는 학습된 모델에 따라 다를 수 있음)
target_classes = [0, 2, 3]  # 학습된 모델의 실제 ID 확인 필요

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection with YOLO
    results = model(frame)

    # Process and display detection results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            name = model.names[class_id]

            # Only display boxes for people, carts, and pillars
            if class_id in target_classes:
                # If the color for that class ID does not exist, create a new one
                if class_id not in colors:
                    colors[class_id] = [random.randint(0, 255) for _ in range(3)]  # Randomly generate RGB colors

                color = colors[class_id]  # Get the color of the class

                # Draw bounding box (use different colors for each class)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{name} {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display detected frames in window (can use cv2.imshow in VSCode)
    cv2.imshow('Detection', frame)

    # Save video (save frame)
    out.write(frame)

    # Press 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Organize video streams and files
cap.release()
out.release()  # End video saving
cv2.destroyAllWindows()
