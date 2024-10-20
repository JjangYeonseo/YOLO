import cv2
import numpy as np
import time
from ultralytics import YOLO
import face_recognition
import argparse
import os
import yt_dlp  # yt-dlp 임포트 추가

class PortraitProtectionSystem:
    def __init__(self, yolo_model='yolov8n.pt', recognition_model='hog'):
        # YOLO 모델과 얼굴 인식 모델 초기화
        self.yolo_model = YOLO(yolo_model)
        self.recognition_model = recognition_model
        self.known_face_encodings = []
        self.known_face_names = []
        self.processing_times = []

    def add_known_face(self, image_path, name):
        # 알려진 얼굴 추가 메서드
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        self.known_face_encodings.append(encoding)
        self.known_face_names.append(name)

    def apply_mosaic(self, image, bbox, level=15):
        # 모자이크 효과 적용 메서드
        x1, y1, x2, y2 = map(int, bbox)
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, ((x2 - x1) // level, (y2 - y1) // level))
        face = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        image[y1:y2, x1:x2] = face
        return image

    def apply_gaussian_blur(self, image, bbox, ksize=(15, 15)):
        # 가우시안 블러 효과 적용 메서드
        x1, y1, x2, y2 = map(int, bbox)
        face = image[y1:y2, x1:x2]
        face = cv2.GaussianBlur(face, ksize, 0)
        image[y1:y2, x1:x2] = face
        return image

    def apply_pixelation(self, image, bbox, blocks=10):
        # 픽셀화 효과 적용 메서드
        x1, y1, x2, y2 = map(int, bbox)
        face = image[y1:y2, x1:x2]
        h, w = face.shape[:2]
        face = cv2.resize(face, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
        image[y1:y2, x1:x2] = face
        return image

    def process_frame(self, frame, method='mosaic'):
        # 프레임 처리 메서드
        start_time = time.time()

        # YOLOv8을 사용한 얼굴 탐지
        results = self.yolo_model(frame, classes=[0])  # 클래스 0은 '사람'으로 가정

        # 얼굴 인식
        face_locations = face_recognition.face_locations(frame, model=self.recognition_model)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            if name == "Unknown":
                # 알려지지 않은 얼굴에 대해 선택된 방법으로 처리
                if method == 'mosaic':
                    frame = self.apply_mosaic(frame, (left, top, right, bottom))
                elif method == 'blur':
                    frame = self.apply_gaussian_blur(frame, (left, top, right, bottom))
                elif method == 'pixelate':
                    frame = self.apply_pixelation(frame, (left, top, right, bottom))
            else:
                # 알려진 얼굴에 대해 박스와 이름 표시
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        end_time = time.time()
        self.processing_times.append(end_time - start_time)

        return frame

    def run(self, video_path, output=None, method='mosaic'):
        # 시스템 실행 메서드
        cap = cv2.VideoCapture(video_path)

        if output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.process_frame(frame, method=method)

            # FPS 정보 표시
            cv2.putText(frame, f"FPS: {1 / np.mean(self.processing_times):.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if output:
                out.write(frame)

            cv2.imshow('Portrait Rights Protection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if output:
            out.release()
        cv2.destroyAllWindows()

        # 성능 지표 출력
        print(f"평균 처리 시간: {np.mean(self.processing_times):.4f} 초")
        print(f"평균 FPS: {1 / np.mean(self.processing_times):.2f}")

def download_youtube_video(youtube_url, output_path):
    # YouTube 비디오 다운로드 메서드
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
        video_file = os.path.join(output_path, ydl.prepare_filename(ydl.extract_info(youtube_url, download=False)))
    
    return video_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="초상권 보호 시스템")
    parser.add_argument("--youtube_url", type=str, required=True, help="유튜브 비디오 URL")
    parser.add_argument("--output", type=str, default="C:\\Users\\jys20\\Desktop\\yolo\\output_video.mp4", help="출력 비디오 파일 경로")
    parser.add_argument("--method", type=str, default="mosaic", choices=['mosaic', 'blur', 'pixelate'], help="얼굴 보호 방법")
    args = parser.parse_args()

    system = PortraitProtectionSystem()

    # 알려진 얼굴 추가 (손흥민 사진 경로 추가)
    system.add_known_face("C:\\Users\\jys20\\Desktop\\yolo\\son.jpg", "Son Heung-min")

    # YouTube 비디오 다운로드 및 처리
    video_path = download_youtube_video(args.youtube_url, "C:\\Users\\jys20\\Desktop\\yolo")
    system.run(video_path, output=args.output, method=args.method)

#출력할 때 명령문 python "C:\\Users\\jys20\\Desktop\\yolo\\sonny.py" --youtube_url "https://www.youtube.com/watch?v=79HpGq2YxVY" --output "C:\\Users\\jys20\\Desktop\\yolo\\output_video.mp4" --method mosaic
#cmake, dlib 설치한 뒤 camke 이용해 dlib 내부에 있는 setup.py 불러옴
