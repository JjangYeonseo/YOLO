pip install pytube opencv-python

import cv2
import os
from pytube import YouTube

# 유튜브 영상 다운로드 함수
def download_youtube_video(url, output_path="video.mp4"):
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    stream.download(filename=output_path)
    print(f"Video downloaded as {output_path}")
    return output_path

# 영상에서 프레임 추출 함수
def video_to_frames(video_path, output_folder="frames", fps=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    success, image = vidcap.read()
    count = 0
    frame_count = 0

    while success:
        if count % frame_interval == 0:
            cv2.imwrite(os.path.join(output_folder, f"frame{frame_count}.jpg"), image)
            print(f"Saved frame{frame_count}.jpg")
            frame_count += 1
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    print(f"Extracted {frame_count} frames.")

# 유튜브 영상 URL
youtube_url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# 영상 다운로드
video_path = download_youtube_video(youtube_url)

# 프레임 추출 (fps=1이면 1초에 한 장씩 이미지 저장)
video_to_frames(video_path, output_folder="output_frames", fps=1)
