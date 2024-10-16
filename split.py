import os
import random
import shutil

# 이미지와 라벨 경로 설정 
image_dir = 'C:/path_to_your_folder/images'  # 이미지가 있는 폴더 경로
label_dir = 'C:/path_to_your_folder/labels'  # 라벨 파일이 있는 폴더 경로

# 학습 및 검증 폴더 경로 설정
train_image_dir = 'C:/path_to_your_folder/images/train'
val_image_dir = 'C:/path_to_your_folder/images/val'
train_label_dir = 'C:/path_to_your_folder/labels/train'
val_label_dir = 'C:/path_to_your_folder/labels/val'

# train/val 디렉토리 생성
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
random.shuffle(image_files)  # 이미지 순서를 섞어줌

# 80%는 학습, 20%는 검증용으로 분리
split_index = int(len(image_files) * 0.8)
train_images = image_files[:split_index]
val_images = image_files[split_index:]

# 학습 데이터로 파일 복사
for image in train_images:
    # 이미지 복사
    shutil.copy(os.path.join(image_dir, image), os.path.join(train_image_dir, image))
    # 라벨 복사 (이미지 파일명과 같은 .txt 파일 복사)
    label_file = image.replace('.jpg', '.txt')
    shutil.copy(os.path.join(label_dir, label_file), os.path.join(train_label_dir, label_file))

# 검증 데이터로 파일 복사
for image in val_images:
    # 이미지 복사
    shutil.copy(os.path.join(image_dir, image), os.path.join(val_image_dir, image))
    # 라벨 복사
    label_file = image.replace('.jpg', '.txt')
    shutil.copy(os.path.join(label_dir, label_file), os.path.join(val_label_dir, label_file))

print(f'학습 데이터: {len(train_images)}장, 검증 데이터: {len(val_images)}장으로 분리되었습니다.')
