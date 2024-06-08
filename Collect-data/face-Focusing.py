import cv2
import os
from mtcnn import MTCNN
import shutil
from PIL import Image

# 입력 및 출력 디렉토리 설정
input_dir = './dataset/Convert_MBTI'
output_dir = './dataset/MBTI_Cropped'
verification_dir = './dataset/MBTI_Cropped/Verification'

# 얼굴 검출 모델 초기화
detector = MTCNN()

# 출력 및 검수 디렉토리가 존재하지 않으면 생성
for directory in [output_dir, verification_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# MBTI 유형 리스트
mbti_types = ['INTJ', 'INTP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'ENFJ', 'ENFP', 
              'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ', 'ISTP', 'ISFP', 'ESTP', 'ESFP']

def crop_and_save_faces(input_path, output_path, file_name):
    # 이미지 읽기
    img = cv2.imread(input_path)
    if img is None:
        print(f"[WARNING] Unable to read image: {input_path}")
        return False

    detections = detector.detect_faces(img)

    if len(detections) == 0:
        print(f"[INFO] No faces found in {input_path}")
        return False

    # 가장 큰 얼굴만 선택
    box = max(detections, key=lambda det: det['box'][2] * det['box'][3])['box']
    x, y, w, h = box

    # 얼굴 영역 크롭
    face = img[y:y+h, x:x+w]

    # 224x224 크기로 리사이즈
    face_resized = cv2.resize(face, (224, 224))

    # 출력 경로에 저장
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file_path = os.path.join(output_path, file_name)
    cv2.imwrite(output_file_path, face_resized)
    print(f"[INFO] Saved cropped image to {output_file_path}")
    return True

def verify_images(input_dir, verification_dir, min_size=50):
    for mbti_type in mbti_types:
        input_mbti_dir = os.path.join(input_dir, mbti_type)
        verification_mbti_dir = os.path.join(verification_dir, mbti_type)
        
        if not os.path.exists(input_mbti_dir):
            continue

        if not os.path.exists(verification_mbti_dir):
            os.makedirs(verification_mbti_dir)

        for file_name in os.listdir(input_mbti_dir):
            file_path = os.path.join(input_mbti_dir, file_name)
            img = Image.open(file_path)

            # 비정상적으로 작은 이미지를 확인하여 삭제
            if img.size[0] < min_size or img.size[1] < min_size:
                print(f"[INFO] Deleting small image: {file_path}")
                os.remove(file_path)
                continue
            
            # 확인 디렉토리로 복사
            shutil.copy(file_path, verification_mbti_dir)

# 얼굴 크롭 및 저장
total_cropped_images = 0
for mbti_type in mbti_types:
    input_mbti_dir = os.path.join(input_dir, mbti_type)
    output_mbti_dir = os.path.join(output_dir, mbti_type)
    
    if not os.path.exists(input_mbti_dir):
        continue

    if not os.path.exists(output_mbti_dir):
        os.makedirs(output_mbti_dir)

    for idx, file_name in enumerate(os.listdir(input_mbti_dir), 1):
        input_file_path = os.path.join(input_mbti_dir, file_name)
        output_file_name = f"{idx}.jpg"
        success = crop_and_save_faces(input_file_path, output_mbti_dir, output_file_name)
        if success:
            total_cropped_images += 1
        else:
            # 얼굴 검출에 실패한 경우 원본 파일을 복사
            verification_mbti_dir = os.path.join(verification_dir, mbti_type)
            if not os.path.exists(verification_mbti_dir):
                os.makedirs(verification_mbti_dir)
            shutil.copy(input_file_path, os.path.join(verification_mbti_dir, f"{idx}_failed.jpg"))

# 이미지 검수 및 처리
verify_images(output_dir, verification_dir)

print(f"[INFO] Total cropped images: {total_cropped_images}")
print("[INFO] Verification process completed.")
