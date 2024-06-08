from PIL import Image
import os

# 기본 디렉토리 설정
input_base_dir = './dataset/MBTI'
output_base_dir = './dataset/Convert_MBTI'
mbti_types = ['INTJ', 'INTP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'ENFJ', 'ENFP', 'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ', 'ISTP', 'ISFP', 'ESTP', 'ESFP']

# MBTI 폴더 생성
def create_mbti_folders(base_dir, mbti_types):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for mbti_type in mbti_types:
        mbti_dir = os.path.join(base_dir, mbti_type)
        if not os.path.exists(mbti_dir):
            os.makedirs(mbti_dir)
            print(f'[INFO] Created folder: {mbti_dir}')

# 함수: WEBP 파일을 JPG로 변환
def convert_webp_to_jpg(input_base_dir, output_base_dir):
    for mbti_type in mbti_types:
        input_mbti_dir = os.path.join(input_base_dir, mbti_type)
        output_mbti_dir = os.path.join(output_base_dir, mbti_type)
        
        if os.path.isdir(input_mbti_dir):
            if not os.path.exists(output_mbti_dir):
                os.makedirs(output_mbti_dir)
                print(f'[INFO] Created output folder: {output_mbti_dir}')
            
            file_index = 1
            for root, _, files in os.walk(input_mbti_dir):
                for file in files:
                    if file.lower().endswith('.webp'):
                        webp_path = os.path.join(root, file)
                        jpg_filename = f'{file_index}.jpg'
                        jpg_path = os.path.join(output_mbti_dir, jpg_filename)
                        
                        try:
                            with Image.open(webp_path) as img:
                                rgb_img = img.convert('RGB')
                                rgb_img.save(jpg_path, 'JPEG')
                            print(f'[INFO] Converted {webp_path} to {jpg_path}')
                            file_index += 1
                        except Exception as e:
                            print(f'[ERROR] Failed to convert {webp_path}: {e}')
                    else:
                        print(f'[INFO] Skipped non-WEBP file: {file}')
        else:
            print(f'[WARNING] Input directory does not exist: {input_mbti_dir}')

if __name__ == "__main__":
    # MBTI 폴더 생성 (출력 디렉토리만)
    create_mbti_folders(output_base_dir, mbti_types)
    # WEBP 파일을 JPG로 변환
    convert_webp_to_jpg(input_base_dir, output_base_dir)
