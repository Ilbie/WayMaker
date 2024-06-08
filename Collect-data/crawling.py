import requests
from bs4 import BeautifulSoup
import os
import time
import shutil

# 나무위키 URL 설정
base_url = 'https://namu.wiki/w/'

# MBTI 유형 리스트
mbti_types = ['INTJ', 'INTP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'ENFJ', 'ENFP', 'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ', 'ISTP', 'ISFP', 'ESTP', 'ESFP']

# 이미지를 저장할 디렉토리 설정
base_dir = './dataset'
output_dir = os.path.join(base_dir, 'MBTI')
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# MBTI 폴더 비우기
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# MBTI별 사진 수집 최대 개수 설정
max_images_per_mbti = 0

# HTML 클래스 이름 변수
main_div_class = 'cLYyzoTj Y7jG8RxA'
img_tag_class = 'yZ6yG6aM'

# 함수: 사용자 에이전트 설정
def get_headers():
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

# 함수: MBTI 페이지에서 특정 섹션의 인물 링크 수집
def get_person_links(mbti_type):
    url = f'{base_url}{mbti_type}'
    print(f'[INFO] Requesting URL: {url}')
    response = requests.get(url, headers=get_headers())
    soup = BeautifulSoup(response.text, 'html.parser')
    person_links = []

    sections_to_include = ['연예인', '정치인', '학자', '프로게이머', '만화가', '체육인', '의료인', '기업인', '인터넷 방송인 및 인플루언서', '기타 인물', 'ENTJ인 인물']

    in_section = False
    for tag in soup.select('h2, h3, ul li'):
        if tag.name in ['h2', 'h3'] and any(section in tag.text for section in sections_to_include):
            in_section = True
            print(f'[INFO] Found section: {tag.text.strip()} in {mbti_type}')
        elif tag.name in ['h2', 'h3'] and in_section:
            in_section = False
        elif in_section and tag.name == 'li':
            link_tag = tag.find('a')
            if link_tag:
                link = link_tag.get('href')
                if link and link.startswith('/w/'):
                    person_links.append(link)
                    print(f'[INFO] Found person link: {link}')

    if max_images_per_mbti == 0:
        return person_links  # 수집한 모든 링크 반환
    else:
        return person_links[:max_images_per_mbti]  # 최대 링크 수 반환

# 함수: 인물 페이지에서 이미지 URL 수집
def get_image_url(person_url):
    url = f'https://namu.wiki{person_url}'
    print(f'[INFO] Requesting person URL: {url}')
    response = requests.get(url, headers=get_headers())
    soup = BeautifulSoup(response.text, 'html.parser')

    # 주요 이미지 찾기
    main_div = soup.find('div', class_=main_div_class)
    if main_div:
        img_tag = main_div.find('img', class_=img_tag_class)
        if img_tag: 
            img_url = img_tag.get('data-src') or img_tag.get('src')
            if img_url and img_url.startswith('//'):
                img_url = 'https:' + img_url
                if img_url.endswith('.svg'):
                    print(f'[INFO] Skipping SVG image: {img_url}')
                    return None
                print(f'[INFO] Found image URL: {img_url}')
                return img_url

    return None

# 함수: 이미지 다운로드
def download_image(img_url, mbti_type, image_number):
    mbti_dir = os.path.join(output_dir, mbti_type)
    if not os.path.exists(mbti_dir):
        os.makedirs(mbti_dir)

    img_path = os.path.join(mbti_dir, f'{image_number}.webp')
    try:
        response = requests.get(img_url, headers=get_headers(), stream=True)
        if response.status_code == 200:
            with open(img_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f'[INFO] Downloaded: {img_path}')
        else:
            print(f'[ERROR] Failed to download {img_url}: HTTP {response.status_code}')
    except Exception as e:
        print(f'[ERROR] Failed to download {img_url}: {e}')

# 메인 함수
def main():
    total_images = 0

    for mbti_type in mbti_types:
        print(f'[INFO] Processing {mbti_type}...')
        person_links = get_person_links(mbti_type)

        image_number = 1
        for person_link in person_links:
            img_url = get_image_url(person_link)

            if img_url:
                download_image(img_url, mbti_type, image_number)
                image_number += 1
            else:
                print(f'[Warning] No image found for {person_link}')

            # 예의상 크롤링 속도를 조절
            time.sleep(1)

        print(f'[INFO] {mbti_type} collected {image_number - 1} images.')
        total_images += image_number - 1

    print(f'[INFO] Total images collected: {total_images}')

if __name__ == "__main__":
    main()
