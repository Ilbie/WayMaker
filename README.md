# WayMaker

**WayMaker**는 사용자의 관상과 MBTI를 분석하여, 그에 맞는 노래를 스포티파이를 통해 추천해주는 자바 애플리케이션입니다.

![WayMaker]()

[![라이센스: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/Ilbie/WayMaker/blob/main/LICENSE)
[![Colab 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ilbie/WayMaker/blob/main/Deep_learning.ipynb)

## 목차
1. [프로젝트 소개](#프로젝트-소개)
2. [기능](#기능)
    1. [사용자 관상 분석](#사용자-관상-분석)
    2. [사용자 MBTI 분석](#사용자-mbti-분석)
    3. [맞춤형 노래 추천](#맞춤형-노래-추천)
3. [사용 기술](#사용-기술)
4. [설치 및 실행 방법](#설치-및-실행-방법)
    1. [프론트엔드 설치 및 실행](#프론트엔드-설치-및-실행)
    2. [백엔드 설치 및 실행](#백엔드-설치-및-실행)
5. [Google Colab에서 실행](#google-colab에서-실행)

## 프로젝트 소개
WayMaker는 관상과 MBTI를 기반으로 사용자에게 맞춤형 노래를 추천하는 자바 애플리케이션입니다. 이를 통해 사용자들은 자신에게 어울리는 음악을 더 쉽게 찾을 수 있습니다.

## 기능

### 데이터 수집
웹사이트에서 크롤링하여 데이터를 전처리 및 수정합니다.
- **사용 파일:** `Convert-files.py`, `crawling.py`, `face-Focusing.py`
- **기술:** MTCNN, OpenCV

### 사용자 관상 분석
사용자의 얼굴을 분석하여 특정한 특징들을 추출합니다. 이를 위해 OpenCV와 같은 얼굴 인식 라이브러리를 사용하며, 심층 학습을 통해 관상 데이터를 학습하고 분석합니다.
- **사용 파일:** `Deep-learning.py` [![Colab 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ilbie/WayMaker/blob/main/Deep_learning.ipynb)
- **기술:** OpenCV, PyTorch 등

### 사용자 MBTI 분석
사용자가 입력한 MBTI 정보를 바탕으로 성격 유형을 분석합니다. 이 정보를 통해 사용자에게 더 적합한 음악을 추천할 수 있습니다.
- **사용 파일:** MBTI 분석 관련 모듈 (파일 명시 필요)
- **기술:** Java 기반 입력 처리 및 분석 로직

### 맞춤형 노래 추천
스포티파이 API를 사용하여 사용자에게 맞춤형 노래를 추천합니다. 관상과 MBTI 분석 결과를 종합하여 사용자에게 최적의 노래 리스트를 제공합니다.
- **사용 파일:** 
- **기술:** Spotify API, Web Crawling

## 사용 기술
- **프로그래밍 언어:** Java, Python
- **라이브러리 및 프레임워크:** OpenCV, PyTorch, Spotify API

## 설치 및 실행 방법

### 프론트엔드 설치 및 실행

1. 저장소를 복제합니다:
    ```sh
    git clone https://github.com/Ilbie/WayMaker.git
    ```

2. 자바 JDK 17을 설치합니다.

3. 프로젝트를 엽니다:
    ```sh
    cd WayMaker/Frontend
    ```

4. Java 프로젝트 설정:
    - [OpenJFX](https://openjfx.io)에서 JavaFX SDK 17.0.11을 다운로드하고 압축을 해제한 후, C:\ 루트 디렉토리에 옮깁니다.

5. Intellij IDEA에서 프로젝트를 엽니다.
    - `File` > `Project Structure` > `Libraries`에서 `+` 버튼을 클릭하여 JavaFX SDK를 추가합니다.
    - 다운로드한 JavaFX SDK의 `lib` 폴더를 선택하여 라이브러리를 추가합니다.

6. `pom.xml` 파일을 프로젝트의 루트 디렉토리에 추가합니다. 이는 프로젝트의 Maven 설정 파일입니다. 이미 제공된 파일을 참고합니다.

7. Maven을 사용하여 프로젝트를 빌드하고 실행합니다:
    ```sh
    mvn clean javafx:run
    ```
    ```
    Shift + F10
    ```

### 백엔드 설치 및 실행

1. 필요한 파이썬 라이브러리를 설치합니다:
    ```sh
    cd WayMaker
    pip install -r requirements.txt
    ```

2. 백엔드 디렉토리로 이동하여 API 서버를 실행합니다:
    ```sh
    cd Backend
    python api.py
    ```

---

