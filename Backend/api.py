import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io
import logging
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# Spotify API 인증 설정
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

# 클래스 이름 설정 (데이터셋에서 사용된 MBTI 유형)
classes = ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP']

# 데이터 전처리 설정 (훈련 시 사용한 것과 동일하게)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 사전 학습된 모델 로드 및 수정
try:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(classes))

    # 모델을 GPU로 이동 (가능한 경우)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 저장된 모델 가중치 로드
    best_model_path = './model/ResNet18.pth'  # 저장된 모델 가중치 경로 설정
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Error loading the model: %s", str(e))
    raise RuntimeError("Error loading the model.")

# FastAPI 애플리케이션 생성
app = FastAPI()

# 추론 함수
def predict_image(image: Image.Image, temperature: float = 1.0):
    try:
        image = transform(image)
        logger.info("Image transformed successfully.")
    except Exception as e:
        logger.error("Error during image transformation: %s", str(e))
        raise RuntimeError("Error during image transformation.")

    try:
        image = image.unsqueeze(0)  # 배치 크기를 1로 설정
        image = image.to(device)
        logger.info("Image prepared for model input.")
    except Exception as e:
        logger.error("Error preparing image for model input: %s", str(e))
        raise RuntimeError("Error preparing image for model input.")

    try:
        with torch.no_grad():
            outputs = model(image)
            logger.info("Model inference completed.")
            outputs = outputs / temperature
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]  # 소프트맥스 확률 계산
            logger.info("Softmax probabilities calculated.")
    except Exception as e:
        logger.error("Error during model inference: %s", str(e))
        raise RuntimeError("Error during model inference.")

    try:
        probabilities_dict = {classes[i]: round(float(probabilities[i]) * 100, 2) for i in range(len(classes))}
        logger.info("Probabilities dictionary created: %s", probabilities_dict)
        return probabilities_dict
    except Exception as e:
        logger.error("Error creating probabilities dictionary: %s", str(e))
        raise RuntimeError("Error creating probabilities dictionary.")

# Spotify 노래 추천 함수 (다양성 추가 및 한국어 인기 노래 집중)
def recommend_songs(mbti: str, mbti_score: dict, limit: int = 5):
    try:
        mbti_genres = {
            'ENFJ': ['k-pop', 'k-ballad', 'k-indie'],
            'ENFP': ['k-indie', 'k-acoustic', 'k-pop'],
            'ENTJ': ['k-rock', 'k-metal', 'k-pop'],
            'ENTP': ['k-electronic', 'k-hip-hop', 'k-pop'],
            'ESFJ': ['k-dance', 'k-pop', 'k-ballad'],
            'ESFP': ['k-pop', 'k-dance', 'k-hip-hop'],
            'ESTJ': ['k-classical', 'k-orchestral', 'k-pop'],
            'ESTP': ['k-hip-hop', 'k-rap', 'k-pop'],
            'INFJ': ['k-chill', 'k-ambient', 'k-pop'],
            'INFP': ['k-folk', 'k-acoustic', 'k-indie'],
            'INTJ': ['k-jazz', 'k-blues', 'k-pop'],
            'INTP': ['k-blues', 'k-rock', 'k-pop'],
            'ISFJ': ['k-acoustic', 'k-ballad', 'k-pop'],
            'ISFP': ['k-singer-songwriter', 'k-indie', 'k-pop'],
            'ISTJ': ['k-country', 'k-folk', 'k-pop'],
            'ISTP': ['k-reggae', 'k-ska', 'k-pop']
        }

        # 추천 곡 수 설정
        recommendations = []

        # 장르 선택
        genres = mbti_genres.get(mbti, ['k-pop'])

        # MBTI 점수에 따라 추천할 곡 수 비율 설정
        for genre in genres:
            results = spotify.search(q=f'genre:{genre} lang:ko tag:popular', type='track', limit=limit)
            tracks = results['tracks']['items']
            for track in tracks:
                recommendations.append({
                    'title': track['name'],
                    'artist': track['artists'][0]['name'],
                    'url': track['external_urls']['spotify'],
                    'preview_url': track['preview_url']
                })

            # If no tracks are found for the genre, perform a broader search
            if not tracks:
                logger.warning(f"No tracks found for genre: {genre}. Expanding search.")
                results = spotify.search(q=f'genre:{genre}', type='track', limit=limit)
                tracks = results['tracks']['items']
                for track in tracks:
                    recommendations.append({
                        'title': track['name'],
                        'artist': track['artists'][0]['name'],
                        'url': track['external_urls']['spotify'],
                        'preview_url': track['preview_url']
                    })

        # If no recommendations found, use a general fallback to k-pop popular songs
        if not recommendations:
            logger.warning(f"No recommendations found for MBTI: {mbti}. Using fallback to popular k-pop songs.")
            results = spotify.search(q='genre:k-pop tag:popular', type='track', limit=limit)
            tracks = results['tracks']['items']
            for track in tracks:
                recommendations.append({
                    'title': track['name'],
                    'artist': track['artists'][0]['name'],
                    'url': track['external_urls']['spotify'],
                    'preview_url': track['preview_url']
                })

        logger.info(f"Recommended songs for {mbti}: {recommendations}")
        return recommendations[:limit]
    except Exception as e:
        logger.error("Error during song recommendation: %s", str(e))
        return []

# 이미지 업로드 및 예측 엔드포인트
@app.post("/predict/")
async def predict(file: UploadFile = File(...), temperature: float = 1.0):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        logger.info("Image opened and converted to RGB successfully.")
    except Exception as e:
        logger.error("Error processing the uploaded image: %s", str(e))
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        prediction = predict_image(image, temperature)
        highest_mbti = max(prediction, key=prediction.get)
        song_recommendations = recommend_songs(highest_mbti, prediction)
        return JSONResponse(content={"predicted_probabilities": prediction, "song_recommendations": song_recommendations})
    except RuntimeError as e:
        logger.error("Prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Prediction failed.")

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error("Error starting the server: %s", str(e))
        raise



