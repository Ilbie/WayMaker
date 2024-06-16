import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import io
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 클래스 이름 설정 (데이터셋에서 사용된 MBTI 유형)
classes = ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP']

# 데이터 전처리 설정 (훈련 시 사용한 것과 동일하게)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 사전 학습된 모델 로드 함수
def load_model(model_name: str):
    if model_name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model_path = './model/ResNet18.pth'
    elif model_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model_path = './model/ResNet50.pth'
    else:
        raise ValueError("Unsupported model type")

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 모델을 GPU로 이동 (가능한 경우)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# FastAPI 애플리케이션 생성
app = FastAPI()

# 추론 함수
def predict_image(image: Image.Image, model, temperature: float = 1.0):
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
        
        # Apply adjustment for ResNet50
        if model.__class__.__name__ == 'ResNet' and model.fc.out_features == len(classes) and model.fc.in_features == 2048:
            total = sum(probabilities_dict.values())
            probabilities_dict = {k: round(v / total * 100, 2) for k, v in probabilities_dict.items()}
        
        logger.info("Probabilities dictionary created: %s", probabilities_dict)
        return probabilities_dict
    except Exception as e:
        logger.error("Error creating probabilities dictionary: %s", str(e))
        raise RuntimeError("Error creating probabilities dictionary.")

# 이미지 업로드 및 예측 엔드포인트
@app.post("/predict/")
async def predict(file: UploadFile = File(...), model_name: str = Query(...), temperature: float = 1.0):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        logger.info("Image opened and converted to RGB successfully.")
    except Exception as e:
        logger.error("Error processing the uploaded image: %s", str(e))
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        model = load_model(model_name).to(device)
        prediction = predict_image(image, model, temperature)
        return JSONResponse(content={"predicted_probabilities": prediction})
    except RuntimeError as e:
        logger.error("Prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Prediction failed.")
    except ValueError as e:
        logger.error("Model loading failed: %s", str(e))
        raise HTTPException(status_code=400, detail="Unsupported model type.")

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error("Error starting the server: %s", str(e))
        raise
