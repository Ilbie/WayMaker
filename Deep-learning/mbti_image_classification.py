import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

# 클래스 이름 설정 (데이터셋에서 사용된 MBTI 유형)
classes = ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'] # 실제 클래스 이름으로 교체

# 데이터 전처리 설정 (훈련 시 사용한 것과 동일하게)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 사전 학습된 모델 로드 및 수정
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))

# 모델을 GPU로 이동 (가능한 경우)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 저장된 모델 가중치 로드
best_model_path = '../Backend/model/best_model.pth'  # 저장된 모델 가중치 경로 설정
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

# 추론 함수
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # 배치 크기를 1로 설정
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    
    return classes[predicted.item()]

# 새로운 이미지에 대한 추론 실행
image_path = '../example2.jpg'
prediction = predict_image(image_path)
print(f'Predicted MBTI type: {prediction}')
