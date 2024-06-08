import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np

# 데이터 전처리 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 경로 설정
dataset_path = './dataset/MBTI_Cropped'
dataset = ImageFolder(root=dataset_path, transform=transform)

# 클래스 이름 출력 (MBTI 유형)
print("Classes:", dataset.classes)

# 클래스별 데이터 수 확인
class_counts = np.bincount([label for _, label in dataset])
class_weights = 1. / class_counts
sample_weights = np.array([class_weights[label] for _, label in dataset])

# 데이터셋 분할 (예: 80% 훈련, 20% 검증)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader에 샘플 가중치 적용
train_weights = sample_weights[train_dataset.indices]
train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 사전 학습된 ResNet18 모델 로드 및 수정
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes))

# 모델을 GPU로 이동 (가능한 경우)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실 함수와 옵티마이저 설정
class_weights = torch.tensor(class_weights).float().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)  # 학습률 감소 스케줄러

# 학습 및 검증 손실 및 정확도 기록용 리스트
train_losses = []
val_losses = []
val_accuracies = []

# 모델 저장 경로 설정
best_model_path = './best_model.pth'
best_accuracy = 0.0

# 학습 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20):
    global best_accuracy
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        train_losses.append(running_loss / len(train_loader))

        # 검증 단계
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct / total)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Val Accuracy: {100 * correct / total}%")

        # 모델 저장
        if 100 * correct / total > best_accuracy:
            best_accuracy = 100 * correct / total
            torch.save(model.state_dict(), best_model_path)

# 모델 학습
num_epochs = 100  # 원하는 에포크 수로 조정
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)

# 학습 과정 시각화
def plot_metrics(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r', label='Training loss')
    plt.plot(epochs, val_losses, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.show()

# 학습 과정 시각화 실행
plot_metrics(train_losses, val_losses, val_accuracies)

# 모델 평가 함수
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Final Validation Accuracy: {100 * correct / total}%')

# 모델 평가 실행
evaluate_model(model, val_loader)