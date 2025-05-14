import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Configurações
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = (64, 64)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device('cpu')

# Dataset personalizado
class ParkingDataset(Dataset):
    def __init__(self, root, transform=None):
        self.file_paths = []
        self.labels = []
        self.transform = transform
        for class_name in ['free', 'busy']:
            class_dir = os.path.join(root, class_name)
            label = 0 if class_name == 'free' else 1
            for fname in os.listdir(class_dir):
                if fname.endswith('.jpg'):
                    self.file_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Transformações
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# CNN simples
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Treinamento
def train(model, loader, optimizer, criterion):
    model.train()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device).view(-1, 1)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()

# Avaliação
def evaluate(model, loader):
    model.eval()
    preds, trues, imgs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            pred = (outputs > 0.5).float().cpu().numpy()
            preds.extend(pred)
            trues.extend(labels)
            imgs.extend(inputs)
    return preds, trues, imgs

# Carregar dados
train_data = ParkingDataset("data/A", transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# Instanciar e treinar modelo
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

for epoch in range(EPOCHS):
    train(model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1} concluída.")

# Avaliação final
preds, trues, imgs = evaluate(model, train_loader)
accuracy = accuracy_score(trues, preds)
print(f"Acurácia do modelo: {accuracy:.4f}")
correct_imgs = [img for img, p, t in zip(imgs, preds, trues) if int(p[0]) == t]
incorrect_imgs = [img for img, p, t in zip(imgs, preds, trues) if int(p[0]) != t]

num_correct = min(len(correct_imgs), 5)
num_incorrect = min(len(incorrect_imgs), 5)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(num_correct):
    axes[0][i].imshow(correct_imgs[i].permute(1, 2, 0) * 0.5 + 0.5)
    axes[0][i].set_title("Classificada correta")
    axes[0][i].axis('off')

for i in range(num_incorrect):
    axes[1][i].imshow(incorrect_imgs[i].permute(1, 2, 0) * 0.5 + 0.5)
    axes[1][i].set_title("Classificada errada")
    axes[1][i].axis('off')

plt.tight_layout()
plt.show()
