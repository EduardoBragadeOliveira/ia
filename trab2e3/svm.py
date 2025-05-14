import os
import random
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configurações
IMG_SIZE = (64, 64)
DATA_ROOT = "C:/Users/Aluno/Downloads/T_IA/T_IA/data"
DATA_WEATHER = "C:/Users/Aluno/Downloads/T_IA/T_IA/data/Dados"
CAMERAS = ['A', 'B']
CLASSES = ['free', 'busy']
WEATHER_CONDITIONS = ['OVERCAST', 'RAINY']
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

# Função para carregar imagens
def load_images_by_folder(root, folder_list):
    features, labels = [], []
    for folder in folder_list:
        for cls in CLASSES:
            label = 0 if cls == 'free' else 1
            dir_path = os.path.join(root, folder, cls)
            if not os.path.exists(dir_path):
                continue
            for fname in tqdm(os.listdir(dir_path), desc=f"{folder}/{cls}"):
                if fname.endswith('.jpg'):
                    img_path = os.path.join(dir_path, fname)
                    img = Image.open(img_path).convert('RGB')
                    img = transform(img)
                    features.append(img.view(-1).numpy())
                    labels.append(label)
    return np.array(features), np.array(labels)

# Pasta de saída
output_dir = "C:/Users/Aluno/Downloads/T_IA/resultados"
os.makedirs(output_dir, exist_ok=True)

# ---------- Parte 1: Avaliação por Câmeras ----------
results_camera = {}
X_total, y_total = [], []

for cam in CAMERAS:
    X, y = load_images_by_folder(DATA_ROOT, [cam])
    X_total.append(X)
    y_total.append(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results_camera[cam] = {'acc': acc, 'cm': cm}

# Geral (todas as câmeras)
X_all = np.vstack(X_total)
y_all = np.concatenate(y_total)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=RANDOM_SEED)

clf_all = SVC(kernel='linear')
clf_all.fit(X_train_all, y_train_all)
y_pred_all = clf_all.predict(X_test_all)

acc_all = accuracy_score(y_test_all, y_pred_all)
cm_all = confusion_matrix(y_test_all, y_pred_all)
results_camera['Geral'] = {'acc': acc_all, 'cm': cm_all}

# Gráfico de acurácia por câmera
acc_data_camera = pd.DataFrame({
    'Câmera': CAMERAS + ['Geral'],
    'Acurácia': [results_camera[cam]['acc'] for cam in CAMERAS] + [acc_all]
})

plt.figure(figsize=(7, 5))
sns.barplot(x='Câmera', y='Acurácia', data=acc_data_camera)
plt.title('Acurácia por Câmera e Geral - SVM')
plt.ylim(0.9, 1.01)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "acuracia_por_camera.png"))
plt.close()

# Matrizes de confusão por câmera
for cam in results_camera:
    cm = results_camera[cam]['cm']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['free', 'busy'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Matriz de Confusão - {cam}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"matriz_confusao_{cam}.png"))
    plt.close()

# ---------- Parte 2: Avaliação por Condição Climática ----------
results_weather = {}
X_weather, y_weather = [], []

for cond in WEATHER_CONDITIONS:
    X, y = load_images_by_folder(DATA_WEATHER, [cond])
    X_weather.append(X)
    y_weather.append(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results_weather[cond] = {'acc': acc, 'cm': cm}

# Geral (todas as condições climáticas)
X_weather_all = np.vstack(X_weather)
y_weather_all = np.concatenate(y_weather)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_weather_all, y_weather_all, test_size=0.2, random_state=RANDOM_SEED)

clf_weather = SVC(kernel='linear')
clf_weather.fit(X_train_all, y_train_all)
y_pred_weather = clf_weather.predict(X_test_all)

acc_weather_all = accuracy_score(y_test_all, y_pred_weather)
cm_weather_all = confusion_matrix(y_test_all, y_pred_weather)
results_weather['Geral'] = {'acc': acc_weather_all, 'cm': cm_weather_all}

# Gráfico de acurácia por clima

acc_data_weather = pd.DataFrame([
    {'Clima': cond, 'Acurácia': results_weather[cond]['acc']} for cond in WEATHER_CONDITIONS + ['Geral']
])

plt.figure(figsize=(7, 5))
plot = sns.barplot(x='Clima', y='Acurácia', data=acc_data_weather)

# Mostrar valores acima das barras
for i, row in acc_data_weather.iterrows():
    plot.text(i, row['Acurácia'] + 0.002, f"{row['Acurácia']:.4f}", color='black', ha="center", fontsize=9)

plt.title('Acurácia por Condição Climática - SVM')
plt.ylim(0.85, 1.01)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "acuracia_por_clima.png"))
plt.close()

# Matrizes de confusão por clima
for cond in results_weather:
    cm = results_weather[cond]['cm']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['free', 'busy'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Matriz de Confusão - {cond}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"matriz_confusao_clima_{cond}.png"))
    plt.close()

# Exibir acurácias no terminal
print("\n=== Acurácias por Câmera ===")
for cam in CAMERAS:
    print(f"Câmera {cam}: {results_camera[cam]['acc']:.4f}")
print(f"Geral: {results_camera['Geral']['acc']:.4f}")

print("\n=== Acurácias por Clima ===")
for cond in WEATHER_CONDITIONS:
    print(f"{cond}: {results_weather[cond]['acc']:.4f}")
print(f"Geral: {results_weather['Geral']['acc']:.4f}")
