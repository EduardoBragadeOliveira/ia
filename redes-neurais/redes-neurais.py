import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def load_data():
    data = loadmat("redes-neurais/mnistdata.mat")
    X = data["X"]
    y = data["y"].flatten()
    return X, y

def load_weights():
    weights = loadmat("redes-neurais/pesos.mat")
    W2 = weights["W2"]
    b2 = weights["b2"]
    W3 = weights["W3"]
    b3 = weights["b3"]

    return W2, b2, W3, b3

print("Carregando dados...")
X, y = load_data()

imagem = X[0,:].reshape(20,20).T
plt.imshow(imagem, cmap='gray')
plt.colorbar(label="Intensidade do Pixel")
plt.show()

def display_random_images(X, y):
    indices = np.random.choice(X.shape[0], 100, replace=False)
    images = X[indices, :].reshape(-1, 20, 20).transpose(0, 2, 1)
    labels = y[indices]

    fig, axes = plt.subplots(10, 10, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f"{labels[i]}", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
display_random_images(X, y)

print("Carregando pesos treinados...")
W2, b2, W3, b3 = load_weights()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(W2, b2, W3, b3, X):
    # Camada de entrada
    a1 = X  # (5000, 400)

    # Camada intermediária
    z2 = np.dot(a1, W2.T) + b2.squeeze()  
    a2 = sigmoid(z2)

    # Camada de saída
    z3 = np.dot(a2, W3.T) + b3.squeeze()  
    a3 = sigmoid(z3)

    # Previsão = índice da maior ativação + 1
    return np.argmax(a3, axis=1) + 1

print("Fazendo previsões...")
pred = predict(W2, b2, W3, b3, X)
acc = np.mean(pred == y) * 100
print(f"Acurácia da rede: {acc:.2f}%")