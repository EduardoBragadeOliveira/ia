import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def load_data():
    data = loadmat("treinamento-redes-neurais/mnistdata.mat")
    X = data["X"]
    y = data["y"].flatten()
    return X, y

print("Carregando dados...")
X, y = load_data()

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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_):
    # Extrair os parâmetros
    W2_size = hidden_layer_size * input_layer_size
    b2_size = hidden_layer_size
    W3_size = num_labels * hidden_layer_size
    b3_size = num_labels

    W2 = nn_params[:W2_size].reshape(hidden_layer_size, input_layer_size)
    b2 = nn_params[W2_size:W2_size + b2_size].reshape(1, hidden_layer_size)
    W3 = nn_params[W2_size + b2_size:W2_size + b2_size + W3_size].reshape(num_labels, hidden_layer_size)
    b3 = nn_params[W2_size + b2_size + W3_size:].reshape(1, num_labels)

    m = X.shape[0]
    a1 = X
    Y = np.eye(num_labels)[y.flatten() - 1]

    # Forward propagation
    z2 = a1 @ W2.T + b2
    a2 = sigmoid(z2)

    z3 = a2 @ W3.T + b3
    a3 = sigmoid(z3)

    # Cálculo do custo
    J = -np.sum(Y * np.log(a3) + (1 - Y) * np.log(1 - a3)) / m
    reg = (lambda_ / (2 * m)) * (np.sum(W2 ** 2) + np.sum(W3 ** 2))
    J += reg

    # Backpropagation
    delta3 = a3 - Y
    delta2 = (delta3 @ W3) * sigmoid_gradient(z2)

    # Gradientes (sem regularização)
    # Aqui, é feito o produto matricial entre o erro e o A2, que vem da camada de saída, a divisão por m se dá para a normalização (média dos gradientes) (para os pesos)
    W3_grad = (delta3.T @ a2) / m
    # Aqui vai ser o cálculo do gradiente da função de custo para a camada de saída, no caso a terceira camada (para os bias)
    b3_grad = np.sum(delta3, axis=0, keepdims=True) / m

    # Aqui, é feito o produto matricial entre o erro e o A1, que vem da camada intermediária (oculta), a divisão por m se dá para a normalização (média dos gradientes) (para os pesos)
    W2_grad = (delta2.T @ a1) / m
    # Aqui vai ser o cálculo do gradiente da função de custo para a camada intermediária (oculta), no caso a segunda camada (para os bias)
    b2_grad = np.sum(delta2, axis=0, keepdims=True) / m


    # Regularização (aplicada apenas aos pesos, não aos biases)
    W3_grad += (lambda_ / m) * W3
    W2_grad += (lambda_ / m) * W2

    # Juntar todos os gradientes
    grad = np.concatenate([W2_grad.ravel(), b2_grad.ravel(), W3_grad.ravel(), b3_grad.ravel()])

    return J, grad

def computeNumericalGradient(costFunc, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    epsilon = 1e-4

    for i in range(len(theta)):
        perturb[i] = epsilon
        loss1 = costFunc(theta - perturb)[0]
        loss2 = costFunc(theta + perturb)[0]
        numgrad[i] = (loss2 - loss1) / (2 * epsilon)
        perturb[i] = 0

    return numgrad

def debugInitializeWeights(fan_out, fan_in):
    return np.sin(np.arange(1, (fan_out * fan_in) + 1)).reshape(fan_out, fan_in) / 10.0

def debugInitializeBias(size):
    return np.sin(np.arange(1, size + 1)).reshape(1, size) / 10.0

def checkNNGradients(lambda_):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    W2 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    b2 = debugInitializeBias(hidden_layer_size)
    W3 = debugInitializeWeights(num_labels, hidden_layer_size)
    b3 = debugInitializeBias(num_labels)

    X = debugInitializeWeights(m, input_layer_size)
    y = np.array([i % num_labels + 1 for i in range(1, m + 1)])

    nn_params = np.concatenate([W2.ravel(), b2.ravel(), W3.ravel(), b3.ravel()])
    costFunc = lambda p: cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)

    grad = computeNumericalGradient(costFunc, nn_params)
    _, grad_analytic = costFunc(nn_params)

    print("Numerical Gradient vs Analytical Gradient:")
    print(np.c_[grad, grad_analytic])

    diff = np.linalg.norm(grad - grad_analytic) / np.linalg.norm(grad + grad_analytic)
    print(f"Relative Difference: {diff}")

lambda_ = 1.0
checkNNGradients(lambda_ )

#Aquitetura da Rede Neural
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
lambda_ = 1

# Inicialização dos pesos de forma separada
initial_W2 = np.random.rand(hidden_layer_size, input_layer_size) * 0.12 - 0.06
initial_b2 = np.zeros((1, hidden_layer_size))
initial_W3 = np.random.rand(num_labels, hidden_layer_size) * 0.12 - 0.06
initial_b3 = np.zeros((1, num_labels))

initial_nn_params = np.concatenate([
    initial_W2.ravel(),
    initial_b2.ravel(),
    initial_W3.ravel(),
    initial_b3.ravel()
])

print("Treinando a rede neural...")
options = {'maxiter': 50}

costFunc = lambda p: cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)

result = minimize(fun=lambda p: costFunc(p)[0], x0=initial_nn_params,
                  jac=lambda p: costFunc(p)[1], method='CG', options=options)

nn_params = result.x

# Extrair os parâmetros após o treinamento
W2_size = hidden_layer_size * input_layer_size
b2_size = hidden_layer_size
W3_size = num_labels * hidden_layer_size

W2 = nn_params[:W2_size].reshape(hidden_layer_size, input_layer_size)
b2 = nn_params[W2_size:W2_size + b2_size].reshape(1, hidden_layer_size)
W3 = nn_params[W2_size + b2_size:W2_size + b2_size + W3_size].reshape(num_labels, hidden_layer_size)
b3 = nn_params[W2_size + b2_size + W3_size:].reshape(1, num_labels)

print("Rede treinada com sucesso!")

def predict(W2, b2, W3, b3, X):
    #W = pesos e B = bias, que são utilizados no gradiente
    # Camada de entrada, que recebe os dados de entrada x1, x2 e x3. (Camada AND?)
    a1 = X  # (5000, 400)

    # Camada intermediária (oculta), que processa os dados intermediários e cria representações abstratas (Camada XOR?)
    z2 = np.dot(a1, W2.T) + b2.squeeze()  
    a2 = sigmoid(z2)

    # Camada de saída, gera a previsão final do modelo (Camada OR?)
    z3 = np.dot(a2, W3.T) + b3.squeeze()  
    a3 = sigmoid(z3)
    # As sigmoids são utilizadas para criar funções não lineares, que permitem identificar padrões complexos que modelos lineares não conseguem.

    # Previsão = índice da maior ativação + 1
    return np.argmax(a3, axis=1) + 1

# Testando a rede
pred = predict(W2, b2, W3, b3, X)
accuracy = np.mean(pred == y) * 100
print(f"Precisão do treinamento: {accuracy:.2f}%")