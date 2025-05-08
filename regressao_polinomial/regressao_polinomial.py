import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

def cost_function(theta, X, y, lambda_=0.1):
    """
    Calcula o custo e o gradiente para a regressão linear com regularização.

    Parâmetros:
    theta -- Parâmetros do modelo (vetor 1D)
    X -- Matriz de características (m x n)
    y -- Vetor de valores alvo (m x 1)
    lambda_ -- Parâmetro de regularização (default: 0.1)

    Retorna:
    J -- O valor do custo
    grad -- O vetor de gradiente (1D)
    """
    m = y.size
    h = np.dot(X, theta)
    # Calculo do erro no conjunto de testes, que são as previsões - valores reais
    error = h - y
    
    # Com isso, calculamos o custo, a partir da fórmula estabelecida na nota de aula (utilizando funções do numpy), mas isso sem a regularização
    cost = (1 / (2 * m)) * np.dot(error, error)

    # Aqui é colocado um limite para os coeficientes, ou seja, é feito a regularização para que não aconteça o overfitting e aconteça dos coeficientes subirem demasiado.
    reg = (lambda_ / (2 * m)) * np.dot(theta[1:], theta[1:])
    J = cost + reg

    # Aqui é o cálculo do gradiente, já com a regularização, também para não acontecer o overfitting.
    # O gradiente é necessário para a otimização dos parâmetros, para isso a cada iteração ele ajusta os parâmetros, que se utilizam de alfa para basear o nível de alteração
    # Sem regularização
    grad = (1 / m) * np.dot(X.T, error)
    # Com regularização
    grad[1:] += (lambda_ / m) * theta[1:] 

    return J, grad

def optimize_theta(X, y, initial_theta, lambda_=0.1):
    opt_results = opt.minimize(cost_function, initial_theta, args=(X, y, lambda_), method='L-BFGS-B',
                               jac=True, options={'maxiter': 400})
    if not opt_results.success:
        raise RuntimeError("Otimização falhou: " + opt_results.message)
    return opt_results['x'], opt_results['fun']

def feature_normalize(X, mean=None, std=None):
    X = np.array(X)
    if mean is None or std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0, ddof=1)
    X = (X - mean) / std
    return X, mean, std

def extend_feature(X_ini, k):
    result = X_ini
    for i in range(2, k+1):
        result = np.hstack((result, np.power(X_ini, i)))
    return result

# Geração de dados
N = 30
x = np.linspace(0, 1, N)
y = np.sin(2*np.pi*x) + np.random.normal(0, 0.5, N)

NR = 100
xr = np.linspace(0, 1, NR)
yr = np.sin(2*np.pi*xr)

m = y.size

# Preparação dos dados
k = 9
X_ini = x.copy()
X_ini = X_ini.reshape(-1, 1)
X = extend_feature(X_ini, k)
X, mean, std = feature_normalize(X)
ones = np.ones((m, 1))
print(X.shape)
print(ones.shape)
X = np.hstack([ones, X])

#X = np.hstack([np.ones((m, 1)), X])  # Correção aplicada aqui
theta = np.random.randn(k + 1)

# Otimização
opt_theta, cost = optimize_theta(X, y, theta)

# Previsão
xnew = np.linspace(0, 1, 50)
xnew = xnew.reshape(-1, 1)
X2 = extend_feature(xnew, k)
X2 = (X2 - mean) / std
X2 = np.hstack([np.ones((xnew.shape[0], 1)), X2])
h = np.dot(X2, opt_theta)

# Visualização
line1, = plt.plot(xnew, h, label='Regression')
line2, = plt.plot(xr, yr, label='True distribution')
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression of Order 9')
plt.legend(handles=[line1, line2])
plt.show()