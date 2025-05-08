import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data2.txt", sep=",", header=None)
df.columns = ['exame_1', 'exame_2', 'classe']

# Plotagem do pontos, consequentemente a visualização dos pontos
sns.scatterplot(data=df, x="exame_1", y="exame_2", hue="classe", style="classe", s=80)
plt.title("Distribuição das classes (Exercício final)")
plt.show()

# Como a regressão logística é linear por natureza, para escrever um modelo não-linear, precisamos adicionar termos quadráticos e interação entre as variáveis, como é descrito abaixo, com isso, transforma-se o modelo em não-linear a partir dessa feature polinomial
X = df[["exame_1", "exame_2"]]
y = df["classe"]
X_feature_poli = X.copy()
X_feature_poli["exame_1^2"] = X["exame_1"]**2
X_feature_poli["exame_2^2"] = X["exame_2"]**2
X_feature_poli["exame_1*exame_2"] = X["exame_1"] * X["exame_2"]

# Este 1000 é um número arbitrário, ele é usado para delimitar o máximo de interações que o algoritmo pode fazer, para que ache o melhor peso. Na segunda linha, ele faz o treinamento do modelo com os dados da feature polinomial.
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_feature_poli, y)

# Estas duas linhas abaixo delimitam o gráfico, com essa sobra, que também é arbitrária.
x1_min, x1_max = X["exame_1"].min() - 0.5, X["exame_1"].max() + 0.5
x2_min, x2_max = X["exame_2"].min() - 0.5, X["exame_2"].max() + 0.5
# Cria uma malha de pontos cobrindo a área do gráfico, criando xx1 e xx2, que são matrizes, ou seja, os pontos em um plano.
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                       np.linspace(x2_min, x2_max, 300))
# Aqui ele vai gerar novamente os termos polinomiais para a grid. O dataframe cria uma tabela de dados para organizar os mesmos.
grid = pd.DataFrame({
    "exame_1": xx1.ravel(),
    "exame_2": xx2.ravel()
})
grid["exame_1^2"] = grid["exame_1"]**2
grid["exame_2^2"] = grid["exame_2"]**2
grid["exame_1*exame_2"] = grid["exame_1"] * grid["exame_2"]
# Parte importante, aqui é onde ele aplica o modelo na malha de pontos (na grid), depois, ele vai redesenhar os pontos para que fiquem de acordo com a aplicação do modelo.
pred = modelo.predict(grid)
pred = pred.reshape(xx1.shape)
# Aqui ele vai apenas plotar o gráfico
plt.figure(figsize=(8, 6))
plt.contourf(xx1, xx2, pred, alpha=0.3, cmap="coolwarm")
sns.scatterplot(x=X["exame_1"], y=X["exame_2"], hue=y, style=y, s=80, edgecolor="k")
plt.title("Superfície de decisão - Classificação Não Linear")
plt.xlabel("exame_1")
plt.ylabel("exame_2")
plt.show()