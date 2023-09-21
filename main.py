import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import os

# Função para calcular o erro final
def calcular_erro_final(y_real, y_pred):
    return y_real - y_pred

# Função para salvar a figura em uma pasta
def salvar_figura(nome_figura, pasta):
    if not os.path.exists(pasta):
        os.makedirs(pasta)
    plt.savefig(os.path.join(pasta, nome_figura))
    plt.close()

# Carregar o arquivo de teste uma única vez
print('Carregando Arquivo de teste')
arquivo = np.load('teste2.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])

# Configurações comuns do regressor MLP
regr_params = {
    'hidden_layer_sizes': (5,5),
    'max_iter': 300,
    'activation': 'identity',
    'solver': 'adam',
    'learning_rate': 'adaptive',
    'n_iter_no_change': 300
}

resultados = []

# Executar 10 simulações
erros_finais = []  # Armazenar os erros finais de cada simulação
for i in range(10):
    print(f'Simulação {i+1}')
    regr = MLPRegressor(**regr_params)

    print('Treinando RNA')
    regr = regr.fit(x, y)

    print('Preditor')
    y_est = regr.predict(x)

    plt.figure(figsize=[14, 7])

    # Plotar o gráfico original
    plt.subplot(1, 3, 1)
    plt.plot(x, y)

    # Plotar a curva de perda
    plt.subplot(1, 3, 2)
    plt.plot(regr.loss_curve_)

    # Plotar o regressor
    plt.subplot(1, 3, 3)
    plt.plot(x, y, linewidth=1, color='yellow')
    plt.plot(x, y_est, linewidth=2)

    # Salvar a figura na pasta
    nome_figura = f'simulacao_{i+1}.png'
    salvar_figura(nome_figura, 'resultados_simulacoes')

    # Calcular o erro final e armazená-lo
    erro_final = calcular_erro_final(y, y_est)
    erros_finais.append(erro_final)

# Calcular a média e o desvio padrão do erro final
erros_finais = np.array(erros_finais)
media_erro_final = np.mean(erros_finais)
desvio_padrao_erro_final = np.std(erros_finais)

# Imprimir a média e o desvio padrão do erro final no terminal
print(f'Média do Erro Final das 10 Simulações: {media_erro_final}')
print(f'Desvio Padrão do Erro Final das 10 Simulações: {desvio_padrao_erro_final}')
