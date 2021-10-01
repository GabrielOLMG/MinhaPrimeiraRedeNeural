from my_nn_class import NeuralNetwork
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------------------------------ CONSTANTES
SEED = 999
EPOCAS = 20000
#------------------------------------------------------------------------------------------------------------------------ PRIMEIRO TESTE
#--------------------------------------------------------------
'''
    Objetivo: Estimar a quantidade de bicicletas alugadas dado o clima e a temperatura
'''
#--------------------------------------------------------------
#------------------- ABRE O ARQUIVO
Dados = pd.read_csv('csv/Bicicletas.csv')

#------------------- SEPARA AS VARIAVEIS
X = Dados[Dados.columns[0:2]].values
Y = Dados[Dados.columns[-1]].values

#------------------- NORMALIZA OS VALORES
X_normal = X/np.amax(X,axis=0)
Y_max = np.amax(Y,axis=0)
Y_normal = Y/Y_max

#------------------- SEPARA EM TREINO E TESTE
X_treino, X_teste, y_treino, y_teste = train_test_split(X_normal, Y_normal, train_size = 0.8, random_state=SEED)

#------------------- COLOCA OS DADOS DE FORMA QUE SEJAM ACEITOS NA REDE (CADA LINHA TEM QUE SER UM DADO(CLIMA ,TEMPERATURA) E CADA COLUNA REPRESENTA UM VALOR))

X_treino = np.transpose(X_treino)
X_teste = np.transpose(X_teste)

y_treino = np.transpose(y_treino.reshape((y_treino.shape[0], 1)))
y_teste = np.transpose(y_teste.reshape((y_teste.shape[0], 1)))

#------------------- ARQUITETURA DA REDE NEURAL
arquitetura = [ 
    {"dim_entrada":1,"dim_saida":50,"activacao": "relu"},
    {"dim_entrada":50,"dim_saida":1,"activacao": "sigmoid"}
]

#------------------- REDE NEURAL
rede = NeuralNetwork(arquitetura, seed = SEED)

#------------------- TREINA REDE NEURAL
input_  = np.array([[0,0.5,1]])
correct_ = np.array([[0,1,0]])
#historico_loss, pesos, historico_acc, diferenca_hist = rede.treino(X_treino,y_treino,'cross_entropy',0.001,EPOCAS,1000)
#historico_loss, pesos, historico_acc, diferenca_hist = rede.treino_stoc(X_treino,y_treino,'cross_entropy',0.001,EPOCAS,150,1000)
historico_loss, pesos, historico_acc, diferenca_hist = rede.treino(input_,correct_,'cross_entropy',0.001,EPOCAS,1000)
pred = rede.predict(input_)
print(pred,correct_)

#pred = rede.predict(X_teste)
#print(pred[0,0:3]*Y_max,y_teste[0,0:3]*Y_max)
#------------------- PLOT DA LOSS
plt.plot(historico_loss)
plt.show()


#------------------------------------------------------------------------------------------------------------------------ SEGUNDO TESTE