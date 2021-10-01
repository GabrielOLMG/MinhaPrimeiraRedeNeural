'''
    Todas as funções de ativação implemetandas na classe da rede neural.
    Qualquer função de ativação nova precisa ser colocada nas funções dAtivado_value, A_atual_valueda classe NeuralNetwork
'''
import numpy as np

def softplus(x):
    return np.log(1+np.exp(x))

def softplus_derivada(x):
    return np.divide(np.exp(x),1+np.exp(x))

def None_Ac(x):
    return x

def None_Ac_derivada(x):
    return 1

'''
def relu(x):
    return np.maximum(0,x)

def relu_derivada(x):
    return 1 * (x > 0)
'''

def relu(x,alfa=0.01):
    return np.maximum(alfa*x, x)

def relu_derivada(x,alfa=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alfa
    return dx

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivada(x):
    sig = sigmoid(x)
    return  sig * (1 - sig)