'''
    Todas as funções loss implemetandas na classe da rede neural
    Qualquer função de ativação nova precisa ser colocada nas funções loss_value,loss_derivate_value da classe NeuralNetwork
'''
import numpy as np

#ssr
def SSR(predict,correct):
    valor = pow((correct - predict),2)
    return valor

def SSR_derivada(predict,correct):
    valor = -2*(correct - predict)
    return valor    

#cross_entropy
def cross_entropy(predict,correct):
    m = predict.shape[1]
    cost = -1 / m * (np.dot(correct, np.log(predict).T) + np.dot(1 - correct, np.log(1 - predict).T))
    return np.squeeze(cost)

def cross_entropy_derivada(predict,correct):
    #correct = correct.reshape(predict.shape)
    return  - (np.divide(correct, predict) - np.divide(1 - correct, 1 - predict))