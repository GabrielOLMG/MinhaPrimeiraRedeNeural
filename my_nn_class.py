import random
import numpy as np
from activation_functions import *
from loss_functions import *

class NeuralNetwork:
    def __init__(self,arquitetura,bias = None, pesos = None, seed = 0):
        self.rng = np.random.RandomState(seed)
        self.arquitetura = arquitetura
        self.pesos_bias_original = self.inicializa_camadas(pesos,bias)
        self.pesos_bias = self.pesos_bias_original.copy()

    def inicializa_camadas(self,pesos, bias):
        pesos_bias = {}
        pesosPredefinidos = pesos
        biasPredefinidos = bias

        for indice_camada, camada in enumerate(self.arquitetura):
            indice_camada_atual = indice_camada + 1
            tamanho_entradas = camada['dim_entrada']
            tamanho_saidas = camada['dim_saida']

            if pesosPredefinidos is None:
                pesos = self.rng.randn(tamanho_saidas,tamanho_entradas)
            else:
                pesos = pesosPredefinidos[indice_camada]

            if biasPredefinidos is None:
                bias = np.zeros((tamanho_saidas,1))
            else:
                bias = biasPredefinidos[indice_camada]
            
            pesos_bias['P' + str(indice_camada_atual)] = pesos
            pesos_bias['B' + str(indice_camada_atual)] = bias
        
        return pesos_bias

    #Acrescentar aqui qualquer nova função de ativação acrescentada no activation_functions.py
    def A_atual_value(self,Z,ativado_nome):
        if ativado_nome == 'softplus':
            A_atual = softplus(Z)
        elif ativado_nome == 'relu':
            A_atual = relu(Z)
        elif ativado_nome == 'sigmoid':
            A_atual = sigmoid(Z)
        elif ativado_nome == 'None':
            A_atual = None_Ac(Z)
        else:
            raise Exception('Ainda não implementamos essa funcao: ' + ativado_nome)

        return Z,A_atual

    #Acrescentar aqui qualquer nova função de ativação acrescentada no activation_functions.py
    def dAtivado_value(self,funcao_ativacao_atual,Z):
        if funcao_ativacao_atual == "None":
            dAtivado = None_Ac_derivada(Z)
        elif funcao_ativacao_atual == "softplus":
            dAtivado = softplus_derivada(Z)
        elif funcao_ativacao_atual == "relu":
            dAtivado = relu_derivada(Z)
        elif funcao_ativacao_atual == "sigmoid":
            dAtivado = sigmoid_derivada(Z)
        else:
            raise Exception('Ainda não implementamos essa funcao: ' + funcao_ativacao_atual)
        
        return dAtivado

    #Acrescentar aqui qualquer nova função de loss acrescentada no loss_functions.py
    def loss_derivate_value(self,predict,correct,loss_name):
        if loss_name == 'ssr':
            loss = SSR_derivada(predict,correct) * 1
        elif loss_name == 'cross_entropy':
            loss = cross_entropy_derivada(predict,correct) * 1
        else:
            raise Exception('Ainda não implementamos essa funcao: ' + loss_name)
        return loss

    #Acrescentar aqui qualquer nova função de loss acrescentada no loss_functions.py
    def loss_value(self,predict,correct,loss_name):
        if loss_name == 'ssr':
            loss = SSR(predict,correct)
        elif loss_name == 'cross_entropy':
            loss = cross_entropy(predict,correct)
        else:
            raise Exception('Ainda não implementamos essa funcao: ' + loss_name)
        return loss

    def propaga_uma_camada(self,pesos,bias,A_antigo,camada):  
        Z = np.dot(pesos,A_antigo) + bias 
        return self.A_atual_value(Z,camada['activacao'])
           
    def propaga_todas_camadas(self,input):
        historico ={}
        A_atual = input 
        for indice_camada_anterior, camada  in enumerate(self.arquitetura):
            A_antigo = A_atual 
            indice_camada_atual = indice_camada_anterior + 1
            pesos = self.pesos_bias['P' + str(indice_camada_atual)]
            bias  = self.pesos_bias['B' + str(indice_camada_atual)]
            Z,A_atual = self.propaga_uma_camada(pesos,bias,A_antigo,camada)

            historico['A_antigo'+str(indice_camada_anterior)] = A_antigo
            historico['Z' + str(indice_camada_atual)] = Z
        
        output = A_atual 

        return output, historico

    def backpropagation_local(self,derivadas_acumuladas_antiga, pesos_atual, bias_atual, Z, A_antigo, funcao_ativacao_atual):
        #m = A_antigo.shape[1]
        m = 1
        dAtivado = self.dAtivado_value(funcao_ativacao_atual,Z)

        derivadas_acumuladas_atual = (derivadas_acumuladas_antiga * dAtivado)

        #Att bias
        dBias = np.sum(derivadas_acumuladas_atual*1, axis=1, keepdims=True)/m

        #Att pesos
        dPesos = np.dot(derivadas_acumuladas_atual, A_antigo.T)/m
        
        acumulado_novo = np.dot(pesos_atual.T, derivadas_acumuladas_atual) # o acumulado final tem que ter a mesma quantidade de neuronios da camada anterior, isso é o valor da coluna de pesos_atual(que é a quantiodade de inputs)

        return acumulado_novo,dPesos,dBias
    
    def backpropagation_total(self,historico, correct,predict, loss_name):
        gradientes = {}
        dAtivado_anterior = self.loss_derivate_value(predict,correct,loss_name)

        for index_camada_anterior,camada in reversed(list(enumerate(self.arquitetura))):
            index_camada_atual = 1 + index_camada_anterior

            funcao_ativacao_atual = camada["activacao"]

            dAtivado_atual = dAtivado_anterior
            
            Z = historico['Z'+str(index_camada_atual)]
            
            A_antigo = historico['A_antigo'+str(index_camada_anterior)]
            
            pesos_atual = self.pesos_bias["P" + str(index_camada_atual)]
            bias_atual = self.pesos_bias["B" + str(index_camada_atual)]
            

            dAtivado_anterior, dPesos,dBias = self.backpropagation_local(dAtivado_atual, pesos_atual,
                                bias_atual, Z,
                                A_antigo, funcao_ativacao_atual)

            gradientes["dP" + str(index_camada_atual)] = dPesos
            gradientes["db" + str(index_camada_atual)] = dBias
            
            
        return gradientes
    
    def atualiza_rede(self, gradidentes, taxa_aprendizagem):
        for index_camada_anterior,camada in reversed(list(enumerate(self.arquitetura))):
            index_camada_atual = 1 + index_camada_anterior
            dPesos = gradidentes["dP" + str(index_camada_atual)]
            dBias  = gradidentes["db" + str(index_camada_atual)]

            #Step size
            peso_step_size = dPesos * taxa_aprendizagem
            bias_step_size = dBias  * taxa_aprendizagem
            
            #new value
            new_bias = self.pesos_bias["B" + str(index_camada_atual)] - bias_step_size
            new_peso = self.pesos_bias["P" + str(index_camada_atual)] - peso_step_size.reshape(self.pesos_bias["P" + str(index_camada_atual)].shape )

            
            self.pesos_bias["B" + str(index_camada_atual)]=new_bias
            self.pesos_bias["P" + str(index_camada_atual)]=new_peso
        
        return self.pesos_bias
    
    def acc_value(self,Y,Y_predict):
        acc = 0
        diff = []
        for a,b in zip(Y[0],Y_predict[0]):
            if a == b:
                acc+=1
            diff.append(a-b)
        return acc/Y.shape[1],diff

    def treino_stoc(self,X,Y,loss_name, taxa_aprendizagem, epocas, batch_size, valor_print = 100):
        training_data = list(zip(X.T,Y[0]))
        n = len(training_data)
        print(n)
        loss_historico = []
        diff_historico = []
        acc_historico = []

        for i in range(epocas):
            random.shuffle(training_data)
            mini_batchs = [training_data[k:k+batch_size] for k in range(0,n,batch_size)]
            for batch in mini_batchs:
                X_,Y_ = zip(*batch)
                X_ = np.array(X_).T
                Y_ = np.array([Y_])
                Y_predict,historico = self.propaga_todas_camadas(X_)

                gradientes = self.backpropagation_total(historico, Y_, Y_predict, loss_name) 

                self.pesos_bias = self.atualiza_rede(gradientes,taxa_aprendizagem)
            
            Y_predict = self.predict(X)
            loss_v = np.sum(self.loss_value(Y_predict,Y, loss_name))
            acc,diff = self.acc_value(Y,Y_predict)
                
            loss_historico.append(loss_v)
            acc_historico.append(acc)
            diff_historico.append(diff)
            if(i % valor_print == 0):
                print("Iteração: {0} - loss: {1} - acc: {2} ".format(i, loss_v,acc))


        return loss_historico,self.pesos_bias,acc_historico,diff_historico

    def treino(self,X,Y,loss_name, taxa_aprendizagem,epocas,valor_print = 100):

        loss_historico = []
        diff_historico = []
        acc_historico = []
        for i in range(epocas):
            Y_predict,historico = self.propaga_todas_camadas(X)

            gradientes = self.backpropagation_total(historico, Y, Y_predict, loss_name) 

            self.pesos_bias = self.atualiza_rede(gradientes,taxa_aprendizagem)

            loss_v = np.sum(self.loss_value(Y_predict,Y, loss_name))
            loss_historico.append(loss_v)

            acc,diff = self.acc_value(Y,Y_predict)
            acc_historico.append(acc)
            diff_historico.append(diff)


            if(i % valor_print == 0):
                print("Iteração: {0} - loss: {1} - acc: {2} ".format(i, loss_v,acc))
                #print(gradientes)

        return loss_historico,self.pesos_bias,acc_historico,diff_historico
    
    def predict(self,input):
        return self.propaga_todas_camadas(input)[0]
    
    def salva_pesos(self):
        pass

    def carrega_pesos(self):
        pass