import random, copy

class Perceptron:

    def __init__(self, amostras, saidas, taxa_aprendizado=0.1, epocas=1000, limiar=-1):

        self.amostras = amostras #todas as amostras
        self.saidas = saidas
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.limiar = limiar
        self.num_amostras = len(amostras)   #qtd de amostras
        self.num_amostra = len(amostras[0]) #qtd de elementos por amostras
        self.pesos = []                     #vetor de pesos

    #Função de ativação: degrau bipolar
    def sinal(self, u):
        return 1 if u >= 0 else -1


    #Função para treinamento da rede:
    def treinar(self):

            #Adiciona o -1 para cada uma das amostras
            for amostra in self.amostras:
                amostra.insert(0, -1)

            #Inicia o vetor de pesos com números aleatórios
            for i in range(self.num_amostra + 1):
                self.pesos.append(random.random())

            #Inicia o contador de espocas
            num_epocas = 0

            while True:

                erro = False #erro do programa

                for i in range(self.num_amostras):

                    u = 0

                    #Inicia o somatŕodio
                    for j in range(self.num_amostra + 1):
                        u = u + self.pesos[j] * self.amostras[i][j]

                    #Obtem a saída da função de ativação
                    y = self.sinal(u)

                    if y != self.saidas[i]:

                        #Calcula o erro entre a saída desejada e a saída da rede
                        erro_aux = self.saidas[i] - y

                        #Realiza o ajuste dos pesos da rede:
                        for j in range(self.num_amostra + 1):
                            self.pesos[j] = self.pesos[j] + self.taxa_aprendizado * erro_aux * self.amostras[i][j]

                        erro = True


                self.epocas += 1

                if num_epocas > self.epocas or not erro:
                    break


    # função utilizada para testar a rede
    # recebe uma amostra a ser classificada e os nomes das classes
    # utiliza a função sinal, se é -1 então é classe1, senão é classe2

    def testar(self, amostra, classe1, classe2):

        # insere o -1
        amostra.insert(0, -1)

        # utiliza o vetor de pesos que foi ajustado na fase de treinamento
        u = 0
        for i in range(self.num_amostra + 1):
            u += self.pesos[i] * amostra[i]

        # calcula a saída da rede
        y = self.sinal(u)

        # verifica a qual classe pertence
        if y == -1:
            print('A amostra pertence a classe %s' % classe1)
        else:
            print('A amostra pertence a classe %s' % classe2)


print('\nA ou B?\n')

# amostras: um total de 4 amostras

#     Exemplo: Num de cigarros | Situação Pulmonar | Historico de Doencas
amostras = [[      0.1,         0.4,          0.2],
            [      0.3,         0.7,          0.5],
            [      0.6,         0.9,          0.5],
            [      0.5,         0.7,          0.3]]

# saídas desejadas de cada amostra
saidas = [1, -1, -1, 1, -1]
    #  1: Acontece
    # -1: Não acontece

# conjunto de amostras de testes
testes = copy.deepcopy(amostras)

# cria uma rede Perceptron
rede = Perceptron(amostras=amostras, saidas=saidas,
                  taxa_aprendizado=0.1, epocas=1000)

# treina a rede
rede.treinar()

# testando a rede
for teste in testes:
    rede.testar(teste, 'A', 'B')
