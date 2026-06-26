# Relatório do Projeto Final do 2º bimestre - Programação concorrente e distribuída

# Identificador de fotos de rosto fake paralelo

**Disciplina: Programação Concorrente e Distribuída**<br>
**Aluno(s): Daniel Lohan Costa e Silva Dourado e Naylanne Lissa Gomes Cunha**<br>
**Turma: SIN04M1**<br>
**Professor: Rafael Marconi Ramos**<br>
**Data: 26/06/2026**

---

## 1. Descrição do Problema

Foi desenvolvido um programa para realizar a identificação de imagens de rostos, classificando-as como reais ou falsas, utilizando um modelo de aprendizado de máquina treinado com TensorFlow/Keras. O objetivo principal do experimento foi aplicar paralelismo no processamento de imagens, reduzindo o tempo total de execução em comparação com a versão serial.

O problema consiste em processar uma grande quantidade de imagens, realizar o pré-processamento de cada uma delas e, em seguida, executar a inferência no modelo treinado. Cada imagem passa por etapas como leitura do arquivo, conversão de cores, redimensionamento para o tamanho esperado pelo modelo e predição da classe correspondente.

Para explorar o paralelismo, foi utilizado um algoritmo baseado em divisão de dados, onde o conjunto total de imagens é dividido em partes menores, chamadas de fatias e processadas por múltiplos processos utilizando a biblioteca concurrent.futures, por meio da classe ProcessPoolExecutor. Cada processo realiza a leitura das imagens, o pré-processamento e a predição utilizando o modelo carregado com TensorFlow/Keras.

Foram utilizadas 20.000 imagens, separadas entre as classes REAL e FAKE, com o objetivo de avaliar o desempenho da paralelização em um volume significativo de dados.

A complexidade do processamento na versão sequencial pode ser considerada aproximadamente O(n), onde n representa o número total de imagens. Na versão paralela, em um cenário ideal, essa complexidade tende a se aproximar de O(n/p), onde p representa o número de processos utilizados. No entanto, na prática, o desempenho não cresce de forma perfeitamente proporcional devido ao overhead de paralelização, disputa por recursos de hardware, leitura de arquivos, carregamento do modelo e funcionamento interno do TensorFlow.

---

## 2. Ambiente Experimental

Os testes foram realizados em um computador com processador Intel Core i7 de 12ª geração, sistema operacional Windows 11 e 16 GB de memória RAM. O programa foi implementado em Python, utilizando TensorFlow/Keras para carregamento do modelo de classificação, OpenCV para leitura e pré-processamento das imagens e ProcessPoolExecutor para execução paralela com múltiplos processos.

| Item                        |      Descrição     |
| --------------------------- |      ---------     |
| Processador                 |      i7-12700      |
| Número de núcleos           |         12         |
| Memória RAM                 |        16gb        |
| Sistema Operacional         |      Windows 11    |
| Linguagem utilizada         |       Python       |
| Biblioteca de paralelização | concurrent.futures |
| Bibliotecas auxiliar        |  TensorFlow/Keras  |
| Bibliotecas auxiliar        |       OpenCV       |
| Bibliotecas auxiliar        |        NumPy       |
| Compilador / Versão         |   VScode 1.126.0   |

---

## 3. Metodologia de Testes

Os experimentos foram conduzidos com o objetivo de avaliar o desempenho do programa em diferentes quantidades de processos. O tempo de execução foi medido considerando o processamento completo das imagens na versão serial, incluindo leitura, pré-processamento e inferência pelo modelo.

Na versão paralela, o conjunto de imagens foi dividido em fatias. Cada fatia foi enviada para um processo, que executou a mesma função de processamento sobre aquele subconjunto de imagens. Dentro de cada processo, as imagens foram agrupadas em batches internos para melhorar o desempenho da inferência.

O código utilizou a seguinte estratégia:

Carregamento dos caminhos das imagens;
Divisão das imagens em fatias;
Distribuição das fatias entre processos com ProcessPoolExecutor;
Carregamento do modelo TensorFlow/Keras dentro de cada processo;
Leitura e pré-processamento das imagens com OpenCV;
Predição em batches;
Agrupamento dos resultados no processo principal;
Cálculo do tempo total, taxa de acerto e precisão.

### As configurações testadas foram:

Configuração	Descrição
1 processo	    Execução serial, usada como base de comparação
2 processos	    Execução paralela com 2 processos
4 processos	    Execução paralela com 4 processos
8 processos	    Execução paralela com 8 processos
12 processos	Execução paralela com 12 processos

O tempo serial foi utilizado como referência para o cálculo do speedup. A partir dele, foi possível avaliar o ganho de desempenho obtido com o aumento do número de processos.

---

## 4. Resultados Experimentais

A tabela a seguir apresenta os tempos de execução obtidos para cada quantidade de processos utilizada.

| Nº Threads/Processos | Tempo de Execução (s) |
| -------------------- | --------------------- |
| 1                    |         1480          |
| 2                    |          842          |
| 4                    |          509          |
| 8                    |          303          |
| 12                   |          271          |


Observa-se que o tempo total de execução diminuiu conforme o número de processos aumentou. A versão serial levou 1480 segundos, enquanto a execução com 12 processos reduziu o tempo para 271 segundos.

No entanto, a redução do tempo não ocorreu de forma perfeitamente proporcional ao aumento do número de processos, indicando a presença de overhead e limitações práticas da paralelização.

---

## 5. Cálculo de Speedup e Eficiência

Fórmula do Speedup

O speedup indica quantas vezes a versão paralela foi mais rápida em relação à versão serial.

Speedup(p) = T(1) / T(p)

Onde:

T(1) = tempo da execução serial;
T(p) = tempo da execução com p processos.
Fórmula da Eficiência

A eficiência indica o quanto os processos foram bem aproveitados em relação ao speedup ideal.

Eficiência(p) = Speedup(p) / p

Onde:

p = número de processos utilizados.

Para apresentar em porcentagem:

Eficiência (%) = (Speedup(p) / p) × 100

## 6. Tabela de Resultados


| Threads/Processos | Tempo (s) | Speedup | Eficiência |
| ----------------- | --------- | ------- | ---------- |
| 1                 |   1480    |   1.0   |    100     |
| 2                 |    842    |   1.8   |    088     |
| 4                 |    509    |   2.9   |    073     |
| 8                 |    303    |   4.9   |    061     |
| 12                |    271    |   5.5   |    046     |

Os resultados mostram que houve ganho de tempo em todas as versões paralelas em comparação com a versão serial. Entretanto, a eficiência diminuiu conforme o número de processos aumentou.

---

## 7. Gráfico de Tempo de Execução

Gráfico mostrando o **tempo de execução em função do número de processos**.


![Gráfico Tempo Execução](img/tempo_execucao.png)


A tendência esperada é uma curva decrescente, mostrando que o tempo total diminui com o aumento da quantidade de processos.

---

## 8. Gráfico de Speedup

Gráfico mostrando o **speedup obtido**.

Eixo X: número de processos
Eixo Y: speedup


![Gráfico Speedup](img//speedup.png)


A comparação entre as duas linhas permite observar o quanto o desempenho real se aproxima ou se distancia do desempenho teórico esperado.

---

## 9. Gráfico de Eficiência

Gráfico mostrando a **eficiência da paralelização**.

Eixo X: número de processos
Eixo Y: eficiência


![Gráfico Eficiência](img/eficiencia.png)


A eficiência apresenta queda conforme o número de processos aumenta, o que indica que o ganho de desempenho não cresce na mesma proporção que a quantidade de processos utilizados.

---

## 10. Análise dos Resultados

Os resultados obtidos demonstram que o uso de paralelismo trouxe ganho real de desempenho em relação à execução serial. O tempo total foi reduzido de 1480 segundos, na execução com 1 processo, para 271 segundos, na execução com 12 processos.

Com 2 processos, o tempo caiu para 842 segundos, resultando em speedup de aproximadamente 1,8 e eficiência de 88%. Esse resultado ficou relativamente próximo do ideal, indicando bom aproveitamento dos processos.

Com 4 processos, o tempo foi reduzido para 509 segundos, com speedup de 2,9 e eficiência de 73%. Esse resultado mostra um bom equilíbrio entre redução de tempo e aproveitamento dos recursos, sendo considerado o maior ponto de eficiência da análise.

Com 8 processos, o tempo caiu para 303 segundos, resultando em speedup de 4,9 e eficiência de 61%. Embora a eficiência tenha diminuído em relação à execução com 4 processos, ainda houve um ganho real significativo no tempo absoluto, pois o tempo foi reduzido de 509 segundos para 303 segundos, uma diferença de 206 segundos. Porém o speedup ficou muito distante do ideal, o que faz com que esse não seja o melhor cenário.

Com 12 processos, o menor tempo absoluto foi alcançado, com 271 segundos. Entretanto, o ganho em relação à execução com 8 processos foi de apenas 32 segundos, mesmo utilizando 4 processos a mais. Além disso, a eficiência caiu para 46%, indicando que o aumento no número de processos passou a gerar menor retorno proporcional.

Dessa forma, é possível observar dois fatores: apesar do menor tempo total ter sido obtido com 12 processos o custo foi muito alto e a eficiência caiu bastante, enquanto que o equilíbrio entre speedup, eficiência e custo-benefício foi obtido com até 4 processos, sendo o melhor resultado prático.

A partir de 8, o programa ainda apresentou redução no tempo total, mas com perda progressiva de eficiência. Isso mostra que o paralelismo trouxe ganhos cada vez menores.

Uma das principais causas para essa queda de eficiência está relacionada ao uso do TensorFlow em conjunto com multiprocessamento, aumentando o consumo de memória RAM e a disputa por recursos da máquina.

Além disso, o TensorFlow já possui mecanismos internos de paralelismo para executar operações matemáticas pesadas, como convoluções e multiplicações de matrizes. Ao utilizar vários processos externos ao mesmo tempo, pode ocorrer sobreposição de paralelismo, gerando contenção por CPU, memória, cache e leitura de disco.

Outro fator que influencia o desempenho é a leitura das imagens com cada processo realizando leitura de arquivos, conversão de cores, redimensionamento e transformação para float32, o aumento da quantidade de processos também pode gerar disputa por acesso ao disco e maior uso de memória.

Portanto, a queda de eficiência não indica necessariamente erro na implementação. Ela demonstra uma limitação prática do paralelismo, em que o aumento no número de processos deixa de trazer ganho proporcional devido ao overhead de gerenciamento, carregamento do modelo, comunicação entre processos e contenção de recursos de hardware.

---

## 11. Conclusão

Com base nos resultados obtidos, conclui-se que a aplicação de paralelismo no identificador de fotos de rosto fake trouxe ganho significativo de desempenho em relação à execução serial.

A versão serial levou 1480 segundos para processar as imagens, enquanto a versão paralela com 12 processos reduziu esse tempo para 271 segundos. Isso demonstra que a divisão do trabalho entre múltiplos processos contribuiu para acelerar o processamento.

Entretanto, o speedup obtido não foi ideal. Em um cenário teórico perfeito, 12 processos poderiam alcançar speedup próximo de 12. Porém, o speedup obtido foi de aproximadamente 5,5. Isso mostra que o programa apresentou escalabilidade parcial, mas não linear.

O melhor equilíbrio entre desempenho e eficiência foi observado com 4 processos. Nessa configuração, o tempo foi reduzido para 509 segundos, com speedup de 2,9 e eficiência de 73%. Esse resultado indica bom aproveitamento dos recursos, mantendo uma eficiência ainda relativamente alta.

A execução com 8 processos também apresentou ganho real relevante, reduzindo o tempo de 509 segundos para 303 segundos. Porém, a eficiência caiu para 61%. Já com 12 processos, apesar de ter sido obtido o menor tempo total, o ganho adicional em relação a 8 processos foi pequeno, apenas 32 segundos, e a eficiência caiu para 46%.

Dessa forma, pode-se concluir que o aumento do número de processos nem sempre resulta no melhor custo-benefício. Para este trabalho, 4 processos representaram o melhor ponto de equilíbrio técnico, enquanto 12 processos representaram o menor tempo absoluto.

A perda de eficiência pode ser explicada pelo overhead de paralelização, pela leitura simultânea de arquivos, pelo carregamento de uma instância do modelo TensorFlow em cada processo e pela disputa por recursos de CPU, memória e cache. Além disso, como o TensorFlow já realiza paralelismo interno em suas operações, o uso de muitos processos externos pode gerar contenção e reduzir o ganho proporcional.

De forma geral, o experimento demonstrou na prática os benefícios e limitações do paralelismo. O trabalho evidenciou que a paralelização pode reduzir significativamente o tempo de execução, mas que o ganho depende dos recursos disponíveis, do tipo de tarefa executada e dos custos adicionais envolvidos no gerenciamento dos processos.