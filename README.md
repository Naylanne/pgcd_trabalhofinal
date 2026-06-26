# Projeto do 2º bimestre - Programação concorrente e distribuída

# Evolução do trabalho:<br>
. Tema - Indentificador de fotos de rosto fake paralelo<br>
. Objetivo - paralelizar um identificador de imagens de rostos reais ou fakes, a fim de agilizar esse processo.<br>
. Base de dados - https://www.kaggle.com/datasets/troykueh/real-vs-fake-faces-stylegan3<br>
. Treinamento do modelo - Foi utilizado o código que já consta na base de dados.<br>
. 1. Primeiro, foi utilizado o código de treinamento já disponibilizado junto à base de dados do Kaggle. Esse treinamento teve como objetivo ensinar o modelo a diferenciar imagens de rostos reais e imagens de rostos fake.<br>
. 2. Após o treinamento, o modelo foi salvo em um arquivo no formato .keras. Esse arquivo passou a representar o modelo já treinado, pronto para ser utilizado na classificação das imagens.<br>
. 3. No código do projeto, o modelo salvo foi carregado utilizando o TensorFlow/Keras. Dessa forma, não foi necessário treinar o modelo novamente durante os testes, apenas carregar o arquivo já treinado.<br>
. 4. Em seguida, o programa percorreu a pasta de imagens de teste, separadas entre imagens reais e imagens fake.<br>
. 5. Antes de enviar cada imagem para o modelo, foi necessário preparar a imagem. Para isso, a imagem foi lida com OpenCV, convertida para o padrão de cores RGB, redimensionada para o tamanho esperado pelo modelo e transformada para o formato numérico float32.<br>
. 6. Depois do pré-processamento, a imagem foi enviada ao modelo para que ele realizasse a predição. O modelo retornava um resultado indicando se a imagem era classificada como real ou fake.<br>
. 7. O resultado previsto pelo modelo foi comparado com a classe real da imagem, de acordo com a pasta em que ela estava armazenada. Assim, foi possível contabilizar os acertos e erros.<br>
. 8. Primeiro, esse processo foi realizado de forma sequencial, analisando uma imagem por vez. Depois, foi criada uma versão paralela, na qual as imagens foram divididas entre vários processos para tentar reduzir o tempo total de execução.<br>
.9. Por fim, foram avaliados o tempo de execução, a quantidade de imagens processadas, a taxa de acerto, o speedup e a eficiência da versão paralela em comparação com a versão sequencial.<br>

# Como Executar:<br>
1. Iniciar ambiente virtual: .venv\Scripts\Activate.ps1
2. Instalar as dependências do projeto: pip install -r requirements.txt<br>
3. Verificar se o modelo treinado está na pasta correta: model/face_detector_model.keras<br>
4. Verificar se as imagens de teste estão na pasta correta: teste_imagens/<br>
5. A estrutura esperada é:<br>
teste_imagens/
├── real/
└── fake/<br>
6. Executar a versão sequencial: python serial.py<br>
7. Executar a versão paralela informando a quantidade de processos: Ex: python paralelo.py 2<br>
8. Testando as outras quantidades de processos:<br>
python paralelo.py 4
python paralelo.py 8
python paralelo.py 12
9. Ao final da execução, o programa exibirá os resultados no terminal, como:<br>
Total de imagens encontradas
Total de imagens processadas
Acertos
Erros
Taxa de acerto
Tempo total de execução
Velocidade em imagens por segundo
Speedup
Eficiência
10. Para sair do ambiente virtual: deactivate