import os
import time
import cv2
import random
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tensorflow.keras.models import load_model

# ==========================
# CONFIGURAÇÕES
# ==========================

MODEL_PATH = "model/face_detector_model.keras"
IMAGE_FOLDER = "teste_imagens"
IMAGE_SIZE = (224, 224)
model = None

# ==========================
# CARREGAR MODELO
# ==========================

def carregar_modelo():
    return load_model(MODEL_PATH)

# ==========================
# ANÁLISE DA IMAGEM
# ==========================

def analisar_imagem(model, caminho_imagem):
    img = cv2.imread(caminho_imagem)

    if img is None:
        return None

    # OpenCV lê BGR → converter para RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # mesmo tamanho do treinamento
    img = cv2.resize(img, (224, 224))

    img = img.astype(np.float32)

    # adiciona dimensão batch
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0][0]

    classe_predita = "REAL" if prediction > 0.5 else "FAKE"

    classe_esperada = ("REAL" if "real" in caminho_imagem.lower() else "FAKE")
    
    correto = classe_predita == classe_esperada
    
    return {
        "imagem": os.path.basename(caminho_imagem),
        "classe_esperada": classe_esperada,
        "predicao": classe_predita,
        "confianca": float(prediction),
        "correto": correto}

# ==========================
# WORKER
# ==========================

def prever_imagem(caminho_imagem):
    global model

    # carrega modelo apenas 1x por processo
    if model is None:
        model = carregar_modelo()
    return analisar_imagem(model, caminho_imagem)

# ==========================
# MAIN
# ==========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("workers", type=int)

    args = parser.parse_args()

    extensoes = (".jpg", ".jpeg", ".png")
    imagens = []

    for raiz, _, arquivos in os.walk(IMAGE_FOLDER):
        for arquivo in arquivos:
            if arquivo.lower().endswith(extensoes):
                imagens.append(os.path.join(raiz, arquivo))

    total = len(imagens)

    print(f"Total de imagens: {total}")
    print("Executando versão paralela...")
    print(f"Quantidade de processos: {args.workers}")

    inicio = time.perf_counter()

    # ==========================
    # PARALELIZAÇÃO
    # ==========================

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        resultados = []

        for i, resultado in enumerate(executor.map(prever_imagem, imagens), start=1):
            print(f"Processando {i}/{total}: {resultado['imagem']}")
            resultados.append(resultado) 
            
    fim = time.perf_counter()
    
    tempo_paralelo = fim - inicio

    resultados = [r for r in resultados if r is not None]

    # ==========================
    # RESULTADOS
    # ==========================

    print("\n===== RESULTADOS =====")

    # pega 10 exemplos aleatórios
    exemplos = random.sample (resultados, min(10, len(resultados)))

    for r in exemplos:

        print(f"\nImagem: {r['imagem']}")
        print(f"Classe esperada: {r['classe_esperada']}")
        print(f"Predição: {r['predicao']}")
        print(f"Confiança: {r['confianca']:.2f}")
        print("✅ Correto" if r["correto"] else "❌ Errou")

    # ==========================
    # ESTATÍSTICAS
    # ==========================

    acertos = sum(r["correto"] for r in resultados)
    erros = total - acertos
    taxa = (acertos / total) * 100 if total > 0 else 0

    print("\n===== ESTATÍSTICAS =====")
    print(f"Total de imagens: {total}")
    print(f"Acertos: {acertos}")
    print(f"Erros: {erros}")
    print(f"Taxa de acerto: {taxa:.2f}%")

    print(f"\nTempo paralelo: {tempo_paralelo:.2f} segundos")
    print(f"Processos utilizados: {args.workers}")