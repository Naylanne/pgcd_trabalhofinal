import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import cv2
import random
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from tensorflow.keras.models import load_model

# ==================================================
# CONFIGURAÇÕES
# ==================================================
MODEL_PATH = "model/face_detector_model.keras"
IMAGE_FOLDER = "teste_imagens"
IMAGE_SIZE = (224, 224)

BATCH_INTERNO = 128
TAMANHO_FATIA = 512

QUANTIDADE_RESULTADOS_ALEATORIOS = 10
SEMENTE_ALEATORIA = None

model = None

# ==================================================
# FUNÇÕES AUXILIARES
# ==================================================
def carregar_modelo():
    return load_model(MODEL_PATH)

def obter_classe_esperada(caminho):
    pasta_pai = os.path.basename(os.path.dirname(caminho)).lower()

    if pasta_pai == "real":
        return "REAL"

    if pasta_pai == "fake":
        return "FAKE"

    return "DESCONHECIDA"

def carregar_imagens():
    extensoes = (".jpg", ".jpeg", ".png")
    imagens = []

    for raiz, _, arquivos in os.walk(IMAGE_FOLDER):
        for arquivo in arquivos:
            if arquivo.lower().endswith(extensoes):
                imagens.append(os.path.join(raiz, arquivo))

    imagens.sort()
    return imagens

def dividir_em_fatias(lista, tamanho_fatia):
    return [
        lista[i:i + tamanho_fatia]
        for i in range(0, len(lista), tamanho_fatia)
    ]

def mostrar_resultados_individuais(resultados, quantidade=10, semente=None):
    print("\n===== RESULTADOS POR IMAGEM - AMOSTRA ALEATÓRIA =====\n")

    if not resultados:
        print("Nenhum resultado disponível.")
        return

    if semente is not None:
        random.seed(semente)

    quantidade = min(quantidade, len(resultados))
    exemplos = random.sample(resultados, quantidade)

    print(f"{'Imagem':40} | {'Esperada':10} | {'Predição':15} | {'Resultado'}")
    print("-" * 85)

    for r in exemplos:
        imagem = r["imagem"]
        esperada = r["classe_esperada"]
        predicao = r["predicao"]
        confianca = r["confianca"]
        status = "✅ Correto" if r["correto"] else "❌ Errou"

        predicao_formatada = f"{predicao} ({confianca:.2f})"

        print(f"{imagem:40} | {esperada:10} | {predicao_formatada:15} | {status}")

    print(f"\nForam exibidas {quantidade} imagens aleatórias de {len(resultados)} processadas.")

# ==================================================
# PROCESSAMENTO DA FATIA NO WORKER
# ==================================================
def processar_fatia(fatia_caminhos, batch_interno):
    global model

    if model is None:
        model = carregar_modelo()

    resultados = []
    imagens_com_erro = 0

    for inicio in range(0, len(fatia_caminhos), batch_interno):
        lote_caminhos = fatia_caminhos[inicio:inicio + batch_interno]

        imagens_lote = []
        caminhos_validos = []
        classes_esperadas = []

        for caminho in lote_caminhos:
            try:
                img = cv2.imread(caminho)

                if img is None:
                    imagens_com_erro += 1
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMAGE_SIZE)
                img = img.astype(np.float32)

                imagens_lote.append(img)
                caminhos_validos.append(caminho)
                classes_esperadas.append(obter_classe_esperada(caminho))

            except Exception:
                imagens_com_erro += 1
                continue

        if not imagens_lote:
            continue

        X_batch = np.array(imagens_lote, dtype=np.float32)

        predicoes = model.predict(
            X_batch,
            batch_size=64,
            verbose=0
        )

        for idx, pred in enumerate(predicoes):
            p = float(pred[0]) if hasattr(pred, "__len__") else float(pred)

            classe_predita = "REAL" if p > 0.5 else "FAKE"
            classe_esperada = classes_esperadas[idx]

            correto = classe_predita == classe_esperada

            caminho_relativo = os.path.relpath(caminhos_validos[idx], IMAGE_FOLDER)

            resultados.append({
                "imagem": caminho_relativo,
                "classe_esperada": classe_esperada,
                "predicao": classe_predita,
                "confianca": p,
                "correto": correto
            })

    return {
        "resultados": resultados,
        "imagens_com_erro": imagens_com_erro,
        "avaliadas": len(fatia_caminhos)
    }

# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "workers",
        type=int,
        help="Quantidade de processos paralelos"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=BATCH_INTERNO,
        help="Tamanho do batch interno. Padrão: 128"
    )

    parser.add_argument(
        "--fatia",
        type=int,
        default=TAMANHO_FATIA,
        help="Quantidade de imagens por fatia. Padrão: 512"
    )

    args = parser.parse_args()

    imagens = carregar_imagens()
    total = len(imagens)

    if total == 0:
        print(f"Nenhuma imagem encontrada na pasta '{IMAGE_FOLDER}'.")
        exit()

    fatias = dividir_em_fatias(imagens, args.fatia)

    print("===== CONFIGURAÇÃO PARALELA =====")
    print()
    print(f"Modelo: {MODEL_PATH}")
    print(f"Pasta de imagens: {IMAGE_FOLDER}")
    print(f"Tamanho da imagem: {IMAGE_SIZE}")
    print(f"Total de imagens: {total}")
    print(f"Quantidade de processos: {args.workers}")
    print(f"Batch interno: {args.batch}")
    print(f"Tamanho da fatia: {args.fatia}")
    print(f"Total de fatias: {len(fatias)}")

    print("\nExecutando versão paralela com fatias e batches...\n")

    resultados = []
    imagens_com_erro = 0
    total_avaliadas = 0

    inicio = time.perf_counter()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futuros = [
            executor.submit(processar_fatia, fatia, args.batch)
            for fatia in fatias
        ]

        with tqdm(total=total, desc="Processando imagens", unit="img") as barra:
            for futuro in as_completed(futuros):
                retorno = futuro.result()

                resultados.extend(retorno["resultados"])
                imagens_com_erro += retorno["imagens_com_erro"]
                total_avaliadas += retorno["avaliadas"]

                barra.update(retorno["avaliadas"])

    fim = time.perf_counter()
    tempo_paralelo = fim - inicio

    # ==================================================
    # RESULTADOS INDIVIDUAIS
    # ==================================================
    mostrar_resultados_individuais(
        resultados,
        quantidade=QUANTIDADE_RESULTADOS_ALEATORIOS,
        semente=SEMENTE_ALEATORIA
    )

    # ==================================================
    # ESTATÍSTICAS
    # ==================================================
    total_processadas = len(resultados)
    acertos = sum(r["correto"] for r in resultados)
    erros = total_processadas - acertos

    taxa = (
        acertos / total_processadas * 100
        if total_processadas > 0
        else 0
    )

    tempo_medio = (
        tempo_paralelo / total_processadas
        if total_processadas > 0
        else 0
    )

    print("\n===== ESTATÍSTICAS PARALELO =====")
    print()
    print(f"Total de imagens encontradas: {total}")
    print(f"Total de imagens avaliadas: {total_avaliadas}")
    print(f"Total de imagens processadas: {total_processadas}")
    print(f"Imagens com erro de leitura: {imagens_com_erro}")
    print(f"Acertos: {acertos}")
    print(f"Erros: {erros}")
    print(f"Taxa de acerto: {taxa:.2f}%")
    print(f"Tempo TOTAL paralelo: {tempo_paralelo:.2f} segundos")
    print(f"Tempo médio por imagem: {tempo_medio:.4f} segundos")
    print(f"Processos utilizados: {args.workers}")

    if tempo_paralelo > 0:
        print(f"Velocidade: {total_processadas / tempo_paralelo:.2f} imagens/segundo")

    print("\n===== VALIDAÇÃO =====")
    print()

    if total_processadas == total:
        print(f"✅ Todas as {total} imagens foram processadas.")
    else:
        print(
            f"⚠️ Foram encontradas {total} imagens, "
            f"mas {total_processadas} foram processadas com predição."
        )
        print("Isso pode indicar imagem corrompida ou erro de leitura.")