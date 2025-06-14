import os
import sys
import requests
import zipfile
from tqdm import tqdm
import tensorflow as tf

# Diretórios
BASE_DIR = r"C:\Users\leand\OneDrive\Desktop\Matthew\NSFW_Detector"
MODEL_FILE = os.path.join(BASE_DIR, "nsfw_model.h5")
os.makedirs(BASE_DIR, exist_ok=True)

def download_file(url, filepath):
    """Baixa um arquivo de uma URL com barra de progresso"""
    print(f"Baixando de {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    with open(filepath, 'wb') as file, tqdm(
            desc=filepath,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

# URL do modelo (link direto para o arquivo .h5)
MODEL_URL = "https://github.com/GantMan/nsfw_model/releases/download/1.1.0/nsfw_mobilenet2.224x224.h5"

# Baixar o modelo
try:
    print(f"Baixando o modelo NSFW para {MODEL_FILE}...")
    download_file(MODEL_URL, MODEL_FILE)
    print("Download concluído!")
except Exception as e:
    print(f"Erro ao baixar o modelo: {e}")
    sys.exit(1)

# Verificar se o modelo é válido tentando carregá-lo
try:
    print("Verificando o modelo...")
    model = tf.keras.models.load_model(MODEL_FILE)
    print("Modelo verificado e carregado com sucesso!")
    print(f"Formato de entrada do modelo: {model.input_shape}")
    print(f"Formato de saída do modelo: {model.output_shape}")
except Exception as e:
    print(f"Erro ao verificar o modelo: {e}")
    print("O arquivo baixado pode estar corrompido. Tente baixar manualmente do GitHub.")
    sys.exit(1)

print("\nConfigurações concluídas!")
print(f"Modelo salvo em: {MODEL_FILE}")