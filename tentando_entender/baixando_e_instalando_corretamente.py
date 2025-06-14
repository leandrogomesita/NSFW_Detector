import os
import sys
import zipfile
import shutil
import subprocess
import time

# Caminhos de diretórios e arquivos
ZIP_FILE = r"C:\\Users\\leand\\OneDrive\\Desktop\\Matthew\\nsfw_model-master.zip"
EXTRACT_DIR = r"C:\\Users\\leand\\OneDrive\\Desktop\\Matthew\\nsfw_model"
MODEL_DIR = r"C:\\Users\\eand\\OneDrive\\Desktop\\Matthew\\models"

# Verificar se o arquivo ZIP existe
if not os.path.exists(ZIP_FILE):
    print(f"Erro: O arquivo {ZIP_FILE} não foi encontrado.")
    sys.exit(1)

# Criar diretórios
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Limpar diretórios existentes
print("Limpando diretórios existentes...")
if os.path.exists(EXTRACT_DIR):
    for item in os.listdir(EXTRACT_DIR):
        item_path = os.path.join(EXTRACT_DIR, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

# Extrair o ZIP
print(f"Extraindo {ZIP_FILE}...")
with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(EXTRACT_DIR))

# Renomear diretório extraído se necessário
zip_extract_dir = r"C:\Users\leand\OneDrive\Desktop\Matthew\nsfw_model-master"
if os.path.exists(zip_extract_dir) and zip_extract_dir != EXTRACT_DIR:
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    shutil.move(zip_extract_dir, EXTRACT_DIR)
    print(f"Renomeado diretório para {EXTRACT_DIR}")

# Verificar se a extração foi bem-sucedida
if not os.path.exists(EXTRACT_DIR):
    print("Erro: Falha na extração do arquivo ZIP.")
    sys.exit(1)

# Instalar dependências
print("\nInstalando dependências...")
subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow", "requests", "pillow", "opencv-python", "mss"])
time.sleep(2)  # Pequena pausa para garantir que a instalação terminou

# Criar script para baixar o modelo
download_script_path = os.path.join(EXTRACT_DIR, "download_model.py")
with open(download_script_path, 'w') as f:
    f.write('''
import os
import sys
import requests
from tqdm import tqdm

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

# Diretório para salvar o modelo
MODEL_DIR = r"''' + MODEL_DIR + r'''"
os.makedirs(MODEL_DIR, exist_ok=True)

# URL do modelo (link direto para o arquivo .h5)
MODEL_URL = "https://github.com/GantMan/nsfw_model/releases/download/1.1.0/nsfw_mobilenet2.224x224.h5"
MODEL_PATH = os.path.join(MODEL_DIR, "nsfw_mobilenet2.224x224.h5")

# Baixar o modelo
try:
    print(f"Baixando o modelo NSFW para {MODEL_PATH}...")
    download_file(MODEL_URL, MODEL_PATH)
    print("Download concluído!")
except Exception as e:
    print(f"Erro ao baixar o modelo: {e}")
    sys.exit(1)

print(f"\\nModelo salvo em: {MODEL_PATH}")
''')

# Baixar o modelo
print("\nBaixando o modelo...")
try:
    subprocess.run([sys.executable, download_script_path], check=True)
except subprocess.CalledProcessError:
    print("Erro ao baixar o modelo.")
    sys.exit(1)

# Criar script principal para executar o detector
main_script_path = os.path.join(EXTRACT_DIR, "run_nsfw_detector.py")
with open(main_script_path, 'w') as f:
    f.write('''
import os
import sys
import time
import cv2
import numpy as np
import mss
import winsound
import threading
import tensorflow as tf

# Configurações
NSFW_THRESHOLD = 0.4  # Limiar mais baixo para maior sensibilidade
BLUR_INTENSITY = 35   # Intensidade do blur
OPACITY = 0.9         # Opacidade da máscara de blur
WINDOW_NAME = "Detector de Conteúdo Adulto"
CAPTURE_INTERVAL = 0.5  # Intervalo entre capturas

# Caminho para o modelo
MODEL_PATH = r"''' + os.path.join(MODEL_DIR, "nsfw_mobilenet2.224x224.h5") + r'''"

# Categorias do modelo
CATEGORIES = ["drawings", "hentai", "neutral", "porn", "sexy"]

# Carregar o modelo
def load_model():
    """Carrega o modelo NSFW"""
    print(f"Carregando modelo NSFW de {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Modelo carregado com sucesso!")
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        sys.exit(1)

def predict_nsfw(model, img):
    """Prediz o conteúdo NSFW usando o modelo"""
    try:
        # Pré-processar a imagem para o formato esperado pelo modelo
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = np.array(img_rgb, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalizar
        
        # Fazer a predição
        predictions = model.predict(img_array, verbose=0)
        
        # Criar dicionário com as pontuações para cada categoria
        scores = dict(zip(CATEGORIES, predictions[0]))
        
        # Calcular pontuação NSFW combinada (soma de categorias não neutras)
        nsfw_score = scores.get('porn', 0) + scores.get('sexy', 0) + scores.get('hentai', 0)
        
        # Encontrar categoria com maior pontuação
        top_category = max(scores.items(), key=lambda x: x[1])[0]
        
        # Determinar se é NSFW
        is_nsfw = nsfw_score > NSFW_THRESHOLD
        
        return is_nsfw, nsfw_score, scores, top_category
        
    except Exception as e:
        print(f"Erro na predição NSFW: {e}")
        return False, 0.0, dict(zip(CATEGORIES, [0, 0, 1, 0, 0])), "neutral"

def play_warning_sound():
    """Reproduz um som de aviso quando conteúdo adulto é detectado"""
    # Padrão de alerta (mais sutil para uso constante)
    winsound.Beep(1200, 200)
    time.sleep(0.1)
    winsound.Beep(800, 300)

def apply_blur_to_image(image, intensity=15, opacity=0.7):
    """Aplica blur na imagem inteira com a opacidade especificada"""
    # Aplicar blur forte à imagem inteira
    blurred = cv2.GaussianBlur(image, (intensity, intensity), 0)
    
    # Combinar a imagem original com a borrada
    result = cv2.addWeighted(image, 1 - opacity, blurred, opacity, 0)
    
    return result

def draw_status_info(image, is_nsfw, nsfw_score, scores, top_category):
    """Desenha informações de status na imagem"""
    # Definir cores
    color = (0, 0, 255) if is_nsfw else (0, 255, 0)  # Vermelho se NSFW, verde se seguro
    
    # Criar uma área escura para o texto na parte superior
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Adicionar status principal
    status_text = "CONTEUDO ADULTO DETECTADO!" if is_nsfw else "Conteudo seguro"
    cv2.putText(image, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Adicionar pontuação NSFW
    cv2.putText(image, f"Pontuacao NSFW: {nsfw_score:.4f} (Limiar: {NSFW_THRESHOLD})", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Adicionar categoria principal
    cv2.putText(image, f"Categoria principal: {top_category}", 
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Mostrar pontuações detalhadas
    scores_text = " | ".join([f"{k}: {v:.2f}" for k, v in scores.items()])
    cv2.putText(image, scores_text, (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Adicionar aviso vermelho na parte inferior quando NSFW
    if is_nsfw:
        overlay = image.copy()
        cv2.rectangle(overlay, (0, h-40), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        cv2.putText(image, "ALERTA: CONTEUDO SENSIVEL DETECTADO", (w//2 - 200, h-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return image

def monitor_screen(model):
    """Monitora a tela e aplica blur em tempo real em conteúdo NSFW"""
    with mss.mss() as sct:
        # Obter as dimensões da tela
        monitor = sct.monitors[0]
        
        # Configurar janela
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        
        # Para evitar alertas sonoros repetidos
        last_alert_time = 0
        alert_cooldown = 3  # segundos entre alertas
        
        print("Monitoramento iniciado! Pressione 'q' para sair.")
        
        while True:
            try:
                start_time = time.time()
                
                # Capturar a tela
                screenshot = sct.grab(monitor)
                
                # Converter para um array NumPy
                img = np.array(screenshot)
                
                # Converter BGRA para BGR
                img = img[:, :, :3]
                
                # Redimensionar para melhorar desempenho
                height, width = img.shape[:2]
                scale_factor = 0.5  # Ajuste conforme necessário para desempenho
                resized = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))
                
                # Predizer conteúdo NSFW
                is_nsfw, nsfw_score, scores, top_category = predict_nsfw(model, resized)
                
                # Calcular intensidade de blur baseada na pontuação NSFW
                dynamic_intensity = int(BLUR_INTENSITY * min(2.0, 1.0 + nsfw_score))
                dynamic_opacity = min(1.0, OPACITY * (1.0 + nsfw_score * 0.5))
                
                # Aplicar blur se NSFW for detectado
                if is_nsfw:
                    result = apply_blur_to_image(resized, dynamic_intensity, dynamic_opacity)
                    
                    # Alertar (com cooldown)
                    current_time = time.time()
                    if current_time - last_alert_time > alert_cooldown:
                        threading.Thread(target=play_warning_sound).start()
                        last_alert_time = current_time
                else:
                    result = resized
                
                # Adicionar informações de status
                final_img = draw_status_info(result, is_nsfw, nsfw_score, scores, top_category)
                
                # Mostrar a imagem protegida
                cv2.imshow(WINDOW_NAME, final_img)
                
                # Calcular FPS e adicionar atraso dinâmico
                process_time = time.time() - start_time
                wait_time = max(1, int((CAPTURE_INTERVAL - process_time) * 1000))
                
                # Sair ao pressionar a tecla 'q'
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break
                
            except Exception as e:
                print(f"Erro no monitoramento: {e}")
                time.sleep(0.5)
    
    cv2.destroyAllWindows()
    print("Monitoramento encerrado.")

if __name__ == "__main__":
    print("Iniciando detector de conteúdo adulto...")
    print(f"Limiar de detecção: {NSFW_THRESHOLD} (mais baixo = mais sensível)")
    
    # Carregar modelo
    model = load_model()
    
    print("\\nModelo carregado. Iniciando monitoramento...")
    print("Pressione 'q' para sair")
    monitor_screen(model)
''')

print("\nConfiguração concluída!")
print("Arquivos criados:")
print(f"1. Script de download do modelo: {download_script_path}")
print(f"2. Script principal do detector: {main_script_path}")
print("\nPara executar o detector, use o comando:")
print(f"python {main_script_path}")