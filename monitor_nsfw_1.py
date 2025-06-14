import os
import sys
import time
import cv2
import numpy as np
import mss
import winsound
import threading
import tensorflow as tf
import tensorflow_hub as hub
import win32gui
import win32con

# Configurações
MODEL_FILE = r"C:\\Users\\leand\\OneDrive\\Desktop\\Matthew\\NSFW_Detector\\model_1\\saved_model.h5"
NSFW_THRESHOLD = 0.4  # Limiar de detecção (mais baixo = mais sensível)
CAPTURE_INTERVAL = 0.5  # Intervalo entre capturas em segundos
WINDOW_NAME = "NSFW Info Monitor"
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 300

# Categorias do modelo
CATEGORIES = ["drawings", "hentai", "neutral", "porn", "sexy"]

# Função para carregar o modelo NSFW
def load_model():
    print(f"Carregando modelo NSFW de {MODEL_FILE}...")
    try:
        # Registrar a camada personalizada do TF Hub
        custom_objects = {'KerasLayer': hub.KerasLayer}
        model = tf.keras.models.load_model(MODEL_FILE, custom_objects=custom_objects)
        print("Modelo carregado com sucesso!")
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        sys.exit(1)

# Função para fazer predição NSFW
def predict_nsfw(model, img):
    try:
        # Pré-processar a imagem
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = np.array(img_rgb, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalizar
        
        # Fazer a predição
        predictions = model.predict(img_array, verbose=0)
        
        # Processar resultados
        scores = dict(zip(CATEGORIES, predictions[0]))
        nsfw_score = scores.get('porn', 0) + scores.get('sexy', 0) + scores.get('hentai', 0)
        top_category = max(scores.items(), key=lambda x: x[1])[0]
        is_nsfw = nsfw_score > NSFW_THRESHOLD
        
        return is_nsfw, nsfw_score, scores, top_category
    except Exception as e:
        print(f"Erro na predição: {e}")
        return False, 0.0, dict(zip(CATEGORIES, [0, 0, 1, 0, 0])), "neutral"

# Função para alertar quando conteúdo adulto for detectado
def play_warning_sound():
    winsound.Beep(1200, 200)
    time.sleep(0.1)
    winsound.Beep(800, 300)

# Função para criar a janela de informações
def create_info_window(is_nsfw, nsfw_score, scores, top_category):
    # Criar uma imagem em branco para exibir informações
    info_img = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 255  # Fundo branco
    
    # Definir cores
    color = (0, 0, 255) if is_nsfw else (0, 150, 0)  # Vermelho se NSFW, verde se seguro
    bg_color = (240, 240, 240)  # Cinza claro para o fundo
    
    # Preencher o fundo
    info_img[:] = bg_color
    
    # Adicionar barra superior colorida
    cv2.rectangle(info_img, (0, 0), (WINDOW_WIDTH, 50), color, -1)
    
    # Adicionar status principal
    status_text = "CONTEUDO ADULTO DETECTADO!" if is_nsfw else "Conteudo seguro"
    cv2.putText(info_img, status_text, (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Adicionar pontuação NSFW
    cv2.putText(info_img, f"Pontuacao NSFW: {nsfw_score:.4f} (Limiar: {NSFW_THRESHOLD})", 
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Adicionar categoria principal
    cv2.putText(info_img, f"Categoria principal: {top_category}", 
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Mostrar pontuações detalhadas
    y_pos = 150
    cv2.putText(info_img, "Pontuações por categoria:", 
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Mostrar cada categoria com barra de progresso
    for i, (category, score) in enumerate(scores.items()):
        y_pos = 180 + i * 25
        
        # Nome da categoria
        cv2.putText(info_img, f"{category}:", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Barra de progresso
        bar_length = int(score * 300)
        bar_color = (0, 0, 255) if category in ['porn', 'sexy', 'hentai'] and score > 0.15 else (0, 200, 0)
        cv2.rectangle(info_img, (120, y_pos-15), (120 + bar_length, y_pos-5), bar_color, -1)
        
        # Valor numérico
        cv2.putText(info_img, f"{score:.4f}", 
                   (430, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return info_img

# Função para obter a posição da janela de informação
def get_info_window_position():
    try:
        hwnd = win32gui.FindWindow(None, WINDOW_NAME)
        if hwnd:
            rect = win32gui.GetWindowRect(hwnd)
            x, y, w, h = rect
            return (x, y, w - x, h - y)
        return None
    except Exception as e:
        print(f"Erro ao obter posição da janela: {e}")
        return None

# Função principal para monitorar a tela
def monitor_screen():
    # Carregar modelo
    model = load_model()
    
    # Configurar captura de tela
    with mss.mss() as sct:
        # Configurar janela de informações
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Definir a posição da janela no canto superior direito
        screen_width = sct.monitors[0]["width"]
        info_window_x = screen_width - WINDOW_WIDTH - 20
        info_window_y = 50
        cv2.moveWindow(WINDOW_NAME, info_window_x, info_window_y)
        
        # Para evitar alertas sonoros repetidos
        last_alert_time = 0
        alert_cooldown = 3  # segundos entre alertas
        
        print("Monitoramento iniciado! Pressione 'q' para sair.")
        
        # Para o primeiro frame
        is_nsfw, nsfw_score, scores, top_category = False, 0.0, dict(zip(CATEGORIES, [0, 0, 1, 0, 0])), "neutral"
        
        while True:
            try:
                # Obter a posição atual da janela de informações
                info_window_pos = get_info_window_position()
                
                # Capturar a tela inteira
                screenshot = sct.grab(sct.monitors[0])
                img = np.array(screenshot)
                img = img[:, :, :3]  # Converter BGRA para BGR
                
                # Redimensionar para melhorar desempenho
                height, width = img.shape[:2]
                scale_factor = 0.5
                resized = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))
                
                # Se a janela de informações estiver visível, evitar analisar essa área
                if info_window_pos:
                    x, y, w, h = info_window_pos
                    # Escalar as coordenadas da janela para a imagem redimensionada
                    x_scaled = int(x * scale_factor)
                    y_scaled = int(y * scale_factor)
                    w_scaled = int(w * scale_factor)
                    h_scaled = int(h * scale_factor)
                    
                    # Verificar se as coordenadas estão dentro dos limites da imagem
                    if (x_scaled >= 0 and y_scaled >= 0 and 
                        x_scaled + w_scaled <= resized.shape[1] and 
                        y_scaled + h_scaled <= resized.shape[0]):
                        
                        # Substituir a área da janela de informações por um retângulo preto
                        resized[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled] = 0
                
                # Predizer conteúdo NSFW
                is_nsfw, nsfw_score, scores, top_category = predict_nsfw(model, resized)
                
                # Criar janela de informações
                info_window = create_info_window(is_nsfw, nsfw_score, scores, top_category)
                
                # Mostrar janela de informações
                cv2.imshow(WINDOW_NAME, info_window)
                
                # Alertar se conteúdo adulto for detectado (com cooldown)
                if is_nsfw:
                    current_time = time.time()
                    if current_time - last_alert_time > alert_cooldown:
                        threading.Thread(target=play_warning_sound).start()
                        last_alert_time = current_time
                
                # Sair ao pressionar a tecla 'q'
                if cv2.waitKey(int(CAPTURE_INTERVAL * 1000)) & 0xFF == ord('q'):
                    break
                
            except Exception as e:
                print(f"Erro no monitoramento: {e}")
                time.sleep(0.5)
    
    cv2.destroyAllWindows()
    print("Monitoramento encerrado.")

if __name__ == "__main__":
    print("Iniciando monitor de informações NSFW...")
    print(f"Limiar de detecção: {NSFW_THRESHOLD} (mais baixo = mais sensível)")
    print("Pressione 'q' para sair")
    monitor_screen()