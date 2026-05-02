import cv2
import numpy as np

# ─────────────────────────────────────────────
# Rangos HSV para cada clase (H: 0-179, S/V: 0-255)
# ─────────────────────────────────────────────

# Rojo: dos rangos porque el matiz rojo está en los extremos del círculo HSV
HSV_ROJO_LO1 = np.array([0,   80,  50])
HSV_ROJO_HI1 = np.array([10, 255, 255])
HSV_ROJO_LO2 = np.array([165, 80,  50])
HSV_ROJO_HI2 = np.array([179, 255, 255])

# Azul: cinta adhesiva azul brillante del suelo
HSV_AZUL_LO  = np.array([95, 80,  50])
HSV_AZUL_HI  = np.array([135, 255, 255])

# Colores de visualización por etiqueta (RGB)
COLOR_MARCA = np.array([255,   0,   0], dtype=np.uint8)   # rojo
COLOR_LINEA = np.array([  0,   0, 255], dtype=np.uint8)   # azul
COLOR_FONDO = np.array([  0, 200,   0], dtype=np.uint8)   # verde


def segmentar_frame_escenario1(frame_rgb, k_morph=3):

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    # Máscara roja (marca / flecha)
    m_rojo1 = cv2.inRange(hsv, HSV_ROJO_LO1, HSV_ROJO_HI1)
    m_rojo2 = cv2.inRange(hsv, HSV_ROJO_LO2, HSV_ROJO_HI2)
    mask_marca = cv2.bitwise_or(m_rojo1, m_rojo2)

    # Máscara azul (línea)
    mask_linea = cv2.inRange(hsv, HSV_AZUL_LO, HSV_AZUL_HI)

    # Limpieza morfológica (apertura = erosión + dilatación)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_morph, k_morph))
    mask_marca = cv2.morphologyEx(mask_marca, cv2.MORPH_OPEN, kernel)
    mask_linea = cv2.morphologyEx(mask_linea, cv2.MORPH_OPEN, kernel)

    # Imagen de etiquetas: 0=fondo, 1=marca, 2=línea
    etiquetas = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
    etiquetas[mask_marca > 0] = 1
    etiquetas[mask_linea > 0] = 2

    # Imagen coloreada para visualización
    imagen_coloreada = np.full_like(frame_rgb, COLOR_FONDO)
    imagen_coloreada[etiquetas == 1] = COLOR_MARCA
    imagen_coloreada[etiquetas == 2] = COLOR_LINEA

    return imagen_coloreada, etiquetas


# ==============================================================================
# ESCENARIO 2
# ==============================================================================
# Colores para las etiquetas de salida del Escenario 2 (BGR)
COLOR_FONDO_E2 = [0, 0, 0]         # Negro
COLOR_OBJETO_E2 = [0, 255, 255]    # Amarillo

def calcular_distancia_pinhole(radio_px, diametro_real_cm, focal_px):
    """
    Implementación de la fórmula de la transparencia 37: Z = (f * d) / (u1 - u2)
    u1 - u2 es el diámetro en píxeles (2 * radio_px)
    """
    if radio_px <= 0: return None
    diametro_px = 2 * radio_px
    distancia_z = (focal_px * diametro_real_cm) / diametro_px
    return distancia_z

def segmentar_esfera_hsv(frame_rgb, hsv_low, hsv_high):
    """
    Segmenta un objeto por color y devuelve la máscara y el frame etiquetado.
    """
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    
    # Limpieza de ruido (Morfología)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Crear imagen de etiquetas
    etiquetas = np.zeros_like(frame_rgb)
    etiquetas[mask == 0] = COLOR_FONDO_E2
    etiquetas[mask > 0] = COLOR_OBJETO_E2
    
    return mask, etiquetas

def segmentar_frame_escenario2(frame_rgb, hsv_low, hsv_high, d_real_cm, f_calibrada):
    """
    Segmenta el objeto esférico y calcula su distancia. Devuelve el frame etiquetado.
    """
    mask, frame_etiquetado = segmentar_esfera_hsv(frame_rgb, hsv_low, hsv_high)
    
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    distancia_cm = None
    if contornos:
        c = max(contornos, key=cv2.contourArea)
        ((x, y), radio) = cv2.minEnclosingCircle(c)
        
        if radio > 5:
            distancia_cm = calcular_distancia_pinhole(radio, d_real_cm, f_calibrada)
            
            # Anotar el frame
            cv2.circle(frame_etiquetado, (int(x), int(y)), int(radio), (0, 0, 255), 2)
            texto = f"Distancia: {distancia_cm:.2f} cm"
            cv2.putText(frame_etiquetado, texto, (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
    return frame_etiquetado

def procesar_video_completo(video_entrada, video_salida, escenario, params=None):
    """
    Procesa un vídeo completo usando el algoritmo del escenario indicado.
    escenario: 1 (Líneas y marcas) o 2 (Esfera y distancia)
    """
    cap = cv2.VideoCapture(video_entrada)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_salida, fourcc, fps, (width, height))

    print(f"Procesando vídeo del Escenario {escenario}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if escenario == 1:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resultado, _ = segmentar_frame_escenario1(frame_rgb)
            frame_resultado = cv2.cvtColor(frame_resultado, cv2.COLOR_RGB2BGR)
        elif escenario == 2:
            frame_resultado = segmentar_frame_escenario2(frame, 
                                                         params['hsv_low'], 
                                                         params['hsv_high'], 
                                                         params['d_real'], 
                                                         params['f_calibrada'])
        out.write(frame_resultado)

    cap.release()
    out.release()
    print(f"Vídeo guardado en: {video_salida}")