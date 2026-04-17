"""
segmentacion_hsv.py
===================
Módulo de segmentación por umbral HSV para el robot.

Escenario 1 — Suelo + línea azul + marca/flecha roja.
Escenario 2 — Objeto circular sobre fondo homogéneo.

Algoritmo elegido: umbralización en espacio HSV
Justificación:
  - El espacio HSV desacopla el color (Hue) de la iluminación (Value),
    lo que da más robustez ante cambios de luz que RGB/rgb normalizado.
  - La complejidad es O(N) en el número de píxeles: ideal para tiempo real.
  - Los tres objetos de interés tienen colores muy discriminativos
    (rojo, azul, beige/gris) → umbrales simples son suficientes.
  - Alternativas descartadas:
      · K-means: más lento y no determinista.
      · GMM / Bayes: requiere entrenamiento por secuencia, frágil ante
        cambios de escena.
      · Redes neuronales: excesiva complejidad para este problema.
"""

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
    """
    Segmenta un frame del escenario 1.

    Parámetros
    ----------
    frame_rgb : np.ndarray (H, W, 3) uint8, espacio RGB
    k_morph   : int  tamaño del kernel de apertura morfológica (elimina ruido)

    Devuelve
    --------
    imagen_coloreada : np.ndarray (H, W, 3) uint8
        Cada píxel coloreado según su etiqueta.
    etiquetas : np.ndarray (H, W) uint8
        0 = fondo, 1 = marca roja, 2 = línea azul
    """
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


# ─────────────────────────────────────────────
# Escenario 2 — objeto circular + distancia
# ─────────────────────────────────────────────

def detectar_circulo(frame_rgb, dp=1.2, min_dist=50,
                     param1=80, param2=30,
                     min_radius=10, max_radius=200):
    """
    Detecta el círculo más prominente en el frame usando HoughCircles.

    Devuelve
    --------
    circulo : (cx, cy, r) en píxeles o None si no se detecta.
    """
    gris = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    gris = cv2.GaussianBlur(gris, (9, 9), 2)

    circulos = cv2.HoughCircles(
        gris,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circulos is None:
        return None

    # Tomamos el círculo con mayor radio (más prominente)
    circulos = np.round(circulos[0]).astype(int)
    mejor = max(circulos, key=lambda c: c[2])
    return tuple(mejor)   # (cx, cy, r)


def calcular_distancia(radio_px, diametro_real_cm, focal_px):
    """
    Calcula la distancia cámara-objeto usando el modelo pinhole.

        distancia = (diámetro_real * focal) / diámetro_píxeles

    Parámetros
    ----------
    radio_px        : radio del objeto en píxeles
    diametro_real_cm: diámetro real del objeto en cm
    focal_px        : longitud focal de la cámara en píxeles

    Devuelve
    --------
    distancia_cm : float
    """
    if radio_px <= 0:
        return None
    diametro_px = 2 * radio_px
    return (diametro_real_cm * focal_px) / diametro_px


def segmentar_frame_escenario2(frame_rgb,
                                diametro_real_cm=7.0,
                                focal_px=600.0,
                                hsv_lo=None, hsv_hi=None):
    """
    Segmenta el objeto esférico y calcula su distancia.

    Para el fondo homogéneo se usa sustracción de color: el usuario puede
    proporcionar un rango HSV del objeto o se usa detección por Hough.

    Devuelve
    --------
    frame_anotado : frame con el círculo dibujado y la distancia escrita
    distancia_cm  : float o None
    """
    frame_anotado = frame_rgb.copy()
    distancia_cm = None

    circulo = detectar_circulo(frame_rgb)

    if circulo is not None:
        cx, cy, r = circulo
        distancia_cm = calcular_distancia(r, diametro_real_cm, focal_px)

        # Dibujar círculo y centro
        cv2.circle(frame_anotado, (cx, cy), r, (0, 255, 0), 3)
        cv2.circle(frame_anotado, (cx, cy), 4, (0, 0, 255), -1)

        # Escribir distancia en el lateral izquierdo
        if distancia_cm is not None:
            texto = f"Dist: {distancia_cm:.1f} cm"
            cv2.putText(
                frame_anotado, texto,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 0), 2, cv2.LINE_AA,
            )

    return frame_anotado, distancia_cm
