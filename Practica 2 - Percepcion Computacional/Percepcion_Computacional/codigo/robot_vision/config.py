from __future__ import annotations

import os
from typing import Tuple

import numpy as np


# ===========================================================================
# RUTAS POR DEFECTO (relativas a la raíz del proyecto)
# ===========================================================================
RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Imagen de calibración del Escenario 1 (frame con suelo, línea y marca).
RUTA_IMG_ORIGINAL = os.path.join(RAIZ_PROYECTO, 'imagen_original.png')

#: Misma imagen pintada manualmente con tres colores puros: rojo (marca),
#: verde (fondo) y azul (línea). Sirve como dataset etiquetado para QDA.
RUTA_IMG_MARCADA  = os.path.join(RAIZ_PROYECTO, 'imagen_marcada.png')

#: Carpeta con las 28 imágenes de las 4 marcas del simulador (Parte 4).
RUTA_DATASET_MARCAS = os.path.join(RAIZ_PROYECTO, 'marcas-capturasStage')


# ===========================================================================
# PALETA RGB Y CLASES
# ===========================================================================
COLOR_FONDO = (0, 255, 0)      # 0 — verde
COLOR_MARCA = (255, 0, 0)      # 1 — rojo
COLOR_LINEA = (0, 0, 255)      # 2 — azul
PALETA_CLASES = np.array([COLOR_FONDO, COLOR_MARCA, COLOR_LINEA], dtype=np.uint8)

CLASES_QDA  = ('fondo', 'marca', 'linea')
LADOS       = ('abajo', 'arriba', 'izquierda', 'derecha')


# ===========================================================================
# RANGOS HSV (segmentación HSV del simulador, sin QDA)
# ===========================================================================
#: Azul (línea). Saturación moderada-alta para descartar el suelo grisáceo.
HSV_AZUL_LO = np.array([90,  80,  60], dtype=np.uint8)
HSV_AZUL_HI = np.array([130, 255, 255], dtype=np.uint8)

#: Rojo (marca). Envuelve en H≈0/180, así que se necesitan dos rangos.
HSV_ROJO_LO1 = np.array([0,   80,  60], dtype=np.uint8)
HSV_ROJO_HI1 = np.array([12, 255, 255], dtype=np.uint8)
HSV_ROJO_LO2 = np.array([165, 80,  60], dtype=np.uint8)
HSV_ROJO_HI2 = np.array([179, 255, 255], dtype=np.uint8)


# ===========================================================================
# PARÁMETROS DE SEGMENTACIÓN 
# ===========================================================================
QDA_REG_PARAM     = 0.01    # regularización de Σ_k para evitar singularidades
AREA_MIN_LINEA    = 120     # px — descarta blobs azules pequeños (bordes baldosa)
AREA_MIN_MARCA    = 200     # px — descarta blobs rojos pequeños (sombras y reflejos)
KERNEL_MORF_SIZE  = 3       # tamaño del *structuring element*
ITERS_OPEN_LINEA  = 1
ITERS_CLOSE_LINEA = 3
ITERS_OPEN_MARCA  = 1
ITERS_CLOSE_MARCA = 2


# ===========================================================================
# PARÁMETROS DE DETECCIÓN DE EXTREMOS / ESCENA (Parte 2)
# ===========================================================================
EXTREMO_BANDA_PX        = 4    # ancho de la banda perimetral
EXTREMO_MIN_SEGMENTO    = 5    # longitud mínima de un *run* en el borde
EXTREMO_FUSION_GAP      = 8    # *runs* a menos de este gap se fusionan
EXTREMO_AREA_MIN_COMP   = 200  # área mínima de la componente conexa válida

#: Tolerancia para clasificar "recta" — fracción del ancho del frame que
#: la salida superior puede desviarse de la entrada y seguir siendo recta.
RECTA_TOLERANCIA_X = 0.18

#: Fracción mínima de píxeles de línea para considerar que se ve algo.
FRAC_MIN_LINEA = 0.012


# ===========================================================================
# PARÁMETROS DEL ERROR Y CONTROL PD 
# ===========================================================================
BANDA_INFERIOR_ERROR_PX = 40   # franja inferior donde se mide el error

CONTROL_PD_KP      = 1.2
CONTROL_PD_KD      = 0.4
CONTROL_PD_V_MAX   = 0.5
CONTROL_PD_V_MIN   = 0.05
CONTROL_PD_ALPHA_V = 0.7


# ===========================================================================
# PARÁMETROS DE DISTANCIA A OBJETO CIRCULAR 
# ===========================================================================
#: Diámetro real del balón, en metros. Ajustar a la pelota que use el robot.
PELOTA_DIAMETRO_M = 0.07     # 7 cm

#: Distancia focal calibrada en píxeles. Se obtiene con
#: :func:`robot_vision.distancia.calibrar_focal` o midiendo manualmente.
PELOTA_FOCAL_PX   = 600.0

#: Color de la pelota (HSV). Ajustar según el balón que use el robot.
PELOTA_HSV_LO = np.array([35,  80,  60], dtype=np.uint8)   # verde por defecto
PELOTA_HSV_HI = np.array([85, 255, 255], dtype=np.uint8)

PELOTA_AREA_MIN_PX = 80      # tamaño mínimo del blob de pelota
PELOTA_HOUGH_DP    = 1.2     # parámetro dp de cv2.HoughCircles
PELOTA_HOUGH_MIN_DIST = 50   # distancia mínima entre círculos (en px)
PELOTA_HOUGH_PARAM1 = 100    # alta-umbral de Canny interno
PELOTA_HOUGH_PARAM2 = 25     # umbral de centro (más bajo = más permisivo)


# ===========================================================================
# PARÁMETROS DEL CLASIFICADOR DE MARCAS (Parte 4)
# ===========================================================================
CLASES_MARCAS = ('man', 'stairs', 'telephone', 'woman')

MARCA_AREA_MIN          = 300       # marca mínima creíble (filtra ruido)
MARCA_UMBRAL_CONFIANZA  = 0.55
MARCA_MARGEN_RANGOS     = 0.05   # margen aditivo para el filtro de outliers
                                  # estricto: rechaza marcas no entrenadas
                                  # (cruces, garabatos, etc.) que el LDA
                                  # clasificaría con alta confianza por
                                  # construcción.


# ===========================================================================
# PARÁMETROS DE VÍDEO Y DEMOS
# ===========================================================================
DEFAULT_FOURCC = 'mp4v'

#: Vídeos de prueba que vienen con el material.
RUTAS_VIDEOS = {
    'video1'         : os.path.join(RAIZ_PROYECTO, 'video1.mp4'),
    'video2017-3'    : os.path.join(RAIZ_PROYECTO, 'video2017-3.avi'),
    'video2017-4'    : os.path.join(RAIZ_PROYECTO, 'video2017-4.avi'),
    'proyectoRobotica' : os.path.join(
        RAIZ_PROYECTO,
        'proyectoRobotica-1920-line1-video-2020-04-20.avi'),
}


def ruta_video(nombre: str) -> str:
    """Devuelve la ruta absoluta del vídeo identificado por su clave."""
    if nombre in RUTAS_VIDEOS:
        return RUTAS_VIDEOS[nombre]
    return os.path.join(RAIZ_PROYECTO, nombre)
