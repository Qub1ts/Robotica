from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config as C


# ===========================================================================
# Estructura
# ===========================================================================
@dataclass
class Pelota:
    """Resultado de detección de un objeto esférico en un frame."""
    centro: Tuple[float, float]    # (cx, cy) en píxeles
    radio_px: float                 # radio en píxeles
    diametro_px: float              # 2 * radio_px
    distancia_m: float              # distancia estimada a la cámara
    bbox: Tuple[int, int, int, int] # (x, y, w, h)
    metodo: str                     # 'contorno' / 'hough'


# ===========================================================================
# Segmentación HSV
# ===========================================================================
def mascara_color(bgr: np.ndarray,
                  hsv_lo: np.ndarray = C.PELOTA_HSV_LO,
                  hsv_hi: np.ndarray = C.PELOTA_HSV_HI) -> np.ndarray:
    """Devuelve la máscara binaria de los píxeles dentro del rango HSV."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, hsv_lo, hsv_hi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return m


# ===========================================================================
# Detección por contorno + cv2.minEnclosingCircle
# ===========================================================================
def _detectar_por_contorno(mascara: np.ndarray,
                           area_min: int = C.PELOTA_AREA_MIN_PX
                           ) -> Optional[Tuple[float, float, float,
                                               Tuple[int, int, int, int]]]:
    cnts, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < area_min:
        return None
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    return float(cx), float(cy), float(r), (int(x), int(y), int(w), int(h))


# ===========================================================================
# Detección por cv2.HoughCircles (más exigente, requiere bordes claros)
# ===========================================================================
def _detectar_por_hough(mascara: np.ndarray
                        ) -> Optional[Tuple[float, float, float,
                                            Tuple[int, int, int, int]]]:
    blur = cv2.GaussianBlur(mascara, (9, 9), 2)
    circulos = cv2.HoughCircles(blur,
                                cv2.HOUGH_GRADIENT,
                                dp=C.PELOTA_HOUGH_DP,
                                minDist=C.PELOTA_HOUGH_MIN_DIST,
                                param1=C.PELOTA_HOUGH_PARAM1,
                                param2=C.PELOTA_HOUGH_PARAM2,
                                minRadius=5,
                                maxRadius=0)
    if circulos is None:
        return None
    circulos = np.round(circulos[0]).astype(int)
    cx, cy, r = max(circulos, key=lambda c: c[2])  # más grande
    return (float(cx), float(cy), float(r),
            (int(cx - r), int(cy - r), int(2 * r), int(2 * r)))


# ===========================================================================
# Estimación de distancia
# ===========================================================================
def estimar_distancia(diametro_px: float,
                      diametro_real_m: float = C.PELOTA_DIAMETRO_M,
                      focal_px: float = C.PELOTA_FOCAL_PX) -> float:
    """Devuelve la distancia (en metros) según el modelo pinhole.

    .. math:: z = f \\cdot D / p
    """
    if diametro_px <= 0:
        return float('inf')
    return float(focal_px * diametro_real_m / diametro_px)


# ===========================================================================
# Calibración de la focal
# ===========================================================================
def calibrar_focal(diametro_px: float,
                   distancia_m: float,
                   diametro_real_m: float = C.PELOTA_DIAMETRO_M
                   ) -> float:
    """A partir de una imagen con la pelota a distancia conocida,
    devuelve la **distancia focal calibrada** en píxeles.

    .. math:: f = p \\cdot z / D
    """
    if diametro_real_m <= 0:
        raise ValueError('diametro_real_m debe ser > 0')
    return float(diametro_px * distancia_m / diametro_real_m)


# ===========================================================================
# Pipeline por frame
# ===========================================================================
def detectar(bgr_frame: np.ndarray,
             hsv_lo: np.ndarray = C.PELOTA_HSV_LO,
             hsv_hi: np.ndarray = C.PELOTA_HSV_HI,
             diametro_real_m: float = C.PELOTA_DIAMETRO_M,
             focal_px: float = C.PELOTA_FOCAL_PX,
             metodo: str = 'contorno') -> Optional[Pelota]:
    """Detección + estimación de distancia en un frame BGR.

    Parameters
    ----------
    metodo : {'contorno', 'hough'}
        Algoritmo de localización del círculo. ``'contorno'`` es más
        rápido y robusto al ruido; ``'hough'`` se queda con la
        circunferencia más limpia, útil cuando la pelota está parcialmente
        ocluida.
    """
    mascara = mascara_color(bgr_frame, hsv_lo, hsv_hi)

    if metodo == 'hough':
        out = _detectar_por_hough(mascara)
    else:
        out = _detectar_por_contorno(mascara)

    if out is None:
        return None
    cx, cy, r, bbox = out
    diam_px = 2.0 * r
    z = estimar_distancia(diam_px, diametro_real_m, focal_px)
    return Pelota(centro=(cx, cy), radio_px=r, diametro_px=diam_px,
                  distancia_m=z, bbox=bbox, metodo=metodo)


# ===========================================================================
# Anotación visual
# ===========================================================================
def anotar(bgr: np.ndarray, det: Optional[Pelota]) -> np.ndarray:
    """Devuelve una copia del frame BGR con el círculo y la distancia."""
    out = bgr.copy()
    h, w = out.shape[:2]
    titulo = 'Deteccion + distancia (modelo pinhole)'
    cv2.putText(out, titulo, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, titulo, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (255, 255, 255), 1, cv2.LINE_AA)
    if det is None:
        msg = 'Sin pelota detectada'
        cv2.putText(out, msg, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, msg, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 180), 1, cv2.LINE_AA)
        return out
    cx, cy = int(det.centro[0]), int(det.centro[1])
    cv2.circle(out, (cx, cy), int(det.radio_px), (0, 200, 255), 2)
    cv2.circle(out, (cx, cy), 3, (0, 200, 255), -1)
    txt = f'd = {det.distancia_m:.2f} m  ({det.diametro_px:.0f} px)'
    ty = max(14, cy - int(det.radio_px) - 6)
    cv2.putText(out, txt, (cx - 60, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, txt, (cx - 60, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 200, 255), 1, cv2.LINE_AA)
    return out


# ===========================================================================
# Procesado de vídeo
# ===========================================================================
def procesar_video(video_in: str, video_out: str,
                   hsv_lo: np.ndarray = C.PELOTA_HSV_LO,
                   hsv_hi: np.ndarray = C.PELOTA_HSV_HI,
                   diametro_real_m: float = C.PELOTA_DIAMETRO_M,
                   focal_px: float = C.PELOTA_FOCAL_PX,
                   metodo: str = 'contorno',
                   verbose: bool = True) -> dict:
    """Procesa un vídeo completo aplicando :func:`detectar` y :func:`anotar`."""
    import time

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise FileNotFoundError(video_in)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*C.DEFAULT_FOURCC)
    writer = cv2.VideoWriter(video_out, fourcc, fps, (w, h))

    distancias = []
    tiempos = []
    detecciones = 0
    for i in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        det = detectar(bgr, hsv_lo, hsv_hi, diametro_real_m, focal_px, metodo)
        tiempos.append(time.perf_counter() - t0)
        writer.write(anotar(bgr, det))
        if det is not None:
            detecciones += 1
            distancias.append(det.distancia_m)
        if verbose and (i + 1) % 200 == 0:
            ms = float(np.mean(tiempos[-200:])) * 1000
            print(f'  Frame {i+1}/{n}  ({ms:.1f} ms/frame)')

    cap.release()
    writer.release()
    return {
        'n_frames'   : len(tiempos),
        'detecciones': detecciones,
        'distancias' : distancias,
        'ms_frame'   : float(np.mean(tiempos) * 1000),
    }
