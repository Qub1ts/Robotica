from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from . import config as C


# ===========================================================================
# Features y entrenamiento
# ===========================================================================
def extraer_features(rgb: np.ndarray) -> np.ndarray:
    """Vector de 9 features por píxel.

    Parameters
    ----------
    rgb : np.ndarray
        ``(N, 3)`` o ``(H, W, 3)`` en formato RGB ``uint8``.

    Returns
    -------
    np.ndarray
        ``(N, 9)`` en ``float32``.
    """
    if rgb.ndim == 3:
        rgb = rgb.reshape(-1, 3)
    img = rgb.reshape(1, -1, 3).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    hsv[:, 0] /= 179.0
    hsv[:, 1] /= 255.0
    hsv[:, 2] /= 255.0

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    L = lab[:, 0:1] / 255.0
    a = (lab[:, 1:2] - 128.0) / 128.0
    b = (lab[:, 2:3] - 128.0) / 128.0

    rgb_f = rgb.astype(np.float32)
    s = rgb_f.sum(axis=1, keepdims=True).clip(min=1.0)
    rgb_n = rgb_f / s

    return np.hstack([hsv, a, b, rgb_n, L]).astype(np.float32)


def entrenar_qda(imagen_orig: np.ndarray,
                 imagen_marc: np.ndarray,
                 reg_param: float = C.QDA_REG_PARAM
                 ) -> QuadraticDiscriminantAnalysis:
    """Entrena el QDA a partir de la imagen etiquetada manualmente."""
    m_marca = (imagen_marc[:, :, 0] == 255) & (imagen_marc[:, :, 1] == 0)   & (imagen_marc[:, :, 2] == 0)
    m_fondo = (imagen_marc[:, :, 0] == 0)   & (imagen_marc[:, :, 1] == 255) & (imagen_marc[:, :, 2] == 0)
    m_linea = (imagen_marc[:, :, 0] == 0)   & (imagen_marc[:, :, 1] == 0)   & (imagen_marc[:, :, 2] == 255)

    X = np.vstack([
        extraer_features(imagen_orig[m_fondo]),
        extraer_features(imagen_orig[m_marca]),
        extraer_features(imagen_orig[m_linea]),
    ])
    y = np.hstack([
        np.zeros(int(m_fondo.sum()), dtype=int),
        np.ones (int(m_marca.sum()), dtype=int),
        np.full (int(m_linea.sum()), 2, dtype=int),
    ])
    return QuadraticDiscriminantAnalysis(reg_param=reg_param).fit(X, y)


# ===========================================================================
# Filtrado morfológico y por componentes
# ===========================================================================
def filtrar_componentes(mask: np.ndarray, area_min: int) -> np.ndarray:
    """Descarta componentes conexas con menos de ``area_min`` píxeles."""
    if area_min <= 0 or not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, dtype=bool)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= area_min:
            keep[labels == lbl] = True
    return keep


def _kernel(size: int = C.KERNEL_MORF_SIZE) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


# ===========================================================================
# Segmentación por frame
# ===========================================================================
#: HSV estricto para validar la marca roja "real" (saturación y valor altos).
#: Se usa como prefiltro junto al QDA para descartar texturas del suelo que
#: el QDA marca como rojas por error.
HSV_ROJO_VALID_LO1 = np.array([0,   130, 80], dtype=np.uint8)
HSV_ROJO_VALID_HI1 = np.array([10, 255, 255], dtype=np.uint8)
HSV_ROJO_VALID_LO2 = np.array([170, 130, 80], dtype=np.uint8)
HSV_ROJO_VALID_HI2 = np.array([179, 255, 255], dtype=np.uint8)


def _mascara_roja_estricta(rgb_frame: np.ndarray) -> np.ndarray:
    """Máscara HSV estricta de píxeles "claramente rojos saturados".

    Sirve como **prefiltro** al QDA: en vídeos reales con suelo
    reflectante, el QDA puede etiquetar texturas amarillentas como
    *marca*; intersectarlo con esta máscara estricta reduce
    drásticamente los falsos positivos sin penalizar las marcas reales.
    """
    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (cv2.inRange(hsv, HSV_ROJO_VALID_LO1, HSV_ROJO_VALID_HI1) |
            cv2.inRange(hsv, HSV_ROJO_VALID_LO2, HSV_ROJO_VALID_HI2)) > 0


def segmentar(rgb_frame: np.ndarray,
              clf: Optional[QuadraticDiscriminantAnalysis],
              modo: str = 'qda',
              area_min_linea: int = C.AREA_MIN_LINEA,
              area_min_marca: int = C.AREA_MIN_MARCA,
              prefiltro_hsv_marca: bool = False
              ) -> Tuple[np.ndarray, np.ndarray]:
   
    h, w = rgb_frame.shape[:2]
    k = _kernel()

    if modo == 'hsv':
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m_lin = cv2.inRange(hsv, C.HSV_AZUL_LO, C.HSV_AZUL_HI)
        m_mar = cv2.inRange(hsv, C.HSV_ROJO_LO1, C.HSV_ROJO_HI1) | \
                cv2.inRange(hsv, C.HSV_ROJO_LO2, C.HSV_ROJO_HI2)
    else:
        if clf is None:
            raise ValueError("modo='qda' requiere un clasificador entrenado")
        feats = extraer_features(rgb_frame)
        pred = clf.predict(feats).reshape(h, w)
        m_lin = (pred == 2).astype(np.uint8) * 255
        m_mar = (pred == 1).astype(np.uint8) * 255

    # --- Limpieza línea --------------------------------------------------
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_OPEN,  k, iterations=C.ITERS_OPEN_LINEA)
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_CLOSE, k, iterations=C.ITERS_CLOSE_LINEA)
    m_lin = filtrar_componentes(m_lin > 0, area_min_linea)

    # --- Limpieza marca + prefiltro HSV ---------------------------------
    m_mar = cv2.morphologyEx(m_mar, cv2.MORPH_OPEN,  k, iterations=C.ITERS_OPEN_MARCA)
    m_mar = cv2.morphologyEx(m_mar, cv2.MORPH_CLOSE, k, iterations=C.ITERS_CLOSE_MARCA)
    m_mar = filtrar_componentes(m_mar > 0, area_min_marca)

    if prefiltro_hsv_marca and modo == 'qda' and m_mar.any():
        m_hsv = _mascara_roja_estricta(rgb_frame)
        # Por componente: aceptar solo las que tengan ≥30 % de píxeles
        # validados por HSV estricto. Esto permite agujeros internos
        # (sombras dentro del rojo) pero descarta blobs amarillentos.
        n, lab, st, _ = cv2.connectedComponentsWithStats(
            m_mar.astype(np.uint8), connectivity=8)
        m_mar_clean = np.zeros_like(m_mar, dtype=bool)
        for c in range(1, n):
            comp = (lab == c)
            area = comp.sum()
            if area == 0:
                continue
            frac_valid = (comp & m_hsv).sum() / area
            if frac_valid >= 0.30:
                m_mar_clean[comp] = True
        m_mar = m_mar_clean

    return m_lin, m_mar


def colorizar_mascaras(mask_linea: np.ndarray,
                       mask_marca: np.ndarray) -> np.ndarray:
    """Devuelve una imagen RGB con la paleta de las clases."""
    h, w = mask_linea.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:] = C.COLOR_FONDO
    out[mask_marca] = C.COLOR_MARCA
    out[mask_linea] = C.COLOR_LINEA
    return out
