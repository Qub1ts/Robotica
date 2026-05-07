from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import anotacion, config as C, control, escena, flecha, marcas, segmentacion
from .escena import Escena, Extremo
from .flecha import Flecha


# ===========================================================================
# Resultado por frame
# ===========================================================================
@dataclass
class ResultadoFrame:
    mask_linea: np.ndarray
    mask_marca: np.ndarray
    extremos: List[Extremo]
    escena: Escena
    flecha: Optional[Flecha]
    salida_elegida: Optional[Extremo]
    error: Optional[float]
    consigna: Tuple[float, float]   # (v, omega)
    marca_clase: Optional[str] = None
    marca_conf:  Optional[float] = None
    marca_bbox:  Optional[Tuple[int, int, int, int]] = None


# ===========================================================================
# Pipeline por frame
# ===========================================================================
def analizar_frame(rgb: np.ndarray,
                   clf_qda,
                   ctrl: control.ControlPD,
                   *,
                   modo_seg: str = 'qda',
                   clf_marcas=None,
                   rangos_marcas=None,
                   frac_min_linea: float = C.FRAC_MIN_LINEA
                   ) -> ResultadoFrame:
    """Aplica el pipeline completo a un frame RGB."""
    h, w = rgb.shape[:2]
    m_lin, m_mar = segmentacion.segmentar(rgb, clf_qda, modo=modo_seg)

    # Reservorio: si la línea es minoritaria, declarar 'sin_linea'
    if m_lin.sum() < frac_min_linea * h * w:
        m_lin = np.zeros_like(m_lin)
        extremos: List[Extremo] = []
        esc = Escena(tipo='sin_linea')
        err: Optional[float] = None
    else:
        extremos = escena.detectar_extremos(m_lin)
        esc = escena.clasificar(extremos, ancho=w)
        err = control.error_seguimiento(m_lin)

    fl = flecha.orientacion(m_mar)
    salida = flecha.seleccionar_salida(esc, fl)
    v, omega = ctrl.actualizar(err)

    res = ResultadoFrame(m_lin, m_mar, extremos, esc, fl, salida, err, (v, omega))

    # Clasificación de marca: solo en escenas sin cruce y si hay máscara roja
    if (clf_marcas is not None and m_mar.any()
            and esc.tipo not in ('cruce_2', 'cruce_3')):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        pm = marcas.predecir(bgr, clf_marcas, rangos=rangos_marcas)
        if pm is not None:
            res.marca_clase = pm.clase
            res.marca_conf  = pm.confianza
            res.marca_bbox  = pm.bbox
    return res


# ===========================================================================
# Procesado de vídeo
# ===========================================================================
def procesar_video(video_in: str, video_out: str,
                   *,
                   clf_qda=None, clf_marcas=None, rangos_marcas=None,
                   ctrl: Optional[control.ControlPD] = None,
                   modo_seg: str = 'qda',
                   indices_muestra: Optional[set] = None,
                   verbose: bool = True) -> dict:
    """Aplica el pipeline a un vídeo entero y lo guarda anotado."""
    if ctrl is None:
        ctrl = control.ControlPD()
    ctrl.reset()

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise FileNotFoundError(video_in)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*C.DEFAULT_FOURCC)
    writer = cv2.VideoWriter(video_out, fourcc, fps, (w, h))

    indices_muestra = set(indices_muestra or [])
    frames_muestra = []
    escenas: Counter = Counter()
    errores: List[Optional[float]] = []
    tiempos: List[float] = []

    for i in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        res = analizar_frame(rgb, clf_qda, ctrl,
                             modo_seg=modo_seg,
                             clf_marcas=clf_marcas,
                             rangos_marcas=rangos_marcas)
        bgr_an = anotacion.anotar_pipeline(rgb, res)
        tiempos.append(time.perf_counter() - t0)
        writer.write(bgr_an)

        escenas[res.escena.tipo] += 1
        errores.append(res.error)
        if i in indices_muestra:
            frames_muestra.append((i, rgb.copy(), bgr_an.copy(), res))
        if verbose and (i + 1) % 200 == 0:
            ms = float(np.mean(tiempos[-200:])) * 1000
            print(f'  Frame {i+1}/{n}  ({ms:.1f} ms/frame)')

    cap.release()
    writer.release()
    return {
        'frames_muestra': frames_muestra,
        'escenas'       : escenas,
        'errores'       : errores,
        'ms_frame'      : float(np.mean(tiempos) * 1000),
        'fps_video'     : float(fps),
        'n_frames'      : len(tiempos),
    }


# ===========================================================================
# Procesado solo-segmentación (Parte 1, sin pipeline de control)
# ===========================================================================
def procesar_video_segmentacion(video_in: str,
                                video_out_overlay: str,
                                video_out_puro: str,
                                clf_qda,
                                modo: str = 'qda',
                                alpha_overlay: float = 0.5,
                                verbose: bool = True) -> dict:
    """Sólo segmentación. Guarda dos vídeos: máscara pura y overlay."""
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise FileNotFoundError(video_in)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*C.DEFAULT_FOURCC)
    w1 = cv2.VideoWriter(video_out_puro,    fourcc, fps, (w, h))
    w2 = cv2.VideoWriter(video_out_overlay, fourcc, fps, (w, h))

    tiempos: List[float] = []
    for i in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        m_lin, m_mar = segmentacion.segmentar(rgb, clf_qda, modo=modo)
        puro_rgb = segmentacion.colorizar_mascaras(m_lin, m_mar)
        tiempos.append(time.perf_counter() - t0)
        overlay_rgb = cv2.addWeighted(rgb, 1 - alpha_overlay,
                                       puro_rgb, alpha_overlay, 0)
        w1.write(cv2.cvtColor(puro_rgb,    cv2.COLOR_RGB2BGR))
        w2.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        if verbose and (i + 1) % 200 == 0:
            ms = float(np.mean(tiempos[-200:])) * 1000
            print(f'  Frame {i+1}/{n}  ({ms:.1f} ms/frame)')

    cap.release()
    w1.release(); w2.release()
    return {
        'n_frames': len(tiempos),
        'ms_frame': float(np.mean(tiempos) * 1000),
        'fps'     : float(fps),
    }
