"""Clasificador de **marcas distintas de la flecha** (Escenario 1, 3ª parte).

Las marcas (man / woman / stairs / telephone) son figuras rojas que aparecen
en tramos rectos de la línea (no en cruces, ver `transp.pdf` p. 80) y le
dicen al robot dónde se encuentra. Este módulo:

1. Segmenta la silueta roja de la marca por umbralización HSV
   (el fondo del simulador Stage es blanco; en escenas reales se delega
   en el QDA del Escenario 1, que ya separa rojo/azul/fondo).
2. Construye un **descriptor invariante** combinando:
   * Logaritmo de los **7 momentos de Hu** (invariantes a traslación,
     rotación y escala — ver `transp.pdf` p. 68-69).
   * **Ratios de forma** (extent, solidity, aspect ratio del bounding
     box ajustado a la elipse mínima, circularidad) que añaden
     discriminación geométrica complementaria.
3. Entrena un **QDA** (con `LDA` y `KNN` como referencia) y valida con
   ``LeaveOneOut`` (solo 28 muestras).
4. Provee :func:`predecir` que toma un frame BGR y un *bounding box* de
   la marca y devuelve la clase predicha + confianza.

El módulo es deliberadamente **autónomo** (no necesita el QDA del
Escenario 1) para poder evaluarse sobre el vídeo `proyectoRobotica…`,
que es de simulador y no se parece a las imágenes reales.
"""

from __future__ import annotations

import glob
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
CLASES = ('man', 'stairs', 'telephone', 'woman')

# Rangos HSV para el rojo (envuelve en H≈0/180, así que se necesitan dos).
HSV_R_LO1 = np.array([0,   80,  60], dtype=np.uint8)
HSV_R_HI1 = np.array([12, 255, 255], dtype=np.uint8)
HSV_R_LO2 = np.array([165, 80,  60], dtype=np.uint8)
HSV_R_HI2 = np.array([179, 255, 255], dtype=np.uint8)


# ===========================================================================
# 1.  SEGMENTACIÓN DE LA MARCA
# ===========================================================================
def mascara_roja_hsv(bgr: np.ndarray) -> np.ndarray:
    """Devuelve máscara booleana de los píxeles rojos saturados."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, HSV_R_LO1, HSV_R_HI1) | \
        cv2.inRange(hsv, HSV_R_LO2, HSV_R_HI2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return m > 0


def silueta_marca(bgr_or_mask: np.ndarray,
                  area_min: int = 80
                  ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Aísla la silueta principal de marca y devuelve (silueta, bbox).

    * ``silueta`` es una imagen binaria recortada al *bounding box*.
    * ``bbox`` es ``(x, y, w, h)`` en coordenadas de la imagen original.

    Acepta tanto un BGR (segmenta internamente con HSV) como una máscara
    booleana ya calculada.
    """
    if bgr_or_mask.ndim == 3:
        mask = mascara_roja_hsv(bgr_or_mask)
    else:
        mask = bgr_or_mask.astype(bool)
    if not mask.any():
        return None
    n, labs, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    # Componente conexa más grande (label 0 es el fondo)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    if stats[idx, cv2.CC_STAT_AREA] < area_min:
        return None
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
    silueta = (labs[y:y+h, x:x+w] == idx).astype(np.uint8) * 255
    return silueta, (x, y, w, h)


# ===========================================================================
# 2.  DESCRIPTOR INVARIANTE
# ===========================================================================
def log_hu(silueta: np.ndarray) -> np.ndarray:
    """Logaritmo signado de los 7 momentos de Hu — invariantes a t/r/s."""
    M = cv2.moments(silueta, binaryImage=True)
    hu = cv2.HuMoments(M).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


def ratios_forma(silueta: np.ndarray) -> np.ndarray:
    """4 ratios de forma robustos."""
    cnts, _ = cv2.findContours(silueta, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros(4, dtype=np.float32)
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, closed=True)
    if perimetro <= 1 or area <= 1:
        return np.zeros(4, dtype=np.float32)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect = h / w if w else 0.0
    extent = area / (w * h) if w and h else 0.0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area else 0.0

    circularity = 4 * math.pi * area / (perimetro ** 2)

    return np.array([aspect, extent, solidity, circularity], dtype=np.float32)


def perfil_anchuras(silueta: np.ndarray, n_franjas: int = 4) -> np.ndarray:
    """Anchura relativa de la silueta en ``n_franjas`` horizontales.

    Capta la *forma vertical* del icono — la mujer es ancha por debajo
    (falda) y estrecha por arriba (cabeza), el hombre es más uniforme,
    el teléfono es ancho arriba (auricular) y la escalera tiene anchos
    crecientes hacia abajo.
    """
    h, w = silueta.shape
    if h == 0 or w == 0 or not silueta.any():
        return np.zeros(n_franjas, dtype=np.float32)
    franjas = np.array_split(silueta, n_franjas, axis=0)
    out = []
    for fr in franjas:
        cols = (fr > 0).any(axis=0)
        # ancho ocupado en columnas / ancho total
        out.append(float(cols.sum()) / max(w, 1))
    return np.asarray(out, dtype=np.float32)


def descriptor(silueta: np.ndarray) -> np.ndarray:
    """Vector descriptor final: 7 log-Hu + 4 ratios = 11 features.

    Las anchuras por franja (:func:`perfil_anchuras`) se probaron pero
    introducen ruido en el dataset pequeño y bajan la accuracy LOO de
    LDA del 96 % al 86 %. Se dejan disponibles por si se quisieran
    explorar con más datos.
    """
    return np.concatenate([
        log_hu(silueta),
        ratios_forma(silueta),
    ]).astype(np.float32)


# ===========================================================================
# 3.  CARGA DEL DATASET
# ===========================================================================
@dataclass
class Dataset:
    X: np.ndarray              # (N, 11)
    y: np.ndarray              # (N,)
    files: List[str]
    siluetas: List[np.ndarray]
    label_names: Tuple[str, ...] = CLASES


_RX = re.compile(r'^([a-z]+)[-_]\d+\.png$', re.IGNORECASE)


def _clase_de_fichero(fname: str) -> Optional[str]:
    m = _RX.match(os.path.basename(fname))
    if not m:
        return None
    nombre = m.group(1).lower()
    return nombre if nombre in CLASES else None


def cargar_dataset(carpeta: str) -> Dataset:
    """Carga todas las imágenes de ``carpeta`` y devuelve un Dataset."""
    files = sorted(glob.glob(os.path.join(carpeta, '*.png')))
    feats, labels, kept_files, sils = [], [], [], []
    for f in files:
        clase = _clase_de_fichero(f)
        if clase is None:
            continue
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        out = silueta_marca(bgr)
        if out is None:
            print(f'  ! sin silueta detectada en {os.path.basename(f)}')
            continue
        sil, _ = out
        feats.append(descriptor(sil))
        labels.append(CLASES.index(clase))
        kept_files.append(f)
        sils.append(sil)
    return Dataset(
        X=np.stack(feats), y=np.array(labels), files=kept_files, siluetas=sils
    )


# ===========================================================================
# 4.  ENTRENAMIENTO + EVALUACIÓN
# ===========================================================================
@dataclass
class Modelo:
    clf: object
    nombre: str
    cv_score: float
    cv_pred: np.ndarray   # predicciones leave-one-out
    label_names: Tuple[str, ...] = CLASES


def evaluar_modelos(ds: Dataset) -> List[Modelo]:
    """Compara QDA, LDA y KNN con LeaveOneOut sobre el dataset."""
    candidatos = {
        'QDA'    : QuadraticDiscriminantAnalysis(reg_param=0.05),
        'LDA'    : LinearDiscriminantAnalysis(),
        'KNN(1)' : KNeighborsClassifier(n_neighbors=1),
        'KNN(3)' : KNeighborsClassifier(n_neighbors=3),
    }
    loo = LeaveOneOut()
    out = []
    for nombre, base in candidatos.items():
        scores = cross_val_score(base, ds.X, ds.y, cv=loo)
        # Predicciones LOO completas (para la matriz de confusión)
        preds = np.empty_like(ds.y)
        for tr, te in loo.split(ds.X):
            base.fit(ds.X[tr], ds.y[tr])
            preds[te] = base.predict(ds.X[te])
        # Reentrena sobre TODO
        clf = type(base)(**base.get_params())
        clf.fit(ds.X, ds.y)
        out.append(Modelo(clf=clf, nombre=nombre,
                          cv_score=float(scores.mean()),
                          cv_pred=preds))
    return out


def entrenar_qda(ds: Dataset, reg_param: float = 0.05
                 ) -> QuadraticDiscriminantAnalysis:
    return QuadraticDiscriminantAnalysis(reg_param=reg_param).fit(ds.X, ds.y)


def entrenar(ds: Dataset) -> LinearDiscriminantAnalysis:
    """Clasificador por defecto: LDA con `solver='svd'` (mejor para datasets
    pequeños con más features que muestras por clase)."""
    return LinearDiscriminantAnalysis(solver='svd').fit(ds.X, ds.y)


# ===========================================================================
# 5.  PREDICCIÓN EN INFERENCIA
# ===========================================================================
@dataclass
class PrediccionMarca:
    clase: str
    confianza: float
    bbox: Tuple[int, int, int, int]
    silueta: np.ndarray


def _rangos_por_clase(ds: 'Dataset', margen: float = 0.10
                      ) -> Dict[int, np.ndarray]:
    """Para cada clase, devuelve [min, max] de los 4 ratios de forma
    (índices 7-10 del descriptor) con un margen aditivo.

    Sirve como filtro de outliers: una silueta cuyos 4 ratios no caen
    dentro del rango EXPANDIDO de la clase predicha es probablemente un
    objeto rojo distinto a las marcas entrenadas (típicamente una
    flecha direccional en cruces).
    """
    rangos = {}
    for k in range(len(CLASES)):
        Xc = ds.X[ds.y == k][:, 7:11]
        rangos[k] = np.column_stack([Xc.min(axis=0) - margen,
                                     Xc.max(axis=0) + margen])
    return rangos


def predecir(bgr: np.ndarray,
             clf,
             area_min: int = 200,
             umbral_conf: float = 0.55,
             rangos: Optional[Dict[int, np.ndarray]] = None
             ) -> Optional[PrediccionMarca]:
    """Detecta y clasifica una marca en una imagen BGR.

    Devuelve ``None`` si:
        * no hay silueta roja suficientemente grande,
        * la confianza del clasificador está por debajo de ``umbral_conf``,
        * o, si ``rangos`` está dado, alguno de los 4 ratios de forma
          cae fuera del rango expandido de la clase predicha
          (filtro de outliers — rechaza, p.ej., flechas direccionales).
    """
    out = silueta_marca(bgr, area_min=area_min)
    if out is None:
        return None
    sil, bbox = out
    feat = descriptor(sil).reshape(1, -1)
    probs = clf.predict_proba(feat)[0] if hasattr(clf, 'predict_proba') \
            else None
    pred = int(clf.predict(feat)[0])
    conf = float(probs[pred]) if probs is not None else 1.0
    if conf < umbral_conf:
        return None

    if rangos is not None:
        rng = rangos.get(pred)
        if rng is not None:
            ratios = feat[0, 7:11]
            if np.any(ratios < rng[:, 0]) or np.any(ratios > rng[:, 1]):
                return None

    return PrediccionMarca(
        clase=CLASES[pred], confianza=conf, bbox=bbox, silueta=sil
    )


# ===========================================================================
# 6.  ANOTACIÓN AUXILIAR
# ===========================================================================
def dibujar_prediccion(bgr: np.ndarray,
                       pred: Optional[PrediccionMarca],
                       color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    out = bgr.copy()
    if pred is None:
        return out
    x, y, w, h = pred.bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    txt = f'{pred.clase} ({pred.confianza:.2f})'
    cv2.putText(out, txt, (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, txt, (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


# ===========================================================================
# 7.  PROCESADO DE VÍDEO — SOLO CLASIFICADOR
# ===========================================================================
def anotar_solo_clasificador(bgr: np.ndarray,
                             clf,
                             area_min: int = 200,
                             umbral_conf: float = 0.55,
                             rangos: Optional[Dict[int, np.ndarray]] = None
                             ) -> Tuple[np.ndarray, Optional[PrediccionMarca]]:
    """Anota un frame con SOLO el resultado del clasificador de marcas.

    No muestra información de escena, entrada/salida, error ni consigna.
    Pensado para validar visualmente el clasificador sobre el vídeo
    ``proyectoRobotica…`` independientemente del resto del pipeline.

    Si no hay marca o la confianza es baja, devuelve el frame con un
    pequeño rótulo ``Sin marca`` y ``pred = None``.
    """
    pred = predecir(bgr, clf, area_min=area_min,
                    umbral_conf=umbral_conf, rangos=rangos)
    out = bgr.copy()
    h, w = out.shape[:2]

    # Cabecera con el nombre del clasificador y las clases
    titulo = 'Clasificador de marcas (LDA)'
    subtitulo = 'Clases: ' + ' / '.join(CLASES)
    for i, txt in enumerate([titulo, subtitulo]):
        y_t = 16 + i * 16
        cv2.putText(out, txt, (6, y_t), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (6, y_t), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)

    if pred is None:
        msg = 'Sin marca'
        cv2.putText(out, msg, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, msg, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 180), 1, cv2.LINE_AA)
        return out, None

    # Bounding box + etiqueta de la clase
    x, y, ww, hh = pred.bbox
    cv2.rectangle(out, (x, y), (x + ww, y + hh), (0, 200, 255), 2)
    txt = f'{pred.clase}  ({pred.confianza:.2f})'
    # Texto debajo del bbox si hay sitio, encima si no
    ty = y + hh + 16 if (y + hh + 16) < h else max(12, y - 6)
    cv2.putText(out, txt, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, txt, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (0, 200, 255), 1, cv2.LINE_AA)
    return out, pred


def procesar_video_clasificador(video_in: str,
                                video_out: str,
                                clf,
                                area_min: int = 200,
                                umbral_conf: float = 0.55,
                                rangos: Optional[Dict[int, np.ndarray]] = None,
                                verbose: bool = True) -> dict:
    """Procesa un vídeo entero anotando SOLO el resultado del clasificador.

    Devuelve un diccionario con el conteo de detecciones por clase y
    el tiempo medio por frame.
    """
    import time
    from collections import Counter

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise FileNotFoundError(video_in)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_out, fourcc, fps, (w, h))

    cuentas = Counter()
    tiempos = []
    for i in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        an, pred = anotar_solo_clasificador(bgr, clf,
                                            area_min=area_min,
                                            umbral_conf=umbral_conf,
                                            rangos=rangos)
        tiempos.append(time.perf_counter() - t0)
        writer.write(an)
        cuentas[pred.clase if pred else 'sin_marca'] += 1
        if verbose and (i + 1) % 200 == 0:
            ms = float(np.mean(tiempos[-200:])) * 1000
            print(f'  Frame {i+1:4d}/{n}   ({ms:.1f} ms/frame)')

    cap.release()
    writer.release()
    return {
        'detecciones': cuentas,
        'ms_frame'   : float(np.mean(tiempos) * 1000),
        'n_frames'   : len(tiempos),
        'fps_video'  : float(fps),
    }
