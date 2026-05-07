from __future__ import annotations

import glob
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from . import config as C


_RX = re.compile(r'^([a-z]+)[-_]\d+\.png$', re.IGNORECASE)


# ===========================================================================
# Segmentación de la silueta
# ===========================================================================
def mascara_roja(bgr: np.ndarray) -> np.ndarray:
    """Máscara binaria de los píxeles rojos saturados."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, C.HSV_ROJO_LO1, C.HSV_ROJO_HI1) | \
        cv2.inRange(hsv, C.HSV_ROJO_LO2, C.HSV_ROJO_HI2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return m > 0


def silueta(bgr_or_mask: np.ndarray, area_min: int = 80
            ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Extrae la silueta (recorte binario) de la marca roja principal."""
    mask = mascara_roja(bgr_or_mask) if bgr_or_mask.ndim == 3 \
           else bgr_or_mask.astype(bool)
    if not mask.any():
        return None
    n, labs, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    if stats[idx, cv2.CC_STAT_AREA] < area_min:
        return None
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
    sil = (labs[y:y+h, x:x+w] == idx).astype(np.uint8) * 255
    return sil, (x, y, w, h)


# ===========================================================================
# Descriptor invariante
# ===========================================================================
def log_hu(silueta_bin: np.ndarray) -> np.ndarray:
    """Logaritmo signado de los 7 momentos de Hu (invariantes a t/r/s)."""
    M = cv2.moments(silueta_bin, binaryImage=True)
    hu = cv2.HuMoments(M).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


def ratios_forma(silueta_bin: np.ndarray) -> np.ndarray:
    """4 ratios geométricos: aspect, extent, solidity, circularity."""
    cnts, _ = cv2.findContours(silueta_bin, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros(4, dtype=np.float32)
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, closed=True)
    if perim <= 1 or area <= 1:
        return np.zeros(4, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = h / w if w else 0.0
    extent = area / (w * h) if w and h else 0.0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area else 0.0
    circ = 4 * math.pi * area / (perim ** 2)
    return np.array([aspect, extent, solidity, circ], dtype=np.float32)


def descriptor(silueta_bin: np.ndarray) -> np.ndarray:
    """Descriptor final: 7 log-Hu + 4 ratios = 11 features."""
    return np.concatenate([log_hu(silueta_bin),
                           ratios_forma(silueta_bin)]).astype(np.float32)


# ===========================================================================
# Carga del dataset
# ===========================================================================
@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    files: List[str]
    siluetas: List[np.ndarray]
    label_names: Tuple[str, ...] = C.CLASES_MARCAS


def _clase_de_fichero(fname: str) -> Optional[str]:
    m = _RX.match(os.path.basename(fname))
    if not m:
        return None
    n = m.group(1).lower()
    return n if n in C.CLASES_MARCAS else None


def cargar_dataset(carpeta: str = C.RUTA_DATASET_MARCAS) -> Dataset:
    """Carga las imágenes del dataset y devuelve un :class:`Dataset`."""
    files = sorted(glob.glob(os.path.join(carpeta, '*.png')))
    feats, labels, kept, sils = [], [], [], []
    for f in files:
        clase = _clase_de_fichero(f)
        if clase is None:
            continue
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        out = silueta(bgr)
        if out is None:
            continue
        sil, _ = out
        feats.append(descriptor(sil))
        labels.append(C.CLASES_MARCAS.index(clase))
        kept.append(f)
        sils.append(sil)
    return Dataset(X=np.stack(feats), y=np.array(labels),
                   files=kept, siluetas=sils)


# ===========================================================================
# Entrenamiento / evaluación
# ===========================================================================
@dataclass
class ResultadoModelo:
    clf: object
    nombre: str
    cv_score: float
    cv_pred: np.ndarray


def evaluar_modelos(ds: Dataset) -> List[ResultadoModelo]:
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
        preds = np.empty_like(ds.y)
        for tr, te in loo.split(ds.X):
            base.fit(ds.X[tr], ds.y[tr])
            preds[te] = base.predict(ds.X[te])
        clf = type(base)(**base.get_params()).fit(ds.X, ds.y)
        out.append(ResultadoModelo(
            clf=clf, nombre=nombre,
            cv_score=float(scores.mean()),
            cv_pred=preds,
        ))
    return out


def entrenar(ds: Dataset) -> LinearDiscriminantAnalysis:
    """Clasificador por defecto (LDA)."""
    return LinearDiscriminantAnalysis(solver='svd').fit(ds.X, ds.y)


def rangos_por_clase(ds: Dataset, margen: float = C.MARCA_MARGEN_RANGOS
                     ) -> Dict[int, np.ndarray]:
    """Para cada clase, ``[min, max]`` de los 4 ratios con margen aditivo."""
    out: Dict[int, np.ndarray] = {}
    for k in range(len(C.CLASES_MARCAS)):
        Xc = ds.X[ds.y == k][:, 7:11]
        out[k] = np.column_stack([Xc.min(axis=0) - margen,
                                  Xc.max(axis=0) + margen])
    return out


# ===========================================================================
# Inferencia
# ===========================================================================
@dataclass
class PrediccionMarca:
    clase: str
    confianza: float
    bbox: Tuple[int, int, int, int]
    silueta: np.ndarray


def predecir(bgr: np.ndarray,
             clf,
             area_min: int = C.MARCA_AREA_MIN,
             umbral_conf: float = C.MARCA_UMBRAL_CONFIANZA,
             rangos: Optional[Dict[int, np.ndarray]] = None
             ) -> Optional[PrediccionMarca]:
    """Detecta y clasifica la marca en la imagen BGR.

    Devuelve ``None`` si:

    * no hay silueta roja suficientemente grande,
    * la confianza del clasificador es baja, o
    * los 4 ratios caen fuera del rango por-clase (filtro de flechas).
    """
    out = silueta(bgr, area_min=area_min)
    if out is None:
        return None
    sil, bbox = out
    feat = descriptor(sil).reshape(1, -1)
    probs = clf.predict_proba(feat)[0] if hasattr(clf, 'predict_proba') else None
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
    return PrediccionMarca(clase=C.CLASES_MARCAS[pred],
                           confianza=conf, bbox=bbox, silueta=sil)


# ===========================================================================
# Anotación + vídeo solo-clasificador
# ===========================================================================
def anotar_solo(bgr: np.ndarray, clf,
                rangos: Optional[Dict[int, np.ndarray]] = None
                ) -> Tuple[np.ndarray, Optional[PrediccionMarca]]:
    """Anota un frame con SOLO el resultado del clasificador."""
    pred = predecir(bgr, clf, rangos=rangos)
    out = bgr.copy()
    h, w = out.shape[:2]
    cabecera = 'Clasificador de marcas (LDA)'
    sub = 'Clases: ' + ' / '.join(C.CLASES_MARCAS)
    for i, txt in enumerate([cabecera, sub]):
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
    x, y, ww, hh = pred.bbox
    cv2.rectangle(out, (x, y), (x + ww, y + hh), (0, 200, 255), 2)
    txt = f'{pred.clase}  ({pred.confianza:.2f})'
    ty = y + hh + 16 if (y + hh + 16) < h else max(12, y - 6)
    cv2.putText(out, txt, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, txt, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (0, 200, 255), 1, cv2.LINE_AA)
    return out, pred


def procesar_video_clasificador(video_in: str, video_out: str, clf,
                                rangos: Optional[Dict[int, np.ndarray]] = None,
                                verbose: bool = True) -> dict:
    """Aplica :func:`anotar_solo` a cada frame del vídeo."""
    import time
    from collections import Counter

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise FileNotFoundError(video_in)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*C.DEFAULT_FOURCC)
    writer = cv2.VideoWriter(video_out, fourcc, fps, (w, h))

    cuentas = Counter()
    tiempos = []
    for i in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        an, pred = anotar_solo(bgr, clf, rangos=rangos)
        tiempos.append(time.perf_counter() - t0)
        writer.write(an)
        cuentas[pred.clase if pred else 'sin_marca'] += 1
        if verbose and (i + 1) % 200 == 0:
            ms = float(np.mean(tiempos[-200:])) * 1000
            print(f'  Frame {i+1}/{n}  ({ms:.1f} ms/frame)')

    cap.release()
    writer.release()
    return {
        'detecciones': cuentas,
        'ms_frame'   : float(np.mean(tiempos) * 1000),
        'n_frames'   : len(tiempos),
    }
