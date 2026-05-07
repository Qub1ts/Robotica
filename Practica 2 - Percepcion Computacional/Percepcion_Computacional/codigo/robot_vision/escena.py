from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config as C


# ===========================================================================
# Estructuras
# ===========================================================================
@dataclass
class Extremo:
    """Punto donde la línea corta uno de los bordes del frame."""
    lado: str                              # 'abajo' / 'arriba' / 'izquierda' / 'derecha'
    posicion: int                          # coordenada a lo largo del borde (px)
    longitud: int                          # ancho del segmento (px)
    punto: Tuple[int, int] = (0, 0)        # (x, y) en coordenadas del frame
    es_entrada: bool        = False        # True si el lado es 'abajo'
    angulo_deg: float       = 0.0          # ángulo desde el centro del frame


@dataclass
class Escena:
    """Resultado de la clasificación de la escena."""
    tipo: str                              # 'recta', 'curva_*', 'cruce_*', ...
    entrada: Optional[Extremo] = None
    salidas: List[Extremo]     = field(default_factory=list)


# ===========================================================================
# Detección de extremos
# ===========================================================================
def _segmentos_borde(perfil: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    p = perfil.astype(np.int8)
    d = np.diff(np.concatenate([[0], p, [0]]))
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    return [(int(a), int(b)) for a, b in zip(s, e) if (b - a) >= min_len]


def _fusionar(segmentos: List[Tuple[int, int]], gap: int) -> List[Tuple[int, int]]:
    if not segmentos:
        return segmentos
    out = [segmentos[0]]
    for s, e in segmentos[1:]:
        ps, pe = out[-1]
        if s - pe <= gap:
            out[-1] = (ps, e)
        else:
            out.append((s, e))
    return out


def detectar_extremos(mask_linea: np.ndarray,
                      banda: int = C.EXTREMO_BANDA_PX,
                      min_segmento: int = C.EXTREMO_MIN_SEGMENTO,
                      gap_fusion: int = C.EXTREMO_FUSION_GAP,
                      area_min_componente: int = C.EXTREMO_AREA_MIN_COMP
                      ) -> List[Extremo]:
    """Devuelve los puntos donde la línea corta los bordes del frame.

    Trabaja sobre cada componente conexa suficientemente grande
    (``area_min_componente``) para descartar ruido. Cada intersección
    con un borde se reporta como un :class:`Extremo`.
    """
    h, w = mask_linea.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_linea.astype(np.uint8), connectivity=8)

    extremos: List[Extremo] = []
    cx, cy = w / 2, h / 2

    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] < area_min_componente:
            continue
        m = (labels == lbl)
        bandas = {
            'abajo'    : (m[-banda:, :].any(axis=0),  'x'),
            'arriba'   : (m[:banda, :].any(axis=0),   'x'),
            'izquierda': (m[:, :banda].any(axis=1),   'y'),
            'derecha'  : (m[:, -banda:].any(axis=1),  'y'),
        }
        for lado, (perfil, eje) in bandas.items():
            for s, e in _fusionar(_segmentos_borde(perfil, min_segmento), gap_fusion):
                pos = (s + e) // 2
                if eje == 'x':
                    px, py = pos, (h - 1 if lado == 'abajo' else 0)
                else:
                    px, py = (0 if lado == 'izquierda' else w - 1), pos
                ang = math.degrees(math.atan2(cy - py, px - cx))
                extremos.append(Extremo(
                    lado=lado, posicion=pos, longitud=e - s,
                    punto=(int(px), int(py)),
                    es_entrada=(lado == 'abajo'),
                    angulo_deg=ang,
                ))
    return extremos


# ===========================================================================
# Clasificación de la escena
# ===========================================================================
def clasificar(extremos: List[Extremo], ancho: int) -> Escena:
    """Clasifica la escena en función del número y posición de extremos.

    Tipos posibles:

    * ``recta``        — 1 entrada + 1 salida superior centrada (±18 % W).
    * ``curva_izq``    — 1 entrada + 1 salida en lateral o esq. sup. izq.
    * ``curva_der``    — 1 entrada + 1 salida en lateral o esq. sup. der.
    * ``cruce_2``      — 1 entrada + 2 salidas (T o bifurcación).
    * ``cruce_3``      — 1 entrada + 3 salidas (X).
    * ``fin_linea``    — entrada presente, sin salidas.
    * ``sin_linea``    — no se ve la línea.
    """
    if not extremos:
        return Escena(tipo='sin_linea')

    entradas = [e for e in extremos if e.es_entrada]
    salidas  = [e for e in extremos if not e.es_entrada]
    entrada  = max(entradas, key=lambda e: e.longitud) if entradas else None

    n = len(salidas)
    if n == 0:
        tipo = 'fin_linea' if entrada else 'sin_linea'
    elif n == 1:
        s = salidas[0]
        if s.lado == 'arriba':
            x_ref = entrada.posicion if entrada else ancho / 2
            if abs(s.posicion - x_ref) < ancho * C.RECTA_TOLERANCIA_X:
                tipo = 'recta'
            else:
                tipo = 'curva_izq' if s.posicion < x_ref else 'curva_der'
        elif s.lado == 'izquierda':
            tipo = 'curva_izq'
        elif s.lado == 'derecha':
            tipo = 'curva_der'
        else:
            tipo = 'recta'
    elif n == 2:
        tipo = 'cruce_2'
    else:
        tipo = 'cruce_3'

    return Escena(tipo=tipo, entrada=entrada, salidas=salidas)
