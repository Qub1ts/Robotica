from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .escena import Escena, Extremo


@dataclass
class Flecha:
    """Información de orientación de una marca-flecha."""
    centro: Tuple[float, float]
    angulo_deg: float        # 0=→  90=↑  180=←  -90=↓  (sentido apuntado)
    eje_mayor: float
    eje_menor: float
    contorno: np.ndarray
    punta: Tuple[float, float] = (0.0, 0.0)  # punto del contorno = cabeza


def _vector_unitario(angulo_deg: float) -> Tuple[float, float]:
    rad = math.radians(angulo_deg)
    return math.cos(rad), -math.sin(rad)


def orientacion(mask_marca: np.ndarray,
                area_min: int = 200) -> Optional[Flecha]:
    if not mask_marca.any():
        return None
    cnts, _ = cv2.findContours(mask_marca.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < area_min or len(cnt) < 5:
        return None
    perim = cv2.arcLength(cnt, closed=True)
    circularity = 4 * math.pi * area / max(perim ** 2, 1e-6)
    # Una flecha real es muy alargada (circ < 0.25 en este dataset);
    # las cruces, marcas humanas o blobs cuasi-redondos tienen
    # circ > 0.30 y NO se intentan orientar como flechas.
    if circularity > 0.30:
        return None

    # --- píxeles de la silueta para PCA ---------------------------------
    ys, xs = np.where(mask_marca)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())
    cov = np.cov(pts.T)
    eigval, eigvec = np.linalg.eigh(cov)
    eje = eigvec[:, int(np.argmax(eigval))]      # eje principal en image-coords
    perp = np.array([-eje[1], eje[0]])           # eje perpendicular

    # --- proyecciones sobre el eje principal y el perpendicular ---------
    proy_a = (pts[:, 0] - cx) * eje[0] + (pts[:, 1] - cy) * eje[1]
    proy_p = (pts[:, 0] - cx) * perp[0] + (pts[:, 1] - cy) * perp[1]
    pmin, pmax = float(proy_a.min()), float(proy_a.max())
    longitud = pmax - pmin

    # --- ejes para informar al pipeline ---------------------------------
    eje_may = float(longitud)
    eje_men = float(2 * np.std(proy_p))

    # --- ancho perpendicular en cada extremo (franja 25 %) y en el centro
    def ancho(banda_proy: np.ndarray, banda_perp: np.ndarray) -> float:
        if len(banda_proy) < 5:
            return 0.0
        return float(banda_perp.max() - banda_perp.min())

    franja = 0.25 * longitud
    sel_pos = proy_a > (pmax - franja)
    sel_neg = proy_a < (pmin + franja)
    sel_cent = np.abs(proy_a) < (0.20 * longitud)
    w_pos = ancho(proy_a[sel_pos], proy_p[sel_pos])
    w_neg = ancho(proy_a[sel_neg], proy_p[sel_neg])
    w_cen = ancho(proy_a[sel_cent], proy_p[sel_cent]) + 1e-6

    # En las flechas reales del Escenario 1 la **cola** se afila más
    # rápido que la cabeza triangular (la cola termina en punta y la
    # cabeza mantiene un ancho relativo mayor cerca del extremo). Por
    # tanto la cabeza es el lado con **mayor** ratio ancho/centro.
    ratio_pos = w_pos / w_cen
    ratio_neg = w_neg / w_cen
    sentido = +1 if ratio_pos > ratio_neg else -1

    # --- coordenadas del extremo elegido como cabeza --------------------
    idx_cabeza = int(np.argmax(proy_a)) if sentido == +1 else int(np.argmin(proy_a))
    px, py = float(pts[idx_cabeza, 0]), float(pts[idx_cabeza, 1])

    # --- ángulo apuntado en convenio matemático -------------------------
    ang_flecha = math.degrees(math.atan2(cy - py, px - cx))

    return Flecha(
        centro=(float(cx), float(cy)),
        angulo_deg=ang_flecha,
        eje_mayor=float(eje_may),
        eje_menor=float(eje_men),
        contorno=cnt,
        punta=(px, py),
    )


def seleccionar_salida(escena: Escena, flecha: Optional[Flecha]) -> Optional[Extremo]:
    
    if not escena.salidas:
        return None
    if len(escena.salidas) == 1:
        return escena.salidas[0]
    if flecha is None:
        return None

    def diff(a: float, b: float) -> float:
        d = (a - b + 180) % 360 - 180
        return abs(d)

    return min(escena.salidas,
               key=lambda s: diff(s.angulo_deg, flecha.angulo_deg))
