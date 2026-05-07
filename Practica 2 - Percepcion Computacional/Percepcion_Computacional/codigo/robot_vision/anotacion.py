from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .escena import Escena, Extremo
from .flecha import Flecha


_TIPO_ETIQUETA = {
    'recta'     : ('Linea recta',           (0,   200, 0)),
    'curva_izq' : ('Curva IZQ',             (0,   165, 255)),
    'curva_der' : ('Curva DER',             (0,   165, 255)),
    'cruce_2'   : ('Cruce 2 salidas',       (255, 100, 0)),
    'cruce_3'   : ('Cruce 3 salidas',       (255, 0,   0)),
    'fin_linea' : ('Fin de linea',          (0,   0,   200)),
    'sin_linea' : ('Sin linea',             (128, 128, 128)),
}


def _es_cruce(tipo: str) -> bool:
    return tipo in ('cruce_2', 'cruce_3')


def anotar_pipeline(rgb: np.ndarray,
                    res,
                    alpha_overlay: float = 0.30) -> np.ndarray:
    """Devuelve un BGR con la anotación completa del pipeline por frame.

    ``res`` es un :class:`robot_vision.pipeline.ResultadoFrame`.
    """
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()

    # 1) Overlay translúcido de las máscaras
    overlay = bgr.copy()
    overlay[res.mask_linea] = (255, 0, 0)   # azul real (BGR)
    overlay[res.mask_marca] = (0, 0, 255)
    bgr = cv2.addWeighted(overlay, alpha_overlay, bgr, 1 - alpha_overlay, 0)

    # 2) Banda inferior + posición del error
    cv2.rectangle(bgr, (0, h - 40), (w - 1, h - 1), (180, 180, 180), 1)
    cv2.line(bgr, (w // 2, h - 40), (w // 2, h - 1), (180, 180, 180), 1)
    if res.error is not None:
        x_err = int(w / 2 + res.error * (w / 2))
        cv2.line(bgr, (x_err, h - 40), (x_err, h - 1), (0, 255, 255), 2)

    # 3) Entradas / Salidas / ELEG
    elegida = res.salida_elegida
    salidas_list = list(res.escena.salidas)
    for e in res.extremos:
        if e.es_entrada:
            color, etq, gros = (0, 255, 0), 'ENT', 2
        else:
            es_eleg = (elegida is not None and
                       e.lado == elegida.lado and
                       e.posicion == elegida.posicion)
            if es_eleg:
                color, etq, gros = (255, 255, 0), 'ELEG', 3
            else:
                idx = salidas_list.index(e) + 1 if e in salidas_list else 0
                color, etq, gros = (0, 255, 255), f'S{idx}', 2
        cv2.circle(bgr, e.punto, 8, color, gros)
        tx, ty = e.punto
        if e.lado == 'arriba':       ty += 16
        elif e.lado == 'abajo':      ty -= 6
        elif e.lado == 'izquierda':  tx += 12
        else:                        tx -= 36
        cv2.putText(bgr, etq, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, etq, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, color, 1, cv2.LINE_AA)

    # 4) Eje principal de la flecha (solo cruces)
    if res.flecha is not None and _es_cruce(res.escena.tipo):
        f: Flecha = res.flecha
        cv2.drawContours(bgr, [f.contorno], -1, (255, 255, 0), 1)
        cx, cy = map(int, f.centro)
        cv2.circle(bgr, (cx, cy), 3, (255, 255, 0), -1)
        # punta detectada
        px, py = map(int, f.punta)
        cv2.circle(bgr, (px, py), 4, (0, 255, 255), -1)
        # flecha desde el centroide hacia la punta detectada
        cv2.arrowedLine(bgr, (cx, cy), (px, py),
                        (255, 255, 0), 2, tipLength=0.25)

    # 5) Bbox + clase de la marca (solo si NO es cruce)
    if res.marca_bbox is not None and not _es_cruce(res.escena.tipo):
        x0, y0b, ww, hh = res.marca_bbox
        cv2.rectangle(bgr, (x0, y0b), (x0 + ww, y0b + hh), (0, 200, 255), 2)
        if res.marca_clase is not None:
            txt = f'{res.marca_clase} ({res.marca_conf:.2f})'
            cv2.putText(bgr, txt, (x0, max(12, y0b - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0),
                        3, cv2.LINE_AA)
            cv2.putText(bgr, txt, (x0, max(12, y0b - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255),
                        1, cv2.LINE_AA)

    # 6) Panel de texto
    etiqueta_e, color_et = _TIPO_ETIQUETA.get(
        res.escena.tipo, (res.escena.tipo, (255, 255, 255)))
    v, omega = res.consigna
    err_txt = f'{res.error:+.2f}' if res.error is not None else '  --'
    entrada_txt = res.escena.entrada.lado if res.escena.entrada else '--'
    salidas_txt = '/'.join(s.lado[:3] for s in res.escena.salidas) or '--'
    elegida_txt = elegida.lado if elegida is not None else '--'
    flecha_txt  = (f'{res.flecha.angulo_deg:+5.0f}deg'
                   if (res.flecha is not None and _es_cruce(res.escena.tipo))
                   else '--')
    marca_txt   = (f'{res.marca_clase} ({res.marca_conf:.2f})'
                   if (res.marca_clase and not _es_cruce(res.escena.tipo))
                   else '--')

    txt_lines = [
        f'Escena : {etiqueta_e}',
        f'Entr   : {entrada_txt}',
        f'Salidas: {salidas_txt}',
        f'Eleg   : {elegida_txt}',
        f'Flecha : {flecha_txt}',
        f'Marca  : {marca_txt}',
        f'Error  : {err_txt}',
        f'v={v:+.2f}  w={omega:+.2f}',
    ]
    y0 = 14
    for i, line in enumerate(txt_lines):
        y = y0 + i * 13
        cv2.putText(bgr, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (0, 0, 0), 3, cv2.LINE_AA)
        col = color_et if i == 0 else (255, 255, 255)
        cv2.putText(bgr, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, col, 1, cv2.LINE_AA)
    return bgr
