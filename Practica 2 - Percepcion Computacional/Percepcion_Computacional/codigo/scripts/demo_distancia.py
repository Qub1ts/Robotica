#!/usr/bin/env python3
"""Demostración del algoritmo de distancia (Parte 3) sobre un vídeo
sintético generado en memoria.

Crea una pelota de color que se acerca a la cámara desde 1.5 m hasta
0.3 m, aplica el algoritmo y compara la distancia estimada con la
real. Sirve para validar el módulo cuando todavía no se dispone de un
vídeo real.

.. code-block:: bash

    python scripts/demo_distancia.py --salida pelota_demo.mp4 --plot
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import robot_vision as rv  # noqa: E402


# ===========================================================================
# Generador de vídeo sintético
# ===========================================================================
def generar_frame(distancia_m: float,
                  diametro_real_m: float,
                  focal_px: float,
                  W: int = 640, H: int = 480,
                  color_bgr: Tuple[int, int, int] = (60, 200, 60),
                  ruido: float = 5.0) -> np.ndarray:
    """Genera un frame sintético de una pelota de color a la distancia dada."""
    diam_px = focal_px * diametro_real_m / max(distancia_m, 1e-3)
    radio = max(2, int(diam_px / 2))
    bgr = np.full((H, W, 3), 240, dtype=np.uint8)              # fondo blanco-grisáceo
    bgr += np.random.randint(-int(ruido), int(ruido) + 1,
                             size=bgr.shape, dtype=np.int16).astype(np.int16)\
            .clip(-255, 255).astype(np.uint8)
    cv2.circle(bgr, (W // 2, H // 2), radio, color_bgr, -1)
    cv2.circle(bgr, (W // 2, H // 2), radio, (40, 130, 40), 1)
    return np.clip(bgr, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--salida',   default='pelota_demo.mp4')
    p.add_argument('--diametro', type=float, default=rv.config.PELOTA_DIAMETRO_M)
    p.add_argument('--focal',    type=float, default=rv.config.PELOTA_FOCAL_PX)
    p.add_argument('--n',        type=int, default=180,
                   help='nº de frames; 180 frames @ 30 fps = 6 s')
    p.add_argument('--plot',     action='store_true')
    args = p.parse_args()

    distancias_reales = np.linspace(1.5, 0.3, args.n)
    H, W, FPS = 480, 640, 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.salida, fourcc, FPS, (W, H))

    distancias_estimadas: List[float] = []
    for d_real in distancias_reales:
        bgr = generar_frame(d_real, args.diametro, args.focal, W=W, H=H)
        det = rv.distancia.detectar(bgr,
                                    diametro_real_m=args.diametro,
                                    focal_px=args.focal,
                                    metodo='contorno')
        distancias_estimadas.append(det.distancia_m if det else np.nan)
        writer.write(rv.distancia.anotar(bgr, det))

    writer.release()

    err = np.abs(np.array(distancias_estimadas) - distancias_reales)
    print(f'Vídeo demo: {args.salida}')
    print(f'Error medio  |d_est - d_real|: {np.nanmean(err)*100:.2f} cm')
    print(f'Error máximo |d_est - d_real|: {np.nanmax(err)*100:.2f} cm')

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.plot(distancias_reales, label='d real', lw=2)
            plt.plot(distancias_estimadas, label='d estimada', lw=1, ls='--')
            plt.xlabel('Frame'); plt.ylabel('Distancia (m)'); plt.legend()
            plt.title('Distancia real vs estimada (demo sintética)')
            plt.tight_layout()
            out_png = os.path.splitext(args.salida)[0] + '_curva.png'
            plt.savefig(out_png, dpi=120)
            print(f'Gráfica guardada en {out_png}')
        except ImportError:
            print('matplotlib no disponible — omito la gráfica.')


if __name__ == '__main__':
    main()
