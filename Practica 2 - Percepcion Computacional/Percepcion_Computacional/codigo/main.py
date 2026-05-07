#!/usr/bin/env python3
"""Punto de entrada principal del sistema de visión.

Pensado para ejecutarse **en el robot** o desde la línea de comandos
para procesar vídeos de prueba. El binario único permite cinco modos:

.. code-block:: bash

    # 1) Pipeline completo sobre un vídeo (Partes 1+2+4)
    python main.py pipeline --video video1.mp4 --salida video1_control.mp4

    # 2) Sólo segmentación (Parte 1) — genera dos vídeos
    python main.py segmentar --video video2017-3.avi --salida-puro out_puro.mp4 \\
        --salida-overlay out_overlay.mp4

    # 3) Sólo clasificador de marcas (Parte 4)
    python main.py marcas --video proyectoRobotica-1920-line1-video-2020-04-20.avi \\
        --salida marcas.mp4

    # 4) Detección + distancia a pelota (Parte 3)
    python main.py distancia --video pelota.mp4 --salida pelota_anotada.mp4 \\
        --diametro 0.07 --focal 600

    # 5) Modo `live` para integrar con la cámara del robot (ROS o cv2.VideoCapture(0))
    python main.py live --camara 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Asegura que el paquete se encuentra cuando se ejecuta desde la raíz
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import robot_vision as rv  # noqa: E402


# ===========================================================================
# Carga única de modelos (una vez por proceso)
# ===========================================================================
def cargar_modelos(usar_qda: bool = True, usar_marcas: bool = True):
    clf_qda = None
    clf_marcas = None
    rangos = None
    if usar_qda:
        import imageio.v2 as iio
        orig = iio.imread(rv.config.RUTA_IMG_ORIGINAL)
        marc = iio.imread(rv.config.RUTA_IMG_MARCADA)
        clf_qda = rv.segmentacion.entrenar_qda(orig, marc)
    if usar_marcas:
        ds = rv.marcas.cargar_dataset()
        clf_marcas = rv.marcas.entrenar(ds)
        rangos = rv.marcas.rangos_por_clase(ds)
    return clf_qda, clf_marcas, rangos


# ===========================================================================
# Comandos
# ===========================================================================
def cmd_pipeline(args):
    clf_qda, clf_marcas, rangos = cargar_modelos(usar_qda=(args.modo_seg == 'qda'),
                                                 usar_marcas=True)
    ctrl = rv.control.ControlPD()
    print(f'>>> Pipeline completo sobre {args.video}')
    r = rv.pipeline.procesar_video(
        args.video, args.salida,
        clf_qda=clf_qda, clf_marcas=clf_marcas,
        rangos_marcas=rangos, ctrl=ctrl,
        modo_seg=args.modo_seg)
    print(f'OK: {r["n_frames"]} frames · {r["ms_frame"]:.1f} ms/frame · '
          f'{1000/r["ms_frame"]:.1f} FPS')
    print('Escenas:', dict(r['escenas']))


def cmd_segmentar(args):
    clf_qda, _, _ = cargar_modelos(usar_qda=(args.modo == 'qda'),
                                   usar_marcas=False)
    print(f'>>> Sólo segmentación ({args.modo}) sobre {args.video}')
    r = rv.pipeline.procesar_video_segmentacion(
        args.video, args.salida_overlay, args.salida_puro,
        clf_qda=clf_qda, modo=args.modo,
        alpha_overlay=args.alpha)
    print(f'OK: {r["n_frames"]} frames · {r["ms_frame"]:.1f} ms/frame · '
          f'{1000/r["ms_frame"]:.1f} FPS')


def cmd_marcas(args):
    _, clf_marcas, rangos = cargar_modelos(usar_qda=False, usar_marcas=True)
    print(f'>>> Clasificador de marcas sobre {args.video}')
    r = rv.marcas.procesar_video_clasificador(
        args.video, args.salida, clf_marcas, rangos=rangos)
    print(f'OK: {r["n_frames"]} frames · {r["ms_frame"]:.1f} ms/frame · '
          f'{1000/r["ms_frame"]:.1f} FPS')
    print('Detecciones:', dict(r['detecciones']))


def cmd_distancia(args):
    print(f'>>> Detección + distancia sobre {args.video}')
    r = rv.distancia.procesar_video(
        args.video, args.salida,
        diametro_real_m=args.diametro,
        focal_px=args.focal,
        metodo=args.metodo)
    print(f'OK: {r["n_frames"]} frames · {r["ms_frame"]:.1f} ms/frame · '
          f'{r["detecciones"]} detecciones')
    if r['distancias']:
        d = np.array(r['distancias'])
        print(f'Distancia detectada: media={d.mean():.2f} m · '
              f'min={d.min():.2f} · max={d.max():.2f}')


def cmd_calibrar(args):
    """Calibra la focal a partir de una imagen con la pelota a distancia conocida."""
    bgr = cv2.imread(args.imagen)
    if bgr is None:
        print(f'No se pudo leer {args.imagen}')
        sys.exit(1)
    det = rv.distancia.detectar(bgr,
                                diametro_real_m=args.diametro,
                                focal_px=1.0,         # f no influye aquí
                                metodo='contorno')
    if det is None:
        print('No se detectó pelota en la imagen.')
        sys.exit(1)
    f = rv.distancia.calibrar_focal(det.diametro_px, args.distancia, args.diametro)
    print(f'Diámetro detectado: {det.diametro_px:.1f} px')
    print(f'Distancia conocida: {args.distancia:.3f} m')
    print(f'Diámetro real:       {args.diametro:.3f} m')
    print(f'\nDistancia focal calibrada: f = {f:.1f} px')
    print('Actualiza PELOTA_FOCAL_PX en robot_vision/config.py con este valor.')


def cmd_live(args):
    """Lazo en tiempo real desde la cámara del robot (ESC para salir)."""
    clf_qda, clf_marcas, rangos = cargar_modelos()
    ctrl = rv.control.ControlPD()

    cap = cv2.VideoCapture(int(args.camara) if args.camara.isdigit() else args.camara)
    if not cap.isOpened():
        print(f'No se pudo abrir cámara {args.camara}')
        sys.exit(1)

    print('Pulsa ESC en la ventana para terminar.')
    fps_t = time.perf_counter(); n = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = rv.pipeline.analizar_frame(rgb, clf_qda, ctrl,
                                         clf_marcas=clf_marcas,
                                         rangos_marcas=rangos)
        bgr_an = rv.anotacion.anotar_pipeline(rgb, res)
        cv2.imshow('robot_vision (ESC para salir)', bgr_an)
        n += 1
        if (time.perf_counter() - fps_t) > 1.0:
            print(f'  ~{n} fps  | escena={res.escena.tipo} | (v,w)={res.consigna}')
            fps_t = time.perf_counter(); n = 0
        if (cv2.waitKey(1) & 0xFF) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


# ===========================================================================
# CLI
# ===========================================================================
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='main.py',
        description='Sistema de visión del robot — Práctica 2.')
    sub = p.add_subparsers(dest='cmd', required=True)

    # ----- pipeline -----
    sp = sub.add_parser('pipeline', help='Pipeline completo (Partes 1+2+4)')
    sp.add_argument('--video',  required=True)
    sp.add_argument('--salida', required=True)
    sp.add_argument('--modo-seg', choices=['qda', 'hsv'], default='qda',
                    dest='modo_seg')
    sp.set_defaults(func=cmd_pipeline)

    # ----- segmentar -----
    sp = sub.add_parser('segmentar', help='Sólo segmentación (Parte 1)')
    sp.add_argument('--video',          required=True)
    sp.add_argument('--salida-puro',    required=True)
    sp.add_argument('--salida-overlay', required=True)
    sp.add_argument('--modo', choices=['qda', 'hsv'], default='qda')
    sp.add_argument('--alpha', type=float, default=0.5)
    sp.set_defaults(func=cmd_segmentar)

    # ----- marcas -----
    sp = sub.add_parser('marcas', help='Clasificador de marcas (Parte 4)')
    sp.add_argument('--video',  required=True)
    sp.add_argument('--salida', required=True)
    sp.set_defaults(func=cmd_marcas)

    # ----- distancia -----
    sp = sub.add_parser('distancia', help='Detección + distancia (Parte 3)')
    sp.add_argument('--video',     required=True)
    sp.add_argument('--salida',    required=True)
    sp.add_argument('--diametro',  type=float, default=rv.config.PELOTA_DIAMETRO_M,
                    help='Diámetro real del balón en metros')
    sp.add_argument('--focal',     type=float, default=rv.config.PELOTA_FOCAL_PX,
                    help='Distancia focal calibrada en píxeles')
    sp.add_argument('--metodo',    choices=['contorno', 'hough'], default='contorno')
    sp.set_defaults(func=cmd_distancia)

    # ----- calibrar -----
    sp = sub.add_parser('calibrar', help='Calibrar focal a partir de una imagen')
    sp.add_argument('--imagen',     required=True)
    sp.add_argument('--distancia',  required=True, type=float,
                    help='Distancia real cámara→pelota en metros')
    sp.add_argument('--diametro',   type=float, default=rv.config.PELOTA_DIAMETRO_M)
    sp.set_defaults(func=cmd_calibrar)

    # ----- live -----
    sp = sub.add_parser('live', help='Lazo en tiempo real desde cámara')
    sp.add_argument('--camara', default='0',
                    help='Índice de la cámara o ruta de stream')
    sp.set_defaults(func=cmd_live)

    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    args.func(args)
