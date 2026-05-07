#!/usr/bin/env python3
"""Entrena el QDA del Escenario 1 y lo serializa a un fichero ``.pkl``.

.. code-block:: bash

    python scripts/entrenar_qda.py --salida modelos/qda.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import imageio.v2 as iio

# Asegurarse de poder importar robot_vision desde la raíz del proyecto
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import robot_vision as rv  # noqa: E402


def main():
    p = argparse.ArgumentParser(description='Entrena QDA y guarda el modelo.')
    p.add_argument('--imagen-orig',   default=rv.config.RUTA_IMG_ORIGINAL)
    p.add_argument('--imagen-marc',   default=rv.config.RUTA_IMG_MARCADA)
    p.add_argument('--reg-param',     type=float, default=rv.config.QDA_REG_PARAM)
    p.add_argument('--salida',        default=os.path.join(_ROOT, 'modelos', 'qda.pkl'))
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    orig = iio.imread(args.imagen_orig)
    marc = iio.imread(args.imagen_marc)
    clf = rv.segmentacion.entrenar_qda(orig, marc, reg_param=args.reg_param)

    with open(args.salida, 'wb') as f:
        pickle.dump(clf, f)
    print(f'Guardado QDA en {args.salida}')


if __name__ == '__main__':
    main()
