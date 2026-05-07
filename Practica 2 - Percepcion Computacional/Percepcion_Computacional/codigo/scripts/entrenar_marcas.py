#!/usr/bin/env python3
"""Entrena el clasificador de marcas (LDA) y lo serializa a ``.pkl``.

Imprime también la comparativa Leave-One-Out de QDA / LDA / KNN(1) /
KNN(3) sobre el dataset de entrenamiento, con su matriz de confusión.

.. code-block:: bash

    python scripts/entrenar_marcas.py --salida modelos/marcas.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

# Asegurarse de poder importar robot_vision desde la raíz
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import robot_vision as rv                                # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402


def main():
    p = argparse.ArgumentParser(description='Entrena el clasificador de marcas')
    p.add_argument('--dataset', default=rv.config.RUTA_DATASET_MARCAS)
    p.add_argument('--salida',  default=os.path.join(_ROOT, 'modelos', 'marcas.pkl'))
    p.add_argument('--margen',  type=float, default=rv.config.MARCA_MARGEN_RANGOS)
    args = p.parse_args()

    print(f'>>> Cargando dataset desde {args.dataset}')
    ds = rv.marcas.cargar_dataset(args.dataset)
    print(f'    N={ds.X.shape[0]}  features={ds.X.shape[1]}')

    print('\n>>> Comparativa Leave-One-Out')
    modelos = rv.marcas.evaluar_modelos(ds)
    for m in modelos:
        print(f'    {m.nombre:7s}  acc LOO = {m.cv_score:.4f}')

    mejor = max(modelos, key=lambda m: m.cv_score)
    print(f'\n>>> Modelo seleccionado: {mejor.nombre} (acc={mejor.cv_score:.4f})')
    print(classification_report(ds.y, mejor.cv_pred,
                                target_names=rv.config.CLASES_MARCAS,
                                digits=4, zero_division=0))
    print('Matriz de confusión:')
    print(confusion_matrix(ds.y, mejor.cv_pred))

    clf = rv.marcas.entrenar(ds)
    rangos = rv.marcas.rangos_por_clase(ds, margen=args.margen)
    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    with open(args.salida, 'wb') as f:
        pickle.dump({'clf': clf, 'rangos': rangos,
                     'clases': rv.config.CLASES_MARCAS}, f)
    print(f'\nGuardado modelo + rangos en {args.salida}')


if __name__ == '__main__':
    main()
