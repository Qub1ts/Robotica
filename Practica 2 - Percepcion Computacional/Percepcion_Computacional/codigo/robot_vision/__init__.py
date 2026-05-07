"""Paquete *robot_vision*.

Sistema de visión para el robot móvil de la Práctica 2 de Robótica y
Percepción Computacional. Contiene:

* :mod:`robot_vision.config`       — constantes, rutas y parámetros.
* :mod:`robot_vision.segmentacion` — QDA del Escenario 1 (Parte 1).
* :mod:`robot_vision.escena`       — entradas / salidas / tipo de escena.
* :mod:`robot_vision.flecha`       — orientación de la flecha de cruce.
* :mod:`robot_vision.control`      — error de seguimiento + PD.
* :mod:`robot_vision.distancia`    — segmentación esférica + distancia
  cámara→objeto (Parte 3).
* :mod:`robot_vision.marcas`       — clasificador de marcas (Parte 4).
* :mod:`robot_vision.anotacion`    — utilidades de visualización.
* :mod:`robot_vision.pipeline`     — pipeline completo por frame y por vídeo.
"""

from . import config              # noqa: F401
from . import segmentacion        # noqa: F401
from . import escena              # noqa: F401
from . import flecha              # noqa: F401
from . import control             # noqa: F401
from . import distancia           # noqa: F401
from . import marcas              # noqa: F401
from . import anotacion           # noqa: F401
from . import pipeline            # noqa: F401

__version__ = '1.0.0'
__all__ = ['config', 'segmentacion', 'escena', 'flecha', 'control',
           'distancia', 'marcas', 'anotacion', 'pipeline']
