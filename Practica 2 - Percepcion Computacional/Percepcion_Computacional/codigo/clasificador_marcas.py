"""Wrapper de retrocompatibilidad: re-exporta la API del paquete
:mod:`robot_vision.marcas` para que los notebooks antiguos sigan
funcionando.
"""

from __future__ import annotations

from robot_vision.config import CLASES_MARCAS as CLASES   # noqa: F401
from robot_vision.marcas import (                          # noqa: F401
    Dataset, ResultadoModelo, PrediccionMarca,
    cargar_dataset, descriptor, log_hu, ratios_forma,
    mascara_roja, silueta as silueta_marca,
    evaluar_modelos, entrenar, rangos_por_clase,
    predecir, anotar_solo, procesar_video_clasificador,
)


def _rangos_por_clase(ds, margen: float = 0.10):   # alias antiguo
    return rangos_por_clase(ds, margen=margen)
