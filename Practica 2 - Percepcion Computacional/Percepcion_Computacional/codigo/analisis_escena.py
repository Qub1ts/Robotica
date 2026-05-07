"""Wrapper de retrocompatibilidad: re-exporta la API del paquete
:mod:`robot_vision` para que los notebooks antiguos sigan funcionando.

A partir de v1.0 el código vive en ``robot_vision/``. Este fichero
existe únicamente para que los imports antiguos
``from analisis_escena import ...`` no rompan.
"""

from __future__ import annotations

# --- Reexportar funciones / clases utilizadas por los notebooks ----------
from robot_vision.config import (                       # noqa: F401
    COLOR_FONDO, COLOR_LINEA, COLOR_MARCA,
)
from robot_vision.segmentacion import (                  # noqa: F401
    extraer_features, entrenar_qda, segmentar,
)
from robot_vision.escena import (                        # noqa: F401
    Extremo, Escena,
    detectar_extremos,
    clasificar as clasificar_escena,
)
from robot_vision.flecha import (                        # noqa: F401
    Flecha,
    orientacion as orientacion_flecha,
    seleccionar_salida as salida_seleccionada,
)
from robot_vision.control import (                       # noqa: F401
    error_seguimiento, ControlPD,
)
from robot_vision.pipeline import (                      # noqa: F401
    ResultadoFrame, analizar_frame, procesar_video,
)
from robot_vision.anotacion import (                     # noqa: F401
    anotar_pipeline as anotar_frame,
)
