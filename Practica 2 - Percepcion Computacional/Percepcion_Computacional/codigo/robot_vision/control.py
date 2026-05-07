from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import config as C


# ===========================================================================
# Error de seguimiento
# ===========================================================================
def error_seguimiento(mask_linea: np.ndarray,
                      banda_inferior: int = C.BANDA_INFERIOR_ERROR_PX
                      ) -> Optional[float]:
    """Error horizontal normalizado en la franja inferior del frame."""
    h, w = mask_linea.shape
    banda = mask_linea[-banda_inferior:, :]
    if not banda.any():
        return None
    _, xs = np.where(banda)
    cx = xs.mean()
    return float((cx - w / 2) / (w / 2))


# ===========================================================================
# Controlador PD
# ===========================================================================
class ControlPD:
    """Controlador proporcional-derivativo simple."""

    def __init__(self,
                 kp: float    = C.CONTROL_PD_KP,
                 kd: float    = C.CONTROL_PD_KD,
                 v_max: float = C.CONTROL_PD_V_MAX,
                 v_min: float = C.CONTROL_PD_V_MIN,
                 alpha_v: float = C.CONTROL_PD_ALPHA_V):
        self.kp, self.kd = kp, kd
        self.v_max, self.v_min, self.alpha_v = v_max, v_min, alpha_v
        self._prev: Optional[float] = None

    def reset(self) -> None:
        self._prev = None

    def actualizar(self, error: Optional[float], dt: float = 1.0
                   ) -> Tuple[float, float]:
        """Devuelve ``(v, ω)`` a partir del error actual.

        Si el error es ``None`` (línea perdida) devuelve ``(0, 0)`` y
        reinicia el estado interno.
        """
        if error is None:
            self._prev = None
            return 0.0, 0.0
        derror = 0.0 if self._prev is None else (error - self._prev) / max(dt, 1e-3)
        self._prev = error
        omega = -(self.kp * error + self.kd * derror)
        v = max(self.v_min, self.v_max * (1.0 - self.alpha_v * abs(error)))
        return v, omega
