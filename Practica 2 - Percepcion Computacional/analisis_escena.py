"""Análisis de escena para el control de un robot que sigue una línea.

Pipeline completo de la **segunda parte** de la práctica (Escenario 1):

1. Segmentación QDA de cada frame en {fondo, marca, línea}.
2. Detección de **entradas / salidas** por intersección con los bordes.
3. **Clasificación de la escena** (recta, curva D/I, cruce 2 ó 3 salidas, fin de línea).
4. **Orientación de la marca/flecha** mediante ajuste de elipse + análisis de
   asimetría (Hu) sobre el contorno.
5. **Salida seleccionada** = la que mejor se alinea con la dirección que
   apunta la flecha.
6. **Error de seguimiento** medido en la franja inferior del frame
   (única zona "presente" del trazado, ver transparencias p. 79).
7. **Consigna de control** PD que minimiza dicho error.

Suposiciones razonables (pp. 79-80 de ``transp.pdf``):
    * Las líneas son **infinitas** dentro del campo visual del robot: nunca
      terminan en mitad de la imagen, solo intersectan los bordes.
    * En un mismo frame **no se ven segmentos futuros ni pasados** de la
      ruta: por eso solo se procesa una banda de la imagen para el error
      de seguimiento.
    * La cámara está fija sobre el robot mirando hacia adelante; el borde
      **inferior** del frame corresponde a "lo que tiene el robot debajo"
      y, por tanto, suele ser la **entrada** del trazado.
    * La línea es **azul saturado**, la marca **roja**, el suelo
      verde-grisáceo de baja saturación.
    * Hay **una sola marca** visible por escena (las marcas no aparecen en
      cruces, cf. p. 80) y, si aparece, condiciona la salida elegida.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# ---------------------------------------------------------------------------
# Paleta y etiquetas
# ---------------------------------------------------------------------------
COLOR_FONDO = (0, 255, 0)
COLOR_MARCA = (255, 0, 0)
COLOR_LINEA = (0, 0, 255)
PALETA = np.array([COLOR_FONDO, COLOR_MARCA, COLOR_LINEA], dtype=np.uint8)

LADOS = ('abajo', 'arriba', 'izquierda', 'derecha')


# ===========================================================================
# 1. SEGMENTACIÓN  (QDA del Escenario 1, ya validado en la primera parte)
# ===========================================================================
def extraer_features(rgb: np.ndarray) -> np.ndarray:
    """Vector de 9 features por píxel: HSV + a*b* + RGB normalizado + L*."""
    img = rgb.reshape(1, -1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    hsv[:, 0] /= 179.0
    hsv[:, 1] /= 255.0
    hsv[:, 2] /= 255.0
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    L = lab[:, 0:1] / 255.0
    a = (lab[:, 1:2] - 128.0) / 128.0
    b = (lab[:, 2:3] - 128.0) / 128.0
    rgb_f = rgb.reshape(-1, 3).astype(np.float32)
    s = rgb_f.sum(axis=1, keepdims=True).clip(min=1.0)
    return np.hstack([hsv, a, b, rgb_f / s, L]).astype(np.float32)


def entrenar_qda(imagen_orig: np.ndarray,
                 imagen_marc: np.ndarray,
                 reg_param: float = 0.01) -> QuadraticDiscriminantAnalysis:
    """Entrena el clasificador QDA a partir de la imagen etiquetada."""
    m_marca = (imagen_marc[:, :, 0] == 255) & (imagen_marc[:, :, 1] == 0)   & (imagen_marc[:, :, 2] == 0)
    m_fondo = (imagen_marc[:, :, 0] == 0)   & (imagen_marc[:, :, 1] == 255) & (imagen_marc[:, :, 2] == 0)
    m_linea = (imagen_marc[:, :, 0] == 0)   & (imagen_marc[:, :, 1] == 0)   & (imagen_marc[:, :, 2] == 255)
    X = np.vstack([
        extraer_features(imagen_orig[m_fondo]),
        extraer_features(imagen_orig[m_marca]),
        extraer_features(imagen_orig[m_linea]),
    ])
    y = np.hstack([
        np.zeros(m_fondo.sum(), dtype=int),
        np.ones (m_marca.sum(), dtype=int),
        np.full (m_linea.sum(), 2, dtype=int),
    ])
    return QuadraticDiscriminantAnalysis(reg_param=reg_param).fit(X, y)


def _filtrar_componentes(mask: np.ndarray, area_min: int) -> np.ndarray:
    if area_min <= 0 or not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros_like(mask, dtype=bool)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= area_min:
            keep[labels == lbl] = True
    return keep


def segmentar(rgb_frame: np.ndarray,
              clf: Optional[QuadraticDiscriminantAnalysis],
              area_min_linea: int = 120,
              area_min_marca: int = 60,
              modo: str = 'qda') -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve ``(mascara_linea, mascara_marca)`` booleanas.

    Parameters
    ----------
    modo : {'qda', 'hsv'}
        ``'qda'`` usa el clasificador del Escenario 1 (escenas reales con
        suelo). ``'hsv'`` umbraliza directamente azul/rojo en HSV; útil
        para escenas de simulador con fondo blanco, donde el QDA no
        aporta nada y obtiene falsas detecciones por sombras.
    """
    h, w = rgb_frame.shape[:2]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    if modo == 'hsv':
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # Azul (línea)
        m_lin = cv2.inRange(hsv,
                            np.array([90,  80,  60], np.uint8),
                            np.array([130, 255, 255], np.uint8))
        # Rojo (marca) — envuelve en 0/180
        m_mar = cv2.inRange(hsv, np.array([0,   80, 60], np.uint8),
                                  np.array([12, 255, 255], np.uint8)) | \
                cv2.inRange(hsv, np.array([165, 80, 60], np.uint8),
                                  np.array([179, 255, 255], np.uint8))
    else:
        feats = extraer_features(rgb_frame.reshape(-1, 3))
        pred = clf.predict(feats).reshape(h, w)
        m_lin = (pred == 2).astype(np.uint8) * 255
        m_mar = (pred == 1).astype(np.uint8) * 255

    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_OPEN,  k, iterations=1)
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_CLOSE, k, iterations=3)
    m_lin = _filtrar_componentes(m_lin > 0, area_min_linea)

    m_mar = cv2.morphologyEx(m_mar, cv2.MORPH_OPEN,  k, iterations=1)
    m_mar = cv2.morphologyEx(m_mar, cv2.MORPH_CLOSE, k, iterations=2)
    m_mar = _filtrar_componentes(m_mar > 0, area_min_marca)

    return m_lin, m_mar


# ===========================================================================
# 2. DETECCIÓN DE ENTRADAS / SALIDAS
# ===========================================================================
@dataclass
class Extremo:
    """Punto donde la línea corta uno de los bordes del frame."""
    lado: str          # 'abajo', 'arriba', 'izquierda', 'derecha'
    posicion: int      # coordenada a lo largo del borde (px)
    longitud: int      # ancho del segmento de la línea en el borde (px)
    punto: Tuple[int, int]   = (0, 0)   # (x, y) en coordenadas del frame
    es_entrada: bool          = False    # se asume el borde inferior
    angulo_deg: float        = 0.0      # 0=→  90=↑  180=←  -90=↓


def _segmentos_borde(perfil: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    p = perfil.astype(np.int8)
    d = np.diff(np.concatenate([[0], p, [0]]))
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    return [(int(a), int(b)) for a, b in zip(s, e) if (b - a) >= min_len]


def _fusionar(segmentos: List[Tuple[int, int]], gap: int) -> List[Tuple[int, int]]:
    if not segmentos:
        return segmentos
    out = [segmentos[0]]
    for s, e in segmentos[1:]:
        ps, pe = out[-1]
        if s - pe <= gap:
            out[-1] = (ps, e)
        else:
            out.append((s, e))
    return out


def detectar_extremos(mask_linea: np.ndarray,
                      banda: int = 4,
                      min_segmento: int = 5,
                      gap_fusion: int = 8,
                      area_min_componente: int = 200) -> List[Extremo]:
    """Devuelve los puntos donde la línea corta los bordes del frame.

    Trabaja sobre cada **componente conexa** suficientemente grande para
    descartar ruido (bordes de baldosas, sombras…). Cada intersección con
    un borde se reporta como un :class:`Extremo`.
    """
    h, w = mask_linea.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_linea.astype(np.uint8), connectivity=8)

    extremos: List[Extremo] = []
    cx, cy = w / 2, h / 2

    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] < area_min_componente:
            continue
        m = (labels == lbl)

        bandas = {
            'abajo'    : (m[-banda:, :].any(axis=0),   'x'),
            'arriba'   : (m[:banda, :].any(axis=0),    'x'),
            'izquierda': (m[:, :banda].any(axis=1),    'y'),
            'derecha'  : (m[:, -banda:].any(axis=1),   'y'),
        }
        for lado, (perfil, eje) in bandas.items():
            segs = _fusionar(_segmentos_borde(perfil, min_segmento), gap_fusion)
            for s, e in segs:
                pos = (s + e) // 2
                if eje == 'x':
                    px, py = pos, (h - 1 if lado == 'abajo' else 0)
                else:
                    px, py = (0 if lado == 'izquierda' else w - 1), pos
                ang = math.degrees(math.atan2(cy - py, px - cx))  # 0=→ 90=↑
                extremos.append(Extremo(
                    lado=lado, posicion=pos, longitud=e - s,
                    punto=(int(px), int(py)),
                    es_entrada=(lado == 'abajo'),
                    angulo_deg=ang,
                ))
    return extremos


# ===========================================================================
# 3. CLASIFICACIÓN DE LA ESCENA
# ===========================================================================
@dataclass
class Escena:
    tipo: str
    entrada: Optional[Extremo] = None
    salidas: List[Extremo]     = field(default_factory=list)


def clasificar_escena(extremos: List[Extremo], ancho: int) -> Escena:
    """Clasifica la escena en función del número y posición de los extremos.

    Tipos posibles:
        * ``recta``        — 1 entrada + 1 salida en el borde superior, centrada.
        * ``curva_izq``    — 1 entrada + 1 salida en el borde izquierdo o
                             esquina superior izquierda.
        * ``curva_der``    — 1 entrada + 1 salida en el borde derecho o
                             esquina superior derecha.
        * ``cruce_2``      — 1 entrada + 2 salidas (T en frente o bifurcación).
        * ``cruce_3``      — 1 entrada + 3 salidas (X o T completa).
        * ``fin_linea``    — entrada presente, sin salidas.
        * ``sin_linea``    — no se ve la línea.
    """
    if not extremos:
        return Escena(tipo='sin_linea')

    entradas = [e for e in extremos if e.es_entrada]
    salidas  = [e for e in extremos if not e.es_entrada]

    # Si no hay entrada por abajo pero sí salidas, la "entrada" es la salida
    # con la posición más baja (la más cercana al robot).
    entrada = max(entradas, key=lambda e: e.longitud) if entradas else None

    n = len(salidas)
    if n == 0:
        tipo = 'fin_linea' if entrada else 'sin_linea'
    elif n == 1:
        s = salidas[0]
        if s.lado == 'arriba':
            # recta si la salida está aproximadamente sobre la entrada
            x_ref = entrada.posicion if entrada else ancho / 2
            if abs(s.posicion - x_ref) < ancho * 0.18:
                tipo = 'recta'
            else:
                tipo = 'curva_izq' if s.posicion < x_ref else 'curva_der'
        elif s.lado == 'izquierda':
            tipo = 'curva_izq'
        elif s.lado == 'derecha':
            tipo = 'curva_der'
        else:
            tipo = 'recta'
    elif n == 2:
        tipo = 'cruce_2'
    else:
        tipo = 'cruce_3'

    return Escena(tipo=tipo, entrada=entrada, salidas=salidas)


# ===========================================================================
# 4. ORIENTACIÓN DE LA MARCA / FLECHA
# ===========================================================================
@dataclass
class Flecha:
    centro: Tuple[float, float]
    angulo_deg: float          # 0=→  90=↑  180=←  -90=↓ (sentido apuntado)
    eje_mayor: float
    eje_menor: float
    contorno: np.ndarray


def orientacion_flecha(mask_marca: np.ndarray,
                       area_min: int = 200) -> Optional[Flecha]:
    """Orientación de la marca por ajuste de elipse y análisis de asimetría.

    El sentido (cabeza vs cola) se decide comparando la **distancia media
    del contorno al centroide** a cada lado del eje principal: la cabeza
    de una flecha es más estrecha y corta, y una figura humana es más
    estrecha por la cabeza que por las piernas, así que en ambos casos el
    lado **más cercano al centroide** es la dirección apuntada.
    """
    if not mask_marca.any():
        return None
    cnts, _ = cv2.findContours(mask_marca.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < area_min or len(cnt) < 5:
        return None

    (cx, cy), (eje_men, eje_may), ang_elipse = cv2.fitEllipse(cnt)
    # cv2.fitEllipse devuelve el ángulo del eje MENOR respecto al eje X,
    # en el rango [0, 180). El eje MAYOR está rotado 90º.
    ang_eje_mayor = (ang_elipse - 90.0) % 180.0

    # Vectores unidad a lo largo del eje principal y su perpendicular
    rad = math.radians(ang_eje_mayor)
    ux, uy = math.cos(rad), -math.sin(rad)   # eje principal (en imagen y va hacia abajo)

    pts = cnt.reshape(-1, 2).astype(np.float32)
    proy = (pts[:, 0] - cx) * ux + (pts[:, 1] - cy) * uy
    pos = proy[proy > 0]
    neg = proy[proy < 0]
    if len(pos) and len(neg):
        # El lado con extensión MENOR es la "cabeza"
        d_pos = pos.max()
        d_neg = -neg.min()
        sentido = +1 if d_pos < d_neg else -1
    else:
        sentido = +1
    ang_flecha = math.degrees(math.atan2(-uy * sentido, ux * sentido))

    return Flecha(
        centro=(float(cx), float(cy)),
        angulo_deg=ang_flecha,
        eje_mayor=float(eje_may),
        eje_menor=float(eje_men),
        contorno=cnt,
    )


def salida_seleccionada(escena: Escena,
                        flecha: Optional[Flecha]) -> Optional[Extremo]:
    """Devuelve la salida que el robot debe tomar.

    * Si hay **una única salida** (línea recta o curva), no hace falta
      "elegir": esa salida es directamente la elegida.
    * Si hay **varias salidas** (cruces), se elige la cuya dirección desde
      el centro del frame esté más alineada con el ángulo de la flecha.
    * Si hay varias salidas y **no hay flecha**, se devuelve ``None``
      (el robot debería pararse o seguir una política por defecto).
    """
    if not escena.salidas:
        return None
    if len(escena.salidas) == 1:
        return escena.salidas[0]
    if flecha is None:
        return None

    def diff(a, b):
        d = (a - b + 180) % 360 - 180
        return abs(d)

    return min(escena.salidas, key=lambda s: diff(s.angulo_deg, flecha.angulo_deg))


# ===========================================================================
# 5. ERROR DE SEGUIMIENTO
# ===========================================================================
def error_seguimiento(mask_linea: np.ndarray,
                      banda_inferior: int = 40) -> Optional[float]:
    """Error horizontal normalizado a [-1, 1] entre el centro del robot y
    el centroide de la línea en la franja inmediatamente delante del robot.

    Convención:
        * +1  → la línea está totalmente a la derecha
        * -1  → la línea está totalmente a la izquierda
        *  0  → línea centrada
    """
    h, w = mask_linea.shape
    banda = mask_linea[-banda_inferior:, :]
    if not banda.any():
        return None
    ys, xs = np.where(banda)
    centroide_x = xs.mean()
    return float((centroide_x - w / 2) / (w / 2))


# ===========================================================================
# 6. CONSIGNA DE CONTROL
# ===========================================================================
class ControlPD:
    """Control proporcional-derivativo simple para minimizar el error de
    seguimiento.

    .. math::

        \\omega = -K_p\\,e - K_d\\,\\dot e
        \\quad\\quad
        v = v_{max} \\cdot (1 - \\alpha |e|)

    La velocidad lineal se reduce con el módulo del error (más cuidado en
    curvas) y se anula cuando se pierde la línea.
    """

    def __init__(self, kp: float = 1.2, kd: float = 0.4,
                 v_max: float = 0.5, v_min: float = 0.05,
                 alpha_v: float = 0.7):
        self.kp = kp
        self.kd = kd
        self.v_max = v_max
        self.v_min = v_min
        self.alpha_v = alpha_v
        self._prev = None

    def reset(self):
        self._prev = None

    def actualizar(self, error: Optional[float], dt: float = 1.0
                   ) -> Tuple[float, float]:
        if error is None:
            self._prev = None
            return 0.0, 0.0
        derror = 0.0 if self._prev is None else (error - self._prev) / max(dt, 1e-3)
        self._prev = error
        omega = -(self.kp * error + self.kd * derror)
        v = max(self.v_min, self.v_max * (1.0 - self.alpha_v * abs(error)))
        return v, omega


# ===========================================================================
# 7. PIPELINE COMPLETO POR FRAME
# ===========================================================================
@dataclass
class ResultadoFrame:
    mask_linea: np.ndarray
    mask_marca: np.ndarray
    extremos: List[Extremo]
    escena: Escena
    flecha: Optional[Flecha]
    salida_elegida: Optional[Extremo]
    error: Optional[float]
    consigna: Tuple[float, float]   # (v, omega)
    marca_clase: Optional[str] = None      # man / woman / stairs / telephone
    marca_conf:  Optional[float] = None
    marca_bbox:  Optional[Tuple[int, int, int, int]] = None


def analizar_frame(rgb: np.ndarray,
                   clf: Optional[QuadraticDiscriminantAnalysis],
                   control: ControlPD,
                   frac_min_linea: float = 0.012,
                   modo_seg: str = 'qda',
                   clf_marcas=None,
                   umbral_marca: float = 0.55,
                   area_min_marca_clas: int = 200) -> ResultadoFrame:
    """Pipeline completo por frame.

    Parameters
    ----------
    clf : QDA del Escenario 1 (puede ser ``None`` si ``modo_seg='hsv'``).
    control : ControlPD para la consigna.
    frac_min_linea : umbral de fracción mínima de píxeles de línea para
        considerar que se ve la línea.
    modo_seg : {'qda', 'hsv'} — pasado a :func:`segmentar`.
    clf_marcas : clasificador LDA (opcional) para clasificar la marca
        cuando ``escena.tipo`` es ``'recta'`` o ``'fin_linea'``
        (las marcas no aparecen en cruces, según p. 80 de transp.pdf).
    umbral_marca : confianza mínima del clasificador.
    """
    h, w = rgb.shape[:2]
    m_lin, m_mar = segmentar(rgb, clf, modo=modo_seg)

    if m_lin.sum() < frac_min_linea * h * w:
        m_lin = np.zeros_like(m_lin)
        extremos = []
        escena = Escena(tipo='sin_linea')
        error = None
    else:
        extremos = detectar_extremos(m_lin)
        escena   = clasificar_escena(extremos, ancho=w)
        error    = error_seguimiento(m_lin)

    flecha   = orientacion_flecha(m_mar)
    salida   = salida_seleccionada(escena, flecha)
    v, omega = control.actualizar(error)

    res = ResultadoFrame(m_lin, m_mar, extremos, escena, flecha, salida,
                         error, (v, omega))

    # Clasificación de marca solo en escenas sin cruce
    if clf_marcas is not None and m_mar.any() and \
       escena.tipo in ('recta', 'curva_izq', 'curva_der', 'fin_linea', 'sin_linea'):
        try:
            from clasificador_marcas import predecir as _pred_marca
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pm = _pred_marca(bgr, clf_marcas,
                             area_min=area_min_marca_clas,
                             umbral_conf=umbral_marca)
            if pm is not None:
                res.marca_clase = pm.clase
                res.marca_conf  = pm.confianza
                res.marca_bbox  = pm.bbox
        except Exception:
            pass

    return res


# ===========================================================================
# 8. ANOTACIÓN VISUAL
# ===========================================================================
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


def anotar_frame(rgb: np.ndarray, res: ResultadoFrame,
                 alpha_overlay: float = 0.30) -> np.ndarray:
    """Devuelve un BGR anotado listo para escribir al vídeo.

    Convenciones gráficas (acordadas con el usuario):

    * El **eje principal de la marca** (flecha cian) **solo se dibuja en
      cruces** (donde la marca es realmente una flecha que indica salida).
      En tramos rectos / curvas la marca es una figura (`man`, `woman`,
      `stairs`, `telephone`) y se etiqueta con el `bbox` y la clase.
    * **No se dibuja la flecha roja** desde el centro a la salida elegida
      (causaba puntas grandes y rayos confusos). En su lugar, la salida
      elegida se marca con un **círculo cian más grueso** y la etiqueta
      ``ELEG``.
    * Cada extremo se etiqueta: ``ENT`` (entrada por debajo), ``S1``,
      ``S2``, ``S3`` (salidas en orden de detección).
    """
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()

    # Overlay translúcido de las máscaras
    overlay = bgr.copy()
    overlay[res.mask_linea] = (255, 0, 0)
    overlay[res.mask_marca] = (0, 0, 255)
    bgr = cv2.addWeighted(overlay, alpha_overlay, bgr, 1 - alpha_overlay, 0)

    # Banda inferior usada para el error (gris claro)
    cv2.rectangle(bgr, (0, h - 40), (w - 1, h - 1), (180, 180, 180), 1)
    cv2.line(bgr, (w // 2, h - 40), (w // 2, h - 1), (180, 180, 180), 1)
    if res.error is not None:
        x_err = int(w / 2 + res.error * (w / 2))
        cv2.line(bgr, (x_err, h - 40), (x_err, h - 1), (0, 255, 255), 2)

    # Etiquetar entradas y salidas con texto (ENT / S1 / S2 / S3 / ELEG)
    salidas_ordenadas = list(res.escena.salidas)
    elegida = res.salida_elegida
    for e in res.extremos:
        if e.es_entrada:
            color = (0, 255, 0)        # verde
            etiqueta = 'ENT'
            grosor = 2
        else:
            es_eleg = (elegida is not None and
                       e.lado == elegida.lado and
                       e.posicion == elegida.posicion)
            if es_eleg:
                color = (255, 255, 0)  # cian (BGR) — elegida
                etiqueta = 'ELEG'
                grosor = 3
            else:
                color = (0, 255, 255)  # amarillo — otras salidas
                idx = salidas_ordenadas.index(e) + 1 if e in salidas_ordenadas else 0
                etiqueta = f'S{idx}'
                grosor = 2
        cv2.circle(bgr, e.punto, 8, color, grosor)
        # Posición del texto: empuja hacia el interior del frame según el lado
        tx, ty = e.punto
        if e.lado == 'arriba':       ty += 16
        elif e.lado == 'abajo':      ty -= 6
        elif e.lado == 'izquierda':  tx += 12
        else:                        tx -= 36
        cv2.putText(bgr, etiqueta, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, etiqueta, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, color, 1, cv2.LINE_AA)

    # Eje principal de la flecha: SOLO si es un cruce
    if res.flecha is not None and _es_cruce(res.escena.tipo):
        cv2.drawContours(bgr, [res.flecha.contorno], -1, (255, 255, 0), 1)
        cx, cy = map(int, res.flecha.centro)
        cv2.circle(bgr, (cx, cy), 3, (255, 255, 0), -1)
        L = 0.6 * res.flecha.eje_mayor
        rad = math.radians(res.flecha.angulo_deg)
        ex = int(cx + L * math.cos(rad))
        ey = int(cy - L * math.sin(rad))
        cv2.arrowedLine(bgr, (cx, cy), (ex, ey), (255, 255, 0), 2, tipLength=0.30)

    # Bounding box + etiqueta de la marca clasificada (solo si NO es cruce)
    if res.marca_bbox is not None and not _es_cruce(res.escena.tipo):
        x0, y0b, ww, hh = res.marca_bbox
        cv2.rectangle(bgr, (x0, y0b), (x0 + ww, y0b + hh), (0, 200, 255), 2)
        txt = (f'{res.marca_clase} ({res.marca_conf:.2f})'
               if res.marca_clase else 'marca')
        cv2.putText(bgr, txt, (x0, max(12, y0b - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, txt, (x0, max(12, y0b - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)

    # Panel de texto superior izquierda
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


# ===========================================================================
# 9. PROCESADO COMPLETO DE VÍDEO
# ===========================================================================
def procesar_video(video_in: str,
                   video_out: str,
                   clf: Optional[QuadraticDiscriminantAnalysis] = None,
                   control: Optional[ControlPD] = None,
                   indices_muestra: Optional[set] = None,
                   verbose: bool = True,
                   modo_seg: str = 'qda',
                   clf_marcas=None,
                   umbral_marca: float = 0.55) -> dict:
    """Procesa el vídeo y guarda uno anotado.

    Devuelve un diccionario con:
        * ``frames_muestra`` — lista de tuplas (índice, rgb_original, bgr_anotado, ResultadoFrame)
        * ``escenas`` — Counter con los tipos de escena por frame
        * ``errores`` — lista del error por frame
        * ``ms_frame`` — tiempo medio por frame en milisegundos
    """
    import time
    from collections import Counter

    if control is None:
        control = ControlPD()
    control.reset()

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise FileNotFoundError(video_in)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_out, fourcc, fps, (w, h))

    frames_muestra = []
    indices_muestra = set(indices_muestra or [])
    escenas = Counter()
    errores = []
    tiempos = []

    for i in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        res = analizar_frame(rgb, clf, control,
                             modo_seg=modo_seg,
                             clf_marcas=clf_marcas,
                             umbral_marca=umbral_marca)
        bgr_an = anotar_frame(rgb, res)
        tiempos.append(time.perf_counter() - t0)
        writer.write(bgr_an)
        escenas[res.escena.tipo] += 1
        errores.append(res.error)
        if i in indices_muestra:
            frames_muestra.append((i, rgb.copy(), bgr_an.copy(), res))
        if verbose and (i + 1) % 200 == 0:
            ms = np.mean(tiempos[-200:]) * 1000
            print(f'  Frame {i+1:4d}/{n}   ({ms:.1f} ms/frame)')

    cap.release()
    writer.release()

    return {
        'frames_muestra': frames_muestra,
        'escenas'       : escenas,
        'errores'       : errores,
        'ms_frame'      : float(np.mean(tiempos) * 1000),
        'fps_video'     : float(fps),
        'n_frames'      : len(tiempos),
    }
