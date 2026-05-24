#!/usr/bin/env python3
"""BrainFollowLine-3.py - VERSION FINAL (procesamiento offline de video).

Pipeline de percepcion integrado:
  1) Segmentacion por QDA (3 clases: fondo, marca-roja, linea-azul) con
     CLAHE previo y threshold de confianza por pixel.
  2) Deteccion de extremos del camino (ENT = entrada, S = salidas).
  3) Deteccion geometrica de FLECHA (circularidad, solidez, elongacion,
     asimetria) + confirmacion temporal de 2 frames.
  4) MEMORIA de la direccion indicada por la flecha (TTL ~12s).
  5) SALIDA ELEGIDA = la salida del camino que mejor encaja con la
     direccion memorizada (match de lado exacto + fallback angular).
  6) CLASIFICACION DE MARCAS (man / stairs / telephone / woman) con LDA
     de 5 clases (la 5ta es 'flecha' sintetica para que no confunda) +
     rechazo OOD por distancia de Mahalanobis + dataset aumentado con
     erosion/dilatacion + restriccion a la mitad inferior de la imagen.

USO:
    python BrainFollowLine-3.py                 # video 4 por defecto
    python BrainFollowLine-3.py --video 3
    python BrainFollowLine-3.py --no-mostrar    # sin ventana (mas rapido)
    python BrainFollowLine-3.py --sin-output    # no genera mp4 de salida
    python BrainFollowLine-3.py --start 800     # salta los primeros N frames
"""

import argparse
import glob
import math
import os
import re
import sys

import cv2
import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)


_AQUI = os.path.dirname(os.path.abspath(__file__))
RUTA_IMG_ORIGINAL   = os.path.join(_AQUI, 'imagen_original.png')
RUTA_IMG_MARCADA    = os.path.join(_AQUI, 'imagen_marcada.png')
RUTA_DATASET_MARCAS = os.path.join(_AQUI, 'marcas-capturasStage')
CLASES_MARCAS = ('man', 'stairs', 'telephone', 'woman')
# El LDA tiene una 5ta clase 'flecha' entrenada con siluetas sinteticas.
# Cuando el LDA predice 'flecha', NO se reporta como marca -> los blobs
# que son flechas (incluso mal recortadas) no se confunden con teleponos.
CLASES_LDA = CLASES_MARCAS + ('flecha',)
IDX_FLECHA = CLASES_LDA.index('flecha')
_RX_MARCA = re.compile(r'^([a-z]+)[-_]\d+\.png$', re.IGNORECASE)

_KERNEL_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_KERNEL_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# CLAHE: uniforma iluminacion antes del QDA. Aunque el QDA es robusto,
# CLAHE ayuda a que la distribucion de colores se parezca a la del
# training (imagen_original.png) cuando hay sobreexposicion.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def aplicar_clahe(bgr):
    """Ecualiza el canal L de Lab para uniformar el brillo sin cambiar tono."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _filtrar_componentes(mask, area_min):
    """Quita blobs pequenos por area minima."""
    mask = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            out[labels == i] = 255
    return out > 0


# ======================================================================
# SEGMENTACION POR COLOR CON QDA (3 clases: fondo / marca-roja / linea-azul)
# ======================================================================
# Reemplaza los rangos HSV manuales por un clasificador entrenado que
# aprende la distribucion conjunta de colores. Mucho mas robusto ante
# variaciones de iluminacion porque cada clase tiene su propia matriz
# de covarianza (eso es lo que aporta QDA sobre LDA).
#
# Features por pixel (9 dimensiones):
#   - HSV normalizado    (tono + saturacion + valor)
#   - a*, b* de Lab      (cromaticidad sin luminancia)
#   - RGB normalizado    (cromaticidad sin intensidad)
#   - L de Lab           (luminancia)
# Esta mezcla cubre tono, cromaticidad y luminancia -> el QDA puede
# discriminar azul/rojo/madera incluso con brillos y sombras.
def _features_pixel(rgb):
    """RGB -> matriz N x 9 de features para el QDA."""
    if rgb.ndim == 3:
        rgb = rgb.reshape(-1, 3)
    img = rgb.reshape(1, -1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    hsv *= np.array([1 / 179.0, 1 / 255.0, 1 / 255.0], dtype=np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    L  = lab[:, 0:1] / 255.0
    ab = (lab[:, 1:3] - 128.0) / 128.0
    rgb_f = rgb.astype(np.float32)
    rgb_n = rgb_f / rgb_f.sum(axis=1, keepdims=True).clip(min=1.0)
    return np.hstack([hsv, ab, rgb_n, L]).astype(np.float32)


def entrenar_qda_linea():
    """Entrena QDA de 3 clases usando imagen_original.png + imagen_marcada.png.

    En imagen_marcada.png los pixeles tienen colores puros:
      verde puro (0,255,0)  = FONDO   -> clase 0
      rojo puro  (255,0,0)  = MARCA   -> clase 1
      azul puro  (0,0,255)  = LINEA   -> clase 2
    """
    orig = cv2.cvtColor(cv2.imread(RUTA_IMG_ORIGINAL), cv2.COLOR_BGR2RGB)
    marc = cv2.cvtColor(cv2.imread(RUTA_IMG_MARCADA),  cv2.COLOR_BGR2RGB)
    m_marca = (marc[..., 0] == 255) & (marc[..., 1] == 0)   & (marc[..., 2] == 0)
    m_fondo = (marc[..., 0] == 0)   & (marc[..., 1] == 255) & (marc[..., 2] == 0)
    m_linea = (marc[..., 0] == 0)   & (marc[..., 1] == 0)   & (marc[..., 2] == 255)
    X = np.vstack([_features_pixel(orig[m_fondo]),
                   _features_pixel(orig[m_marca]),
                   _features_pixel(orig[m_linea])])
    y = np.hstack([np.zeros(int(m_fondo.sum()), dtype=int),
                   np.ones (int(m_marca.sum()), dtype=int),
                   np.full (int(m_linea.sum()), 2, dtype=int)])
    return QuadraticDiscriminantAnalysis(reg_param=0.01).fit(X, y)


_KERNEL_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def segmentar_qda(clf, bgr, area_lin_min=750, area_mar_min=350,
                  conf_min=0.55):
    """Aplica el QDA al frame y devuelve (m_lin, m_rojo) booleanas.

    Mejoras sobre la version simple:
      - Threshold de confianza: pixeles donde max_proba < conf_min se
        asignan al fondo. Asi no clasificamos pixeles ambiguos (zonas
        donde el rojo es casi-azul por brillo o sombra), evitando que
        la silueta de la flecha se rompa o que parte del rojo aparezca
        pintado como linea azul.
      - Cierre mas fuerte (kernel 7, iter 4) en m_rojo para reconectar
        el zigzag de las escaleras y tapar huecos por sobreexposicion.
    """
    bgr_eq = aplicar_clahe(bgr)
    rgb = cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2RGB)
    feats = _features_pixel(rgb)

    # Probabilidades por pixel y mejor clase
    proba = clf.predict_proba(feats)            # shape (N, 3)
    pred  = np.argmax(proba, axis=1)
    pmax  = proba[np.arange(len(pred)), pred]
    pred[pmax < conf_min] = 0                   # ambiguos -> fondo
    pred = pred.reshape(rgb.shape[:2])

    # Linea (clase 2)
    m_lin = (pred == 2).astype(np.uint8) * 255
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_OPEN,  _KERNEL_3, iterations=1)
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_CLOSE, _KERNEL_5, iterations=2)
    m_lin = _filtrar_componentes(m_lin, area_min=area_lin_min)

    # Marca / flecha (clase 1) -> closing moderado: si lo subimos mas,
    # las flechas pierden su forma alargada y caen en circ_alta.
    m_rojo = (pred == 1).astype(np.uint8) * 255
    m_rojo = cv2.morphologyEx(m_rojo, cv2.MORPH_OPEN,  _KERNEL_3, iterations=1)
    m_rojo = cv2.morphologyEx(m_rojo, cv2.MORPH_CLOSE, _KERNEL_5, iterations=3)
    m_rojo = _filtrar_componentes(m_rojo, area_min=area_mar_min)

    return m_lin, m_rojo


# ======================================================================
# DETECCION GEOMETRICA DE FLECHA
# ======================================================================
# Filtros CALIBRADOS midiendo flechas reales en los videos:
#   frame 560 : circ=0.47 sol=0.78 elong=2.3
#   frame 1309: circ=0.32 sol=0.74 elong=4.2
#   frame 2137: circ=0.26 sol=0.77 elong=6.1
#   frame 2588: circ=0.35 sol=0.67 elong=2.8
# La GUARDIA principal contra marcas (man/woman/etc) es ASIM_MIN: las
# marcas son simetricas (cabeza~pies, brazos~brazos), las flechas no.
FLECHA_CIRC_MAX    = 0.50
FLECHA_ELONG_MIN   = 2.0
FLECHA_ASIM_MIN    = 0.30
FLECHA_SOLIDEZ_MIN = 0.65


def detectar_flecha(m_rojo, area_min, diag=None):
    """Si el blob rojo mas grande pasa los 4 filtros geometricos,
    devuelve dict con angulo, centro, contorno y punta. Si no, None.

    Si diag es un dict, se incrementa diag[motivo] indicando que filtro
    rechazo el blob principal (sirve para calibrar los umbrales)."""
    def _no(motivo):
        if diag is not None:
            diag[motivo] = diag.get(motivo, 0) + 1
        return None

    if not m_rojo.any():
        return _no('sin_rojo')

    cnts, _ = cv2.findContours(m_rojo.astype(np.uint8),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return _no('sin_contornos')
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < area_min or len(cnt) < 5:
        return _no('area_pequena')

    perim = cv2.arcLength(cnt, True)
    circ  = 4.0 * math.pi * area / max(perim * perim, 1e-6)
    if circ > FLECHA_CIRC_MAX:
        return _no('circ_alta')

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull) or 1.0
    solidez = area / hull_area
    if solidez < FLECHA_SOLIDEZ_MIN:
        return _no('solidez_baja')

    # Mascara rellena del contorno para sacar los puntos
    mask = np.zeros(m_rojo.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)

    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    eigval, eigvec = np.linalg.eigh(np.cov(pts.T))
    elong = math.sqrt(float(max(eigval)) / float(max(min(eigval), 1e-6)))
    if elong < FLECHA_ELONG_MIN:
        return _no('elong_baja')

    eje  = eigvec[:, int(np.argmax(eigval))]
    perp = np.array([-eje[1], eje[0]])
    proy_a = (pts[:, 0] - cx) * eje[0]  + (pts[:, 1] - cy) * eje[1]
    proy_p = (pts[:, 0] - cx) * perp[0] + (pts[:, 1] - cy) * perp[1]
    pmin, pmax = float(proy_a.min()), float(proy_a.max())
    L = pmax - pmin
    if L < 5:
        return _no('L_pequena')

    # Asimetria: el ancho perpendicular en el 25% mas lejano de cada extremo
    franja = 0.25 * L
    pp_pos = proy_p[proy_a > pmax - franja]
    pp_neg = proy_p[proy_a < pmin + franja]
    s_pos = float(pp_pos.max() - pp_pos.min()) if len(pp_pos) >= 5 else 0.0
    s_neg = float(pp_neg.max() - pp_neg.min()) if len(pp_neg) >= 5 else 0.0
    max_s = max(s_pos, s_neg)
    if max_s < 1e-3 or abs(s_pos - s_neg) / max_s < FLECHA_ASIM_MIN:
        return _no('asim_baja')

    # La punta es el extremo MAS ANCHO (la cabeza de la flecha)
    idx = int(np.argmax(proy_a)) if s_pos > s_neg else int(np.argmin(proy_a))
    px, py = float(pts[idx, 0]), float(pts[idx, 1])

    if diag is not None:
        diag['ok'] = diag.get('ok', 0) + 1

    return {
        'angulo':   math.degrees(math.atan2(cy - py, px - cx)),
        'centro':   (cx, cy),
        'punta':    (px, py),
        'contorno': cnt,
        'area':     float(area),
        'circ':     float(circ),
        'elong':    float(elong),
        'solidez':  float(solidez),
    }


class ConfirmadorFlecha:
    """Solo reporta la flecha despues de N frames consecutivos con
    angulo similar (dentro de tolerancia). Filtra falsos transitorios."""

    def __init__(self, n_frames=2, tol_grados=30.0):
        self.n_frames = n_frames
        self.tol = tol_grados
        self._ang  = None
        self._cnt  = 0

    def update(self, flecha):
        if flecha is None:
            self._ang = None
            self._cnt = 0
            return None
        ang = flecha['angulo']
        if (self._ang is not None
                and abs((ang - self._ang + 180) % 360 - 180) < self.tol):
            self._cnt += 1
        else:
            self._ang = ang
            self._cnt = 1
        return flecha if self._cnt >= self.n_frames else None


# ======================================================================
# MEMORIA DE DIRECCION
# ======================================================================
def direccion_de_flecha(angulo):
    """Mapea el angulo de la flecha (atan2 estilo, en grados) a un lado.

    Convencion (y crece hacia abajo en imagen):
        0 deg   = derecha
       +90 deg  = arriba
      +/-180 deg = izquierda
       -90 deg  = abajo

    Devuelve 'izquierda', 'derecha', 'arriba' o 'abajo'.
    """
    a = ((angulo + 180.0) % 360.0) - 180.0
    if abs(a) >= 135.0:
        return 'izquierda'
    if abs(a) <= 45.0:
        return 'derecha'
    if a > 0:
        return 'arriba'
    return 'abajo'


class MemoriaDireccion:
    """Cache de la ultima direccion indicada por una flecha confirmada.

    - Si llega una flecha confirmada -> guarda direccion y refresca TTL.
    - Si no llega flecha -> decrementa TTL hasta cero y olvida.
    - Mientras TTL > 0 la direccion sigue 'comprometida'.

    Devuelve un dict {'direccion', 'angulo', 'ttl_left', 'ttl_max',
    'frame_set'} o None si no hay memoria activa.
    """

    def __init__(self, ttl=300):
        self.ttl_max  = ttl
        self.ttl_left = 0
        self.direccion = None
        self.angulo    = None
        self.frame_set = -1

    def update(self, flecha_conf, frame_idx):
        if flecha_conf is not None:
            nueva_dir = direccion_de_flecha(flecha_conf['angulo'])
            # Log al cambiar de direccion o al primer commit
            if nueva_dir != self.direccion:
                print('[%5d] >>> DIR FIJADA: %s  (angulo %+.0f deg)'
                      % (frame_idx, nueva_dir, flecha_conf['angulo']))
            self.direccion = nueva_dir
            self.angulo    = float(flecha_conf['angulo'])
            self.ttl_left  = self.ttl_max
            self.frame_set = frame_idx
        elif self.ttl_left > 0:
            self.ttl_left -= 1
            if self.ttl_left == 0:
                print('[%5d] >>> DIR OLVIDADA (TTL agotado, ultima=%s)'
                      % (frame_idx, self.direccion))
                self.direccion = None
                self.angulo    = None

        if self.direccion is None:
            return None
        return {
            'direccion': self.direccion,
            'angulo'   : self.angulo,
            'ttl_left' : self.ttl_left,
            'ttl_max'  : self.ttl_max,
            'frame_set': self.frame_set,
        }


# ======================================================================
# CLASIFICACION DE MARCAS (man / stairs / telephone / woman / flecha) con LDA
# ======================================================================
# Descriptor de 14 dimensiones:
#   7 log-Hu moments  (invariantes a traslacion / rotacion / escala)
#   4 ratios geometricos (aspect ratio, fill, solidez, circularidad)
#   3 features Canny  (n esquinas, n lineas Hough, densidad de borde)
#
# Los 3 features Canny ayudan a distinguir las marcas con detalle interno:
#   - Escaleras (stairs) -> silueta zig-zag con MUCHAS esquinas
#   - Telefono -> contorno suave, pocas esquinas
#   - Personas -> esquinas en hombros/cabeza
def _descriptor_silueta(sil_bin):
    """Vector de 14 features de una silueta binaria (7 log-Hu + 4 ratios
    + 3 Canny)."""
    hu = cv2.HuMoments(cv2.moments(sil_bin, binaryImage=True)).flatten()
    log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    cnts, _ = cv2.findContours(sil_bin, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if perim <= 1 or area <= 1:
        return None
    _, _, bw, bh = cv2.boundingRect(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt)) or 1.0
    ratios = np.array([
        bh / bw if bw else 0.0,
        area / (bw * bh) if (bw and bh) else 0.0,
        area / hull_area,
        4.0 * math.pi * area / (perim * perim),
    ], dtype=np.float32)

    # ---- Features Canny aplicados a la silueta normalizada ----
    # Reescalo la silueta a tamano fijo para que las cuentas sean
    # comparables entre marcas grandes y chicas.
    REF = 100
    sil_n = cv2.resize(sil_bin, (REF, REF),
                       interpolation=cv2.INTER_NEAREST)
    edges = cv2.Canny(sil_n, 50, 150)
    densidad_borde = float(edges.sum() / 255.0) / (REF * REF)

    # Lineas Hough -> cuantas lineas rectas largas tiene el contorno
    lineas = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15,
                              minLineLength=12, maxLineGap=4)
    n_lineas = float(len(lineas)) if lineas is not None else 0.0

    # Esquinas Harris -> cuantas esquinas pronunciadas tiene el contorno
    sil_f = sil_n.astype(np.float32)
    harris = cv2.cornerHarris(sil_f, blockSize=3, ksize=3, k=0.04)
    n_esquinas = float((harris > 0.01 * harris.max()).sum())

    canny_feats = np.array([densidad_borde, n_lineas / 20.0,
                             n_esquinas / 100.0], dtype=np.float32)

    return np.concatenate([log_hu, ratios, canny_feats]).astype(np.float32)


def _generar_flechas_sinteticas():
    """Crea siluetas binarias de flechas en 18 rotaciones x 4 proporciones
    = 72 muestras. El descriptor de estas alimenta la clase 'flecha' del
    LDA, evitando que las flechas reales (a veces mal segmentadas) se
    clasifiquen como 'telephone' por descarte."""
    out = []
    variantes = [
        (1.0, 1.0, 1.0),   # estandar
        (1.5, 1.0, 1.0),   # asta larga
        (1.0, 1.4, 1.0),   # cabeza ancha
        (1.8, 0.8, 0.7),   # alargada estrecha
    ]
    for ang_deg in range(0, 360, 20):
        for largo_s, ancho_h, ancho_s in variantes:
            sz = 200
            img = np.zeros((sz, sz), dtype=np.uint8)
            cx, cy = sz // 2, sz // 2
            shaft = np.array([
                [-5 * ancho_s, 30 * largo_s],
                [ 5 * ancho_s, 30 * largo_s],
                [ 5 * ancho_s, -10],
                [-5 * ancho_s, -10],
            ], dtype=np.float32)
            head = np.array([
                [-15 * ancho_h, -10],
                [ 15 * ancho_h, -10],
                [  0,           -30],
            ], dtype=np.float32)
            th = math.radians(-ang_deg)
            R = np.array([[math.cos(th), -math.sin(th)],
                          [math.sin(th),  math.cos(th)]])
            shaft_r = (shaft @ R.T + [cx, cy]).astype(np.int32)
            head_r  = (head  @ R.T + [cx, cy]).astype(np.int32)
            cv2.fillPoly(img, [shaft_r], 255)
            cv2.fillPoly(img, [head_r],  255)
            out.append(img)
    return out


def _blob_principal_y_silueta(m_rojo, area_min):
    """Devuelve (sil_binaria, bbox) del componente conexo mas grande
    de la mascara roja, o None si no hay nada >= area_min."""
    if not m_rojo.any():
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(
        m_rojo.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    idx = int(np.argmax(st[1:, cv2.CC_STAT_AREA])) + 1
    area = int(st[idx, cv2.CC_STAT_AREA])
    if area < area_min:
        return None
    x = int(st[idx, cv2.CC_STAT_LEFT])
    y = int(st[idx, cv2.CC_STAT_TOP])
    w = int(st[idx, cv2.CC_STAT_WIDTH])
    h = int(st[idx, cv2.CC_STAT_HEIGHT])
    sil = (lab[y:y + h, x:x + w] == idx).astype(np.uint8) * 255
    return sil, (x, y, w, h)


def _augmentar_silueta(sil):
    """Genera variantes de una silueta binaria para data augmentation.

    Variantes: original + erosionada + dilatada. Esto simula la
    variabilidad en la calidad de segmentacion que vemos en el video
    real (a veces el blob queda mas pequeno por brillos, a veces mas
    grande por fugas de color a los bordes). Resultado: el LDA es mas
    robusto a estas variaciones.
    """
    out = [sil]
    out.append(cv2.erode(sil,  _KERNEL_3, iterations=1))
    out.append(cv2.dilate(sil, _KERNEL_3, iterations=1))
    return out


def entrenar_lda_marcas(clf_qda, ruta_dataset=RUTA_DATASET_MARCAS, area_min=80):
    """Entrena un LDA de 5 clases (man / stairs / telephone / woman / flecha).

    Usa el QDA ya entrenado para segmentar cada PNG del dataset, asi la
    silueta del entrenamiento es coherente con la del runtime. Cada
    imagen genera 3 muestras (original + erosionada + dilatada) para
    robustez ante variaciones de segmentacion.

    Devuelve un dict con:
      'clf'       : el LinearDiscriminantAnalysis entrenado
      'medias'    : centroides por clase (5 x D) para rechazo OOD
      'cov_inv'   : inversa de la covarianza pooled (D x D) para Mahalanobis
      'por_clase' : conteo de muestras por clase
    o None si no hay datos suficientes.
    """
    X, y = [], []
    por_clase = {c: 0 for c in CLASES_LDA}

    # ---- Marcas reales del dataset (segmentadas con QDA + augmentation) ----
    for f in sorted(glob.glob(os.path.join(ruta_dataset, '*.png'))):
        m = _RX_MARCA.match(os.path.basename(f))
        if not (m and m.group(1).lower() in CLASES_MARCAS):
            continue
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        _, m_rojo = segmentar_qda(clf_qda, bgr,
                                   area_lin_min=10 ** 8,  # ignorar linea
                                   area_mar_min=area_min)
        out = _blob_principal_y_silueta(m_rojo, area_min=area_min)
        if out is None:
            continue
        sil_base, _ = out
        nombre = m.group(1).lower()
        clase = CLASES_LDA.index(nombre)
        for sil_var in _augmentar_silueta(sil_base):
            desc = _descriptor_silueta(sil_var)
            if desc is None:
                continue
            X.append(desc); y.append(clase)
            por_clase[nombre] += 1

    # ---- Flechas sinteticas como clase 'flecha' ----
    for sil in _generar_flechas_sinteticas():
        desc = _descriptor_silueta(sil)
        if desc is None:
            continue
        X.append(desc); y.append(IDX_FLECHA)
        por_clase['flecha'] += 1

    if not X or len(set(y)) < 2:
        print('AVISO: dataset insuficiente para entrenar LDA de marcas')
        return None

    X = np.stack(X)
    y = np.array(y)
    clf = LinearDiscriminantAnalysis(solver='svd').fit(X, y)

    # Centroides por clase para rechazo OOD (distancia de Mahalanobis)
    D = X.shape[1]
    medias = np.zeros((len(CLASES_LDA), D), dtype=np.float32)
    for c in range(len(CLASES_LDA)):
        Xc = X[y == c]
        if len(Xc) > 0:
            medias[c] = Xc.mean(axis=0)
    cov = np.cov(X.T) + 0.05 * np.eye(D)        # regularizada
    cov_inv = np.linalg.inv(cov).astype(np.float32)

    return {
        'clf':       clf,
        'medias':    medias.astype(np.float32),
        'cov_inv':   cov_inv,
        'por_clase': por_clase,
    }


def predecir_marca(m_rojo, modelo, area_min=300, umbral_conf=0.80,
                   maha_max=4.0, h_img=None, y_top_max_frac=0.35):
    """Clasifica el blob rojo principal con el LDA de 5 clases + rechazo OOD.

    `modelo` es el dict devuelto por entrenar_lda_marcas: {clf, medias,
    cov_inv}.

    Devuelve (clase_nombre, conf, bbox) si la clase predicha es una de
    las 4 marcas reales. Devuelve None si:
       - no hay blob valido
       - el bbox empieza en la mitad superior de la imagen (lejano,
         dato ambiguo); solo confiamos en marcas cercanas al robot
       - la base del bbox toca el borde inferior
       - confianza < umbral_conf  (estricto: 0.80)
       - el LDA predijo 'flecha' (los blobs-flecha NO son marca)
       - distancia de Mahalanobis al centroide predicho > maha_max
         (estricto: 4.0). Esto bloquea los casos donde el LDA da
         conf 1.00 a algo que no se parece a ninguna marca real.
    """
    if modelo is None:
        return None
    out = _blob_principal_y_silueta(m_rojo, area_min=area_min)
    if out is None:
        return None
    sil, (x, y, w, h) = out
    if h_img is not None:
        if y < h_img * y_top_max_frac:    # blob lejano (mitad superior)
            return None
        if (y + h) >= (h_img - 2):        # toca el borde inferior
            return None
    desc = _descriptor_silueta(sil)
    if desc is None:
        return None

    clf, medias, cov_inv = modelo['clf'], modelo['medias'], modelo['cov_inv']
    probs = clf.predict_proba(desc.reshape(1, -1))[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    if conf < umbral_conf:
        return None
    if pred == IDX_FLECHA:                # LDA dice flecha -> no es marca
        return None

    # Out-Of-Distribution rejection (Mahalanobis al centroide predicho)
    diff = (desc - medias[pred]).astype(np.float32)
    maha = float(np.sqrt(max(0.0, diff @ cov_inv @ diff)))
    if maha > maha_max:
        return None

    return CLASES_LDA[pred], conf, (x, y, w, h)


class ConfirmadorMarca:
    """Confirma la marca por VOTO MAYORITARIO en una ventana movil.

    Mantiene historial de las ultimas `ventana` detecciones crudas.
    Solo reporta si alguna clase concentra >= `min_votos` votos. Asi
    tolera 1-2 frames con clase errada al inicio de la deteccion (p.ej.
    una escalera vista de canto que arranca pareciendose a un telefono
    y se estabiliza como stairs cuando entra completa en cuadro).

    Tras un report, cooldown de C frames sin permitir reportes nuevos
    de la misma clase (para no spamear).

    Tambien lleva un mini-cache de la ultima clase reportada para
    mostrar en el HUD (texto, NUNCA bbox -> el bbox se desactualiza
    cuando el robot se mueve).
    """

    def __init__(self, ventana=7, min_votos=5, cooldown=100, hud_ttl=20):
        self.ventana = ventana
        self.min_votos = min_votos
        self.cooldown_max = cooldown
        self.hud_ttl_max = hud_ttl
        self.historia    = []          # lista de (marca_o_None, clase_o_None)
        self.last_clase  = None
        self.cooldown    = 0
        # Mini cache solo para el HUD (clase + cuanto le queda visible)
        self.hud_clase   = None
        self.hud_conf    = 0.0
        self.hud_ttl     = 0

    def update(self, marca, frame_idx):
        if self.cooldown: self.cooldown -= 1
        if self.hud_ttl:  self.hud_ttl  -= 1

        clase_actual = marca[0] if marca else None
        self.historia.append((marca, clase_actual))
        if len(self.historia) > self.ventana:
            self.historia.pop(0)

        # Conteo de votos (excluyendo None)
        votos = {}
        for _, c in self.historia:
            if c is not None:
                votos[c] = votos.get(c, 0) + 1
        if not votos:
            return None
        clase_top = max(votos, key=votos.get)
        n_votos = votos[clase_top]
        if n_votos < self.min_votos:
            return None
        if clase_top == self.last_clase and self.cooldown > 0:
            return None

        # Tomar la deteccion MAS RECIENTE de esa clase (bbox actualizado)
        marca_rep = None
        for m, c in reversed(self.historia):
            if c == clase_top:
                marca_rep = m
                break
        if marca_rep is None:
            return None

        print('[%5d] >>> MARCA: %s  (conf %.2f, votos %d/%d)'
              % (frame_idx, clase_top, marca_rep[1], n_votos, self.ventana))
        self.last_clase = clase_top
        self.cooldown   = self.cooldown_max
        self.hud_clase  = clase_top
        self.hud_conf   = marca_rep[1]
        self.hud_ttl    = self.hud_ttl_max
        return marca_rep

    def cancelar_hud(self):
        """Apaga el indicador del HUD (e.g. cuando hay flecha activa)."""
        self.hud_ttl = 0
        self.hud_clase = None


# ======================================================================
# DETECCION DE EXTREMOS DEL CAMINO
# ======================================================================
def detectar_extremos(m_lin, banda_borde, min_segmento, fusion_gap):
    """Busca por donde sale la linea en cada borde de la imagen.

    Devuelve lista de dicts con:
      lado        : 'abajo' | 'arriba' | 'izquierda' | 'derecha'
      punto       : (x, y) en pixeles
      es_entrada  : True si lado == 'abajo' (por donde entra el robot)
      angulo      : grados desde el centro de la imagen al punto
    """
    h, w = m_lin.shape
    b, cx, cy = banda_borde, w / 2.0, h / 2.0

    bordes = (
        ('abajo',     m_lin[-b:, :].any(0), 'x', h - 1),
        ('arriba',    m_lin[:b,  :].any(0), 'x', 0),
        ('izquierda', m_lin[:, :b].any(1),  'y', 0),
        ('derecha',   m_lin[:, -b:].any(1), 'y', w - 1),
    )

    out = []
    for lado, perfil, eje, fija in bordes:
        d = np.diff(np.concatenate([[0], perfil.astype(np.int8), [0]]))
        ini = np.where(d == 1)[0]
        fin = np.where(d == -1)[0]
        segs = [(int(a), int(z)) for a, z in zip(ini, fin)
                if (z - a) >= min_segmento]
        if not segs:
            continue
        fus = [segs[0]]
        for s, e in segs[1:]:
            if s - fus[-1][1] <= fusion_gap:
                fus[-1] = (fus[-1][0], e)
            else:
                fus.append((s, e))
        for s, e in fus:
            pos = (s + e) // 2
            px, py = (pos, fija) if eje == 'x' else (fija, pos)
            out.append({
                'lado': lado,
                'punto': (int(px), int(py)),
                'es_entrada': lado == 'abajo',
                'angulo': math.degrees(math.atan2(cy - py, px - cx)),
            })
    return out


# ======================================================================
# OVERLAY VISUAL
# ======================================================================
def _texto(img, txt, pos, color, scale=0.4):
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def elegir_salida_por_direccion(extremos, memoria_dir, tol_grados=90.0):
    """Devuelve la salida que mejor encaja con la flecha memorizada.

    Estrategia:
      1) Si hay salidas en el lado exacto de la memoria
         (ej. memoria='izquierda' y hay extremo 'izquierda'),
         escoge entre esas la mas cercana al angulo de la flecha.
      2) Si no hay match de lado (caso comun: flecha 'arriba' en una Y
         que se ramifica antes del borde superior, asi solo hay extremos
         'izquierda' y 'derecha'), hace FALLBACK ANGULAR: escoge la
         salida cuyo angulo respecto al centro de la imagen este mas
         cerca del angulo de la flecha, siempre dentro de tol_grados.
      3) Sin memoria, o ninguna salida dentro de tolerancia -> None.
    """
    if memoria_dir is None:
        return None
    salidas = [e for e in extremos if not e['es_entrada']]
    if not salidas:
        return None
    ang_mem = memoria_dir['angulo']

    def _dang(e):
        return abs((e['angulo'] - ang_mem + 180) % 360 - 180)

    # 1) Match exacto de lado
    mismo_lado = [e for e in salidas if e['lado'] == memoria_dir['direccion']]
    if mismo_lado:
        return min(mismo_lado, key=_dang)

    # 2) Fallback angular: la salida cuyo angulo este mas cerca del de la flecha
    mejor = min(salidas, key=_dang)
    if _dang(mejor) <= tol_grados:
        return mejor
    return None


def overlay(bgr, m_lin, m_rojo, extremos, salida_elegida,
            flecha_cruda, flecha_conf, memoria_dir,
            marca_conf, marca_hud,
            franja_error, frame_idx, n_frames, fps):
    """Dibuja: mascaras + extremo entrada + salida ELEGIDA + flecha +
    marca + HUD.

    - marca_conf: deteccion ACTUAL de marca (tuple clase,conf,bbox)
      -> dibuja bbox magenta brillante alrededor del blob actual.
    - marca_hud: cache reciente (clase, conf) sin bbox -> SOLO se
      muestra en el texto del HUD, sin recuadro (porque el bbox viejo
      ya no esta donde estaba cuando se reporto)."""
    h, wi = bgr.shape[:2]

    # Mascaras semi-transparentes (azul=linea, rojo=flecha/marca)
    ov = bgr.copy()
    ov[m_lin]  = (255, 0, 0)
    ov[m_rojo] = (0, 0, 255)
    out = cv2.addWeighted(ov, 0.35, bgr, 0.65, 0)

    # Linea de la franja donde se mediria el error de seguimiento
    cv2.line(out, (0, h - franja_error), (wi, h - franja_error),
             (180, 180, 180), 1)

    # Extremos: ENT (verde), salida ELEGIDA (cyan grande), otras amarillas
    n_ent, n_sal = 0, 0
    for e in extremos:
        es_elegida = (salida_elegida is not None
                      and e['punto'] == salida_elegida['punto'])
        if e['es_entrada']:
            col, etq, radio, grosor = (0, 255, 0), 'ENT', 8, 2
            n_ent += 1
        elif es_elegida:
            # SALIDA ELEGIDA -> cyan brillante, grande, label ELEG
            col, etq, radio, grosor = (255, 255, 0), 'ELEG', 12, 3
            n_sal += 1
        else:
            # Otras salidas detectadas: visibles en amarillo con label S
            col, etq, radio, grosor = (0, 255, 255), 'S', 8, 2
            n_sal += 1

        # Reposicion del circulo para que no se corte en los bordes
        punto_draw = e['punto']
        if e['lado'] == 'abajo':
            punto_draw = (e['punto'][0], h - 10)
        elif e['lado'] == 'arriba':
            punto_draw = (e['punto'][0], 10)
        elif e['lado'] == 'izquierda':
            punto_draw = (10, e['punto'][1])
        elif e['lado'] == 'derecha':
            punto_draw = (wi - 10, e['punto'][1])

        cv2.circle(out, punto_draw, radio, col, grosor)
        tx, ty = punto_draw
        if   e['lado'] == 'arriba':    ty += 22
        elif e['lado'] == 'abajo':     ty -= 18
        elif e['lado'] == 'izquierda': tx += 18
        else:                          tx -= 64
        _texto(out, etq, (tx, ty), col, scale=0.45 if es_elegida else 0.4)

    # Flecha cruda (contorno + vector centro->punta), aunque no este confirmada
    if flecha_cruda is not None:
        # Confirmada -> amarillo brillante; sin confirmar todavia -> naranja
        col_flecha = (0, 255, 255) if flecha_conf is not None else (0, 165, 255)
        cv2.drawContours(out, [flecha_cruda['contorno']], -1, col_flecha, 1)
        cf = tuple(map(int, flecha_cruda['centro']))
        pf = tuple(map(int, flecha_cruda['punta']))
        cv2.arrowedLine(out, cf, pf, col_flecha, 2, tipLength=0.30)
        # Etiqueta con angulo cerca de la punta
        etq = 'FLECHA %+.0f deg%s' % (flecha_cruda['angulo'],
                                       '' if flecha_conf else ' ?')
        _texto(out, etq, (pf[0] + 6, max(12, pf[1] - 6)), col_flecha)

    # Marca: bbox magenta SOLO cuando hay deteccion actual.
    # La marca de memoria (marca_hud) NO dibuja bbox: el blob se movio.
    if marca_conf is not None:
        clase, conf, (mx, my, mw, mh) = marca_conf
        cv2.rectangle(out, (mx, my), (mx + mw, my + mh), (255, 0, 200), 2)
        etq_m = '%s %.0f%%' % (clase, conf * 100)
        _texto(out, etq_m, (mx, max(12, my - 4)), (255, 0, 200), scale=0.45)

    # HUD esquina superior
    ang_s = ('%+5.0f deg' % flecha_conf['angulo']) if flecha_conf else '--'
    dir_s = ('--' if memoria_dir is None
             else '%s (TTL %d/%d)' % (memoria_dir['direccion'],
                                       memoria_dir['ttl_left'],
                                       memoria_dir['ttl_max']))
    eleg_s = (salida_elegida['lado'] if salida_elegida else '--')
    if marca_conf is not None:
        marca_s = '%s %.0f%%' % (marca_conf[0], marca_conf[1] * 100)
    elif marca_hud is not None:
        marca_s = '%s (mem)' % marca_hud[0]
    else:
        marca_s = '--'
    lineas = (
        'FRAME %d/%d  (~%.0f fps)' % (frame_idx, n_frames, fps),
        'BrainFollowLine-3 FINAL  (linea+flecha+salida+marca)',
        'Entradas: %d   Salidas detectadas: %d' % (n_ent, n_sal),
        'Flecha conf: ' + ang_s,
        'Memoria dir: ' + dir_s,
        'SALIDA ELEG: ' + eleg_s,
        'MARCA      : ' + marca_s,
    )
    for i, txt in enumerate(lineas):
        _texto(out, txt, (4, 14 + i * 14), (255, 255, 255))

    return out


# ======================================================================
# RUNNER OFFLINE (lee video local, escribe video anotado)
# ======================================================================
class BrainOffline:

    # Parametros relativos a la resolucion del video (referencia 320x240)
    def __init__(self, video_path, output_path=None, mostrar=True):
        self.mostrar = mostrar

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError('No se pudo abrir: ' + video_path)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_video = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Escala de parametros segun resolucion
        scale_w   = self.w / 320.0
        scale_h   = self.h / 240.0
        scale_avg = (scale_w + scale_h) / 2.0
        scale_area = scale_w * scale_h

        self.FRANJA_ERROR  = max(20, int(40 * scale_h))
        self.BANDA_BORDE   = max(3,  int(4 * scale_avg))
        self.MIN_SEGMENTO  = max(4,  int(5 * scale_w))
        self.FUSION_GAP    = max(5,  int(8 * scale_w))
        self.AREA_MIN_LINEA  = max(100, int(300 * scale_area))
        self.AREA_MIN_ROJO   = max(80,  int(150 * scale_area))
        self.AREA_MIN_FLECHA = max(80,  int(200 * scale_area))
        # Marca con area mayor: marcas cercanas al robot ocupan mas pixeles,
        # eso evita clasificar blobs lejanos/ambiguos.
        self.AREA_MIN_MARCA  = max(300, int(800 * scale_area))

        # Confirmador temporal de flecha (2 frames consecutivos similares)
        self.confirmador_flecha = ConfirmadorFlecha(n_frames=2,
                                                     tol_grados=30.0)

        # Memoria de direccion: TTL escalado al fps real del video
        # (~12 segundos), con piso de 150 y techo de 600 frames.
        ttl = int(max(150, min(600, 12.0 * self.fps_video)))
        self.memoria_direccion = MemoriaDireccion(ttl=ttl)

        # PASO 1-bis: QDA de segmentacion (3 clases: fondo/marca/linea).
        # Reemplaza los rangos HSV manuales con un clasificador que aprende
        # la distribucion conjunta de colores -> mas robusto a iluminacion.
        print('Entrenando QDA de segmentacion (3 clases)...')
        self.clf_qda = entrenar_qda_linea()
        print('QDA listo.')

        # PASO 4: clasificador LDA de marcas (5 clases) + confirmador.
        # Parametros ESTRICTOS para minimizar falsos positivos:
        #   - voto mayoritario 5 de 7 frames (tolera 1-2 fallos iniciales)
        #   - cooldown 100 frames despues de reportar
        #   - confianza minima 0.80, Mahalanobis maxima 4.0
        #   - solo blobs en la mitad inferior de la imagen (cercanos)
        print('Entrenando LDA de marcas (5 clases incluyendo flecha)...')
        self.modelo_marcas = entrenar_lda_marcas(self.clf_qda)
        if self.modelo_marcas is not None:
            print('LDA marcas listo. Muestras por clase: %s'
                  % self.modelo_marcas['por_clase'])
        else:
            print('AVISO: LDA marcas no disponible')
        self.MARCA_UMBRAL_CONF   = 0.80
        self.MARCA_MAHA_MAX      = 4.0
        self.MARCA_Y_TOP_MAX_FRAC = 0.35
        self.confirmador_marca = ConfirmadorMarca(ventana=7, min_votos=5,
                                                   cooldown=100, hud_ttl=20)

        print('Video %dx%d @ %.1f fps (%d frames)'
              % (self.w, self.h, self.fps_video, self.n_frames))
        print('Parametros: FRANJA=%d  BORDE=%d  MIN_SEG=%d  FUSION=%d  '
              'AREA_LIN=%d  AREA_ROJO=%d  AREA_FLE=%d'
              % (self.FRANJA_ERROR, self.BANDA_BORDE, self.MIN_SEGMENTO,
                 self.FUSION_GAP, self.AREA_MIN_LINEA,
                 self.AREA_MIN_ROJO, self.AREA_MIN_FLECHA))

        self.writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc,
                                          self.fps_video, (self.w, self.h))
            print('Output -> %s' % output_path)

        self.frame_idx = 0
        self._t_prev = cv2.getTickCount()
        self._fps_inst = 0.0

        # Contadores acumulados para diagnostico
        self.n_frames_con_rojo    = 0
        self.n_frames_flecha_cruda = 0
        self.n_frames_flecha_conf  = 0
        self.n_frames_con_memoria  = 0
        self.n_frames_con_eleg     = 0
        self.n_frames_marca_cruda  = 0
        self.n_frames_marca_conf   = 0
        self.marcas_reportadas     = []  # (frame, clase, conf)
        self.diag_flecha = {}

    def step(self):
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return False
        self.frame_idx += 1

        m_lin, m_rojo = segmentar_qda(self.clf_qda, bgr,
                                       area_lin_min=self.AREA_MIN_LINEA,
                                       area_mar_min=self.AREA_MIN_ROJO)

        extremos = detectar_extremos(m_lin, self.BANDA_BORDE,
                                     self.MIN_SEGMENTO, self.FUSION_GAP)

        flecha_cruda = detectar_flecha(m_rojo, area_min=self.AREA_MIN_FLECHA,
                                       diag=self.diag_flecha)
        flecha_conf  = self.confirmador_flecha.update(flecha_cruda)
        memoria_dir  = self.memoria_direccion.update(flecha_conf, self.frame_idx)
        salida_elegida = elegir_salida_por_direccion(extremos, memoria_dir)

        # PASO 4: marca solo si el blob rojo NO es una flecha (canal separado)
        marca_cruda = None
        if flecha_cruda is None:
            marca_cruda = predecir_marca(m_rojo, self.modelo_marcas,
                                          area_min=self.AREA_MIN_MARCA,
                                          umbral_conf=self.MARCA_UMBRAL_CONF,
                                          maha_max=self.MARCA_MAHA_MAX,
                                          h_img=self.h,
                                          y_top_max_frac=self.MARCA_Y_TOP_MAX_FRAC)
        marca_conf_val = self.confirmador_marca.update(marca_cruda,
                                                       self.frame_idx)
        # Si hay flecha visible, anular el HUD de marca: NO mostrar nada
        # de marca encima de una flecha en curso.
        if flecha_cruda is not None:
            self.confirmador_marca.cancelar_hud()

        # Texto de marca para el HUD (clase + conf), SIN bbox.
        # El bbox se desactualiza cuando el robot se mueve, asi que solo
        # dibujamos bbox cuando hay deteccion ACTUAL (marca_conf_val).
        marca_hud = None
        if marca_conf_val is None and self.confirmador_marca.hud_ttl > 0:
            marca_hud = (self.confirmador_marca.hud_clase,
                         self.confirmador_marca.hud_conf)

        if m_rojo.any():               self.n_frames_con_rojo += 1
        if flecha_cruda is not None:   self.n_frames_flecha_cruda += 1
        if flecha_conf  is not None:   self.n_frames_flecha_conf  += 1
        if memoria_dir  is not None:   self.n_frames_con_memoria  += 1
        if salida_elegida is not None: self.n_frames_con_eleg     += 1
        if marca_cruda is not None:    self.n_frames_marca_cruda  += 1
        if marca_conf_val is not None:
            self.n_frames_marca_conf  += 1
            self.marcas_reportadas.append((self.frame_idx, marca_conf_val[0],
                                            marca_conf_val[1]))

        if self.frame_idx % 10 == 0:
            t_now = cv2.getTickCount()
            dt = (t_now - self._t_prev) / cv2.getTickFrequency()
            self._fps_inst = 10.0 / max(dt, 1e-6)
            self._t_prev = t_now

        out = overlay(bgr, m_lin, m_rojo, extremos, salida_elegida,
                      flecha_cruda, flecha_conf, memoria_dir,
                      marca_conf_val, marca_hud,
                      self.FRANJA_ERROR,
                      self.frame_idx, self.n_frames, self._fps_inst)

        if self.writer is not None:
            self.writer.write(out)
        if self.mostrar:
            try:
                cv2.imshow('BrainFollowLine-3 FINAL', out)
                if cv2.waitKey(1) & 0xFF == 27:
                    return False
            except Exception:
                self.mostrar = False

        if self.frame_idx % 30 == 0:
            n_ent = sum(1 for e in extremos if e['es_entrada'])
            n_sal = len(extremos) - n_ent
            flecha_s = ('CONF %+5.0f' % flecha_conf['angulo']) if flecha_conf \
                       else ('cruda %+5.0f' % flecha_cruda['angulo']) if flecha_cruda \
                       else '--'
            mem_s = ('%s(%d)' % (memoria_dir['direccion'], memoria_dir['ttl_left'])
                     if memoria_dir else '--')
            eleg_s = salida_elegida['lado'] if salida_elegida else '--'
            marca_s = ('%s %.0f%%' % (marca_conf_val[0], marca_conf_val[1] * 100)
                       if marca_conf_val else '--')
            print('[%5d] ENT=%d S=%d rojo=%d flecha=%s dir=%s eleg=%s marca=%s'
                  % (self.frame_idx, n_ent, n_sal,
                     int(m_rojo.sum()), flecha_s, mem_s, eleg_s, marca_s))
        return True

    def cerrar(self):
        self.cap.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()
        total = max(self.frame_idx, 1)
        print('\n=== Resumen BrainFollowLine-3 FINAL ===')
        print('Frames totales         : %d' % self.frame_idx)
        print('Frames con rojo        : %d (%.1f%%)'
              % (self.n_frames_con_rojo,
                 100.0 * self.n_frames_con_rojo / total))
        print('Frames flecha (cruda)  : %d (%.1f%%)'
              % (self.n_frames_flecha_cruda,
                 100.0 * self.n_frames_flecha_cruda / total))
        print('Frames flecha (conf)   : %d (%.1f%%)'
              % (self.n_frames_flecha_conf,
                 100.0 * self.n_frames_flecha_conf / total))
        print('Frames con DIR memoria : %d (%.1f%%)'
              % (self.n_frames_con_memoria,
                 100.0 * self.n_frames_con_memoria / total))
        print('Frames con SALIDA ELEG : %d (%.1f%%)'
              % (self.n_frames_con_eleg,
                 100.0 * self.n_frames_con_eleg / total))
        print('Frames marca (cruda)   : %d (%.1f%%)'
              % (self.n_frames_marca_cruda,
                 100.0 * self.n_frames_marca_cruda / total))
        print('Frames marca (conf)    : %d (%.1f%%)'
              % (self.n_frames_marca_conf,
                 100.0 * self.n_frames_marca_conf / total))
        if self.marcas_reportadas:
            print('Marcas reportadas: %d' % len(self.marcas_reportadas))
            for (fi, cls, conf) in self.marcas_reportadas:
                print('  frame %5d  %-10s  conf=%.2f' % (fi, cls, conf))
        if self.diag_flecha:
            print('Motivos de rechazo del blob rojo principal:')
            for k in sorted(self.diag_flecha, key=lambda x: -self.diag_flecha[x]):
                print('  %-15s %5d  (%.1f%%)'
                      % (k, self.diag_flecha[k],
                         100.0 * self.diag_flecha[k] / total))


def main():
    p = argparse.ArgumentParser(
        description='BrainFollowLine-3 FINAL: linea + flecha + memoria dir + marca.')
    p.add_argument('--video', choices=['3', '4'], default='3',
                   help='Que video de prueba usar (default: 3)')
    p.add_argument('--sin-output', action='store_true',
                   help='No genera video de salida')
    p.add_argument('--no-mostrar', action='store_true',
                   help='No abre la ventana (mas rapido)')
    p.add_argument('--start', type=int, default=0,
                   help='Salta los primeros N frames')
    args = p.parse_args()

    video_in = os.path.join(_AQUI, 'video2017-%s.mp4' % args.video)
    if not os.path.exists(video_in):
        print('ERROR: no encuentro %s' % video_in)
        sys.exit(1)

    output = None
    if not args.sin_output:
        output = os.path.join(_AQUI, 'output_final_video2017-%s.mp4' % args.video)

    runner = BrainOffline(video_in, output_path=output,
                         mostrar=not args.no_mostrar)
    for _ in range(args.start):
        if not runner.cap.read()[0]:
            break

    while runner.step():
        pass
    runner.cerrar()
    print('Listo. BrainFollowLine-3 FINAL completado.')


if __name__ == '__main__':
    main()
