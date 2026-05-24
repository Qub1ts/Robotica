"""BrainFollowLine.py - Cerebro final para el robot real (Pioneer).

Pipeline integrado (mismo que BrainFollowLine-3.py FINAL):

  PERCEPCION (offline-equivalente):
    1) Segmentacion por QDA (3 clases: fondo / marca-roja / linea-azul)
       con CLAHE + threshold de confianza por pixel.
    2) Deteccion de extremos del camino (ENT abajo, S resto de bordes).
    3) Deteccion geometrica de FLECHA + confirmacion temporal.
    4) MEMORIA de la direccion indicada por la flecha (TTL ~12s).
    5) SALIDA ELEGIDA = la salida del camino que mejor encaja con la
       direccion memorizada (match lado exacto + fallback angular).
    6) CLASIFICACION DE MARCAS (man / stairs / telephone / woman) con
       LDA de 5 clases (la 5ta es 'flecha' sintetica) + rechazo OOD por
       Mahalanobis + dataset aumentado + filtro de zona inferior.
       Voto mayoritario 5 de 7 frames para confirmar.

  CONTROL DE MOTORES:
    - Evasion frontal con sonar (DIST_FRONTAL_OBST).
    - En cruce con salida elegida -> gira hacia esa salida.
    - Sin error de linea -> spin para buscarla.
    - Caso normal -> seguimiento PD de la franja inferior.

ENTRADA:
    /dev/video0  (camara C920, 640x360 @ 30 fps por defecto)
SALIDA:
    self.move(v, w) sobre el motor del Pioneer.
    Ventana overlay si hay GUI disponible.
"""

from pyrobot.brain import Brain

import glob
import math
import os
import re

import cv2
import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)


# ======================================================================
# CONSTANTES Y RUTAS
# ======================================================================
_AQUI = os.path.dirname(os.path.abspath(__file__))
RUTA_IMG_ORIGINAL   = os.path.join(_AQUI, 'imagen_original.png')
RUTA_IMG_MARCADA    = os.path.join(_AQUI, 'imagen_marcada.png')
RUTA_DATASET_MARCAS = os.path.join(_AQUI, 'marcas-capturasStage')
CLASES_MARCAS = ('man', 'stairs', 'telephone', 'woman')
CLASES_LDA    = CLASES_MARCAS + ('flecha',)
IDX_FLECHA    = CLASES_LDA.index('flecha')
_RX_MARCA = re.compile(r'^([a-z]+)[-_]\d+\.png$', re.IGNORECASE)

_KERNEL_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_KERNEL_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_CLAHE    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _clamp(valor, minimo=-1.0, maximo=1.0):
    return max(minimo, min(maximo, valor))


def aplicar_clahe(bgr):
    """Ecualiza el canal L de Lab para uniformar brillo sin cambiar tono."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _filtrar_componentes(mask, area_min):
    mask = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            out[labels == i] = 255
    return out > 0


# ======================================================================
# QDA: SEGMENTACION POR COLOR (fondo / marca-roja / linea-azul)
# ======================================================================
def _features_pixel(rgb):
    """RGB -> matriz N x 9 de features (HSV + a*b* Lab + RGB norm + L)."""
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
    """Entrena QDA 3-clases con imagen_original.png + imagen_marcada.png."""
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


def segmentar_qda(clf, bgr, area_lin_min=750, area_mar_min=350,
                  conf_min=0.55):
    """Aplica el QDA al frame y devuelve (m_lin, m_rojo) booleanas.

    Threshold de confianza: pixeles con max_proba < conf_min van al
    fondo. Asi evitamos pintar como linea o marca pixeles ambiguos
    (zonas de transicion donde el rojo es casi-azul por brillo o sombra).
    """
    bgr_eq = aplicar_clahe(bgr)
    rgb = cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2RGB)
    feats = _features_pixel(rgb)

    proba = clf.predict_proba(feats)
    pred  = np.argmax(proba, axis=1)
    pmax  = proba[np.arange(len(pred)), pred]
    pred[pmax < conf_min] = 0
    pred = pred.reshape(rgb.shape[:2])

    m_lin = (pred == 2).astype(np.uint8) * 255
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_OPEN,  _KERNEL_3, iterations=1)
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_CLOSE, _KERNEL_5, iterations=2)
    m_lin = _filtrar_componentes(m_lin, area_min=area_lin_min)

    m_rojo = (pred == 1).astype(np.uint8) * 255
    m_rojo = cv2.morphologyEx(m_rojo, cv2.MORPH_OPEN,  _KERNEL_3, iterations=1)
    m_rojo = cv2.morphologyEx(m_rojo, cv2.MORPH_CLOSE, _KERNEL_5, iterations=3)
    m_rojo = _filtrar_componentes(m_rojo, area_min=area_mar_min)

    return m_lin, m_rojo


# ======================================================================
# DETECCION GEOMETRICA DE FLECHA + CONFIRMADOR
# ======================================================================
FLECHA_CIRC_MAX    = 0.50
FLECHA_ELONG_MIN   = 2.0
FLECHA_ASIM_MIN    = 0.30
FLECHA_SOLIDEZ_MIN = 0.65


def detectar_flecha(m_rojo, area_min):
    """Si el blob rojo principal pasa los 4 filtros geometricos
    (circularidad, solidez, elongacion, asimetria) devuelve dict con
    angulo + centro + punta + contorno. Si no, None."""
    if not m_rojo.any():
        return None
    cnts, _ = cv2.findContours(m_rojo.astype(np.uint8),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < area_min or len(cnt) < 5:
        return None
    perim = cv2.arcLength(cnt, True)
    circ  = 4.0 * math.pi * area / max(perim * perim, 1e-6)
    if circ > FLECHA_CIRC_MAX:
        return None
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull) or 1.0
    if (area / hull_area) < FLECHA_SOLIDEZ_MIN:
        return None

    mask = np.zeros(m_rojo.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    eigval, eigvec = np.linalg.eigh(np.cov(pts.T))
    elong = math.sqrt(float(max(eigval)) / float(max(min(eigval), 1e-6)))
    if elong < FLECHA_ELONG_MIN:
        return None
    eje  = eigvec[:, int(np.argmax(eigval))]
    perp = np.array([-eje[1], eje[0]])
    proy_a = (pts[:, 0] - cx) * eje[0]  + (pts[:, 1] - cy) * eje[1]
    proy_p = (pts[:, 0] - cx) * perp[0] + (pts[:, 1] - cy) * perp[1]
    pmin, pmax = float(proy_a.min()), float(proy_a.max())
    L = pmax - pmin
    if L < 5:
        return None
    franja = 0.25 * L
    pp_pos = proy_p[proy_a > pmax - franja]
    pp_neg = proy_p[proy_a < pmin + franja]
    s_pos = float(pp_pos.max() - pp_pos.min()) if len(pp_pos) >= 5 else 0.0
    s_neg = float(pp_neg.max() - pp_neg.min()) if len(pp_neg) >= 5 else 0.0
    max_s = max(s_pos, s_neg)
    if max_s < 1e-3 or abs(s_pos - s_neg) / max_s < FLECHA_ASIM_MIN:
        return None
    idx = int(np.argmax(proy_a)) if s_pos > s_neg else int(np.argmin(proy_a))
    px, py = float(pts[idx, 0]), float(pts[idx, 1])
    return {
        'angulo':   math.degrees(math.atan2(cy - py, px - cx)),
        'centro':   (cx, cy),
        'punta':    (px, py),
        'contorno': cnt,
    }


class ConfirmadorFlecha:
    """Confirma flecha tras N frames con angulo similar (tolerancia)."""

    def __init__(self, n_frames=2, tol_grados=30.0):
        self.n_frames = n_frames
        self.tol = tol_grados
        self._ang = None
        self._cnt = 0

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
    """atan2-style en grados -> 'izquierda' | 'derecha' | 'arriba' | 'abajo'."""
    a = ((angulo + 180.0) % 360.0) - 180.0
    if abs(a) >= 135.0:
        return 'izquierda'
    if abs(a) <= 45.0:
        return 'derecha'
    if a > 0:
        return 'arriba'
    return 'abajo'


class MemoriaDireccion:
    """Cache de la ultima direccion confirmada por una flecha, con TTL."""

    def __init__(self, ttl=300):
        self.ttl_max  = ttl
        self.ttl_left = 0
        self.direccion = None
        self.angulo    = None
        self.frame_set = -1

    def update(self, flecha_conf, frame_idx):
        if flecha_conf is not None:
            nueva_dir = direccion_de_flecha(flecha_conf['angulo'])
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
            'angulo':    self.angulo,
            'ttl_left':  self.ttl_left,
            'ttl_max':   self.ttl_max,
            'frame_set': self.frame_set,
        }


# ======================================================================
# LDA DE MARCAS (5 clases: man / stairs / telephone / woman / flecha)
# ======================================================================
def _descriptor_silueta(sil_bin):
    """Vector 14-D: 7 log-Hu + 4 ratios + 3 features Canny normalizados."""
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

    REF = 100
    sil_n = cv2.resize(sil_bin, (REF, REF),
                       interpolation=cv2.INTER_NEAREST)
    edges = cv2.Canny(sil_n, 50, 150)
    densidad_borde = float(edges.sum() / 255.0) / (REF * REF)
    lineas = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15,
                              minLineLength=12, maxLineGap=4)
    n_lineas = float(len(lineas)) if lineas is not None else 0.0
    sil_f = sil_n.astype(np.float32)
    harris = cv2.cornerHarris(sil_f, blockSize=3, ksize=3, k=0.04)
    n_esquinas = float((harris > 0.01 * harris.max()).sum())
    canny_feats = np.array([densidad_borde, n_lineas / 20.0,
                             n_esquinas / 100.0], dtype=np.float32)
    return np.concatenate([log_hu, ratios, canny_feats]).astype(np.float32)


def _generar_flechas_sinteticas():
    """72 siluetas binarias de flechas (18 rotaciones x 4 proporciones)."""
    out = []
    variantes = [(1.0, 1.0, 1.0), (1.5, 1.0, 1.0),
                 (1.0, 1.4, 1.0), (1.8, 0.8, 0.7)]
    for ang_deg in range(0, 360, 20):
        for largo_s, ancho_h, ancho_s in variantes:
            sz = 200
            img = np.zeros((sz, sz), dtype=np.uint8)
            cx, cy = sz // 2, sz // 2
            shaft = np.array([
                [-5 * ancho_s, 30 * largo_s], [ 5 * ancho_s, 30 * largo_s],
                [ 5 * ancho_s, -10],          [-5 * ancho_s, -10],
            ], dtype=np.float32)
            head = np.array([
                [-15 * ancho_h, -10], [ 15 * ancho_h, -10], [0, -30],
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
    """Original + erosionada + dilatada -> robustez ante segmentacion variable."""
    return [sil,
            cv2.erode(sil,  _KERNEL_3, iterations=1),
            cv2.dilate(sil, _KERNEL_3, iterations=1)]


def entrenar_lda_marcas(clf_qda, ruta_dataset=RUTA_DATASET_MARCAS, area_min=80):
    """LDA de 5 clases + centroides + cov_inv para rechazo OOD por Mahalanobis."""
    X, y = [], []
    por_clase = {c: 0 for c in CLASES_LDA}
    for f in sorted(glob.glob(os.path.join(ruta_dataset, '*.png'))):
        m = _RX_MARCA.match(os.path.basename(f))
        if not (m and m.group(1).lower() in CLASES_MARCAS):
            continue
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        _, m_rojo = segmentar_qda(clf_qda, bgr,
                                   area_lin_min=10 ** 8,
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
    for sil in _generar_flechas_sinteticas():
        desc = _descriptor_silueta(sil)
        if desc is None:
            continue
        X.append(desc); y.append(IDX_FLECHA)
        por_clase['flecha'] += 1

    if not X or len(set(y)) < 2:
        print('AVISO: dataset insuficiente para entrenar LDA')
        return None

    X = np.stack(X); y = np.array(y)
    clf = LinearDiscriminantAnalysis(solver='svd').fit(X, y)
    D = X.shape[1]
    medias = np.zeros((len(CLASES_LDA), D), dtype=np.float32)
    for c in range(len(CLASES_LDA)):
        Xc = X[y == c]
        if len(Xc) > 0:
            medias[c] = Xc.mean(axis=0)
    cov = np.cov(X.T) + 0.05 * np.eye(D)
    cov_inv = np.linalg.inv(cov).astype(np.float32)
    return {'clf': clf, 'medias': medias.astype(np.float32),
            'cov_inv': cov_inv, 'por_clase': por_clase}


def predecir_marca(m_rojo, modelo, area_min=300, umbral_conf=0.80,
                   maha_max=4.0, h_img=None, y_top_max_frac=0.35):
    """Devuelve (clase, conf, bbox) o None. Aplica:
      - solo blobs en la mitad inferior (cercanos al robot)
      - confianza >= umbral_conf (estricto 0.80)
      - LDA != 'flecha'  (las flechas no son marca)
      - Mahalanobis al centroide <= maha_max (rechazo OOD)
    """
    if modelo is None:
        return None
    out = _blob_principal_y_silueta(m_rojo, area_min=area_min)
    if out is None:
        return None
    sil, (x, y, w, h) = out
    if h_img is not None:
        if y < h_img * y_top_max_frac:
            return None
        if (y + h) >= (h_img - 2):
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
    if pred == IDX_FLECHA:
        return None
    diff = (desc - medias[pred]).astype(np.float32)
    maha = float(np.sqrt(max(0.0, diff @ cov_inv @ diff)))
    if maha > maha_max:
        return None
    return CLASES_LDA[pred], conf, (x, y, w, h)


class ConfirmadorMarca:
    """Voto mayoritario en ventana movil para reportar marca."""

    def __init__(self, ventana=7, min_votos=5, cooldown=100, hud_ttl=20):
        self.ventana = ventana
        self.min_votos = min_votos
        self.cooldown_max = cooldown
        self.hud_ttl_max = hud_ttl
        self.historia    = []
        self.last_clase  = None
        self.cooldown    = 0
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
        self.hud_ttl = 0
        self.hud_clase = None


# ======================================================================
# EXTREMOS DEL CAMINO + SALIDA ELEGIDA
# ======================================================================
def detectar_extremos(m_lin, banda_borde, min_segmento, fusion_gap):
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
                'lado': lado, 'punto': (int(px), int(py)),
                'es_entrada': lado == 'abajo',
                'angulo': math.degrees(math.atan2(cy - py, px - cx)),
            })
    return out


def elegir_salida_por_direccion(extremos, memoria_dir, tol_grados=90.0):
    """Match exacto de lado, fallback angular dentro de tol_grados."""
    if memoria_dir is None:
        return None
    salidas = [e for e in extremos if not e['es_entrada']]
    if not salidas:
        return None
    ang_mem = memoria_dir['angulo']

    def _dang(e):
        return abs((e['angulo'] - ang_mem + 180) % 360 - 180)

    mismo_lado = [e for e in salidas if e['lado'] == memoria_dir['direccion']]
    if mismo_lado:
        return min(mismo_lado, key=_dang)
    mejor = min(salidas, key=_dang)
    if _dang(mejor) <= tol_grados:
        return mejor
    return None


# ======================================================================
# CEREBRO PRINCIPAL (Brain de pyrobot)
# ======================================================================
class BrainFollowLine(Brain):

    # ---- Camara C920 ----
    CAM_W, CAM_H = 640, 360

    # ---- Velocidades y PD ----
    SLOW_FORWARD, FULL_FORWARD = 0.08, 0.40
    KP        = 1.2
    CRUCE_KP  = 1.8

    # ---- Percepcion (tunneados para 640x360) ----
    FRANJA_ERROR     = 60
    BANDA_BORDE      = 6
    MIN_SEGMENTO     = 12
    FUSION_GAP       = 18
    JUNCTION_Y_FRAC  = 0.75
    JUNCTION_MIN_RUN = 6
    SIDE_EXIT_Y_FRAC = 0.70

    AREA_MIN_LINEA  = 1500
    AREA_MIN_ROJO   = 600
    AREA_MIN_FLECHA = 800
    AREA_MIN_MARCA  = 1600

    # ---- LDA marcas (estricto para minimizar falsos positivos) ----
    MARCA_UMBRAL_CONF    = 0.80
    MARCA_MAHA_MAX       = 4.0
    MARCA_Y_TOP_MAX_FRAC = 0.35

    # ---- Memoria flecha ----
    ARROW_TTL = 288     # ~12s a 24fps

    # ---- Sonar / evasion ----
    DIST_FRONTAL_OBST    = 0.50
    DIST_FRENTE_LIBRE    = 0.45
    DIST_OBJETIVO_PARED  = 0.30
    AVOID_TICKS_MIN      = 40
    AVOID_TICKS_MAX      = 250
    AVOID_EXIT_ERR_MAX   = 0.50
    AVOID_FLAG           = 1.0
    POST_AVOID_GRACE     = 60
    PRINT_EVERY_N_FRAMES = 15

    def setup(self):
        print('Cargando QDA (segmentacion) y LDA (marcas)...')
        self.clf_qda       = entrenar_qda_linea()
        self.modelo_marcas = entrenar_lda_marcas(self.clf_qda)
        if self.modelo_marcas is not None:
            print('LDA marcas listo. Muestras por clase: %s'
                  % self.modelo_marcas['por_clase'])
        else:
            print('AVISO: LDA marcas no disponible')

        self.confirmador_flecha   = ConfirmadorFlecha(n_frames=2, tol_grados=30.0)
        self.memoria_direccion    = MemoriaDireccion(ttl=self.ARROW_TTL)
        self.confirmador_marca    = ConfirmadorMarca(ventana=7, min_votos=5,
                                                      cooldown=100, hud_ttl=20)

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,  self.CAM_W)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_H)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        if not self.capture.isOpened():
            print('ERROR: no se pudo abrir la camara /dev/video0')
        else:
            print('Camara real abierta correctamente (%dx%d)'
                  % (self.CAM_W, self.CAM_H))

        # Estado del controlador
        self.prev_error       = None
        self.last_error       = 0.0
        self.avoiding         = False
        self.avoid_ticks      = 0
        self.post_avoid_grace = 0
        self.marcas_vistas    = []
        self._frame_idx       = 0
        self._gui_ok          = True

    def destroy(self):
        self.move(0.0, 0.0)
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()

    # ----- Sonar -----
    def leer_distancias(self):
        try:
            min_front = min(self.robot.range[i].distance() for i in range(2, 6))
            min_left  = min(self.robot.range[i].distance() for i in range(0, 3))
        except Exception:
            min_front, min_left = 99.0, 99.0
        return min_front, min_left

    # ----- Helpers percepcion -----
    def calcular_error_linea(self, m_lin, cx_img):
        franja = m_lin[-self.FRANJA_ERROR:, :].astype(np.uint8)
        M = cv2.moments(franja, binaryImage=True)
        if M['m00'] < 1:
            return None
        return float((M['m10'] / M['m00'] - cx_img) / cx_img)

    def junction_y(self, m_lin):
        min_run = self.JUNCTION_MIN_RUN
        for y in range(m_lin.shape[0] - 1, -1, -1):
            d = np.diff(np.concatenate([[0], m_lin[y].astype(np.uint8), [0]]))
            ini = np.where(d == 1)[0]
            fin = np.where(d == -1)[0]
            n = 0
            for s, e in zip(ini, fin):
                if (e - s) >= min_run:
                    n += 1
                    if n >= 2:
                        return int(y)
        return None

    def hay_cruce_inminente(self, m_lin, salidas, h_img):
        if self.post_avoid_grace != 0 or len(salidas) < 2:
            return False
        jy = self.junction_y(m_lin)
        if jy is not None and jy >= h_img * self.JUNCTION_Y_FRAC:
            return True
        lim_y = h_img * self.SIDE_EXIT_Y_FRAC
        for s in salidas:
            if s['lado'] in ('izquierda', 'derecha') and s['punto'][1] >= lim_y:
                return True
        return False

    # ----- Overlay (solo si hay GUI) -----
    def overlay(self, bgr, m_lin, m_rojo, extremos, salida_elegida,
                flecha_cruda, flecha_conf, memoria_dir,
                marca_conf, marca_hud, error, v, w, estado):
        h, wi = bgr.shape[:2]
        ov = bgr.copy()
        ov[m_lin]  = (255, 0, 0)
        ov[m_rojo] = (0, 0, 255)
        out = cv2.addWeighted(ov, 0.35, bgr, 0.65, 0)
        cv2.line(out, (0, h - self.FRANJA_ERROR), (wi, h - self.FRANJA_ERROR),
                 (180, 180, 180), 1)

        for e in extremos:
            es_eleg = (salida_elegida is not None
                       and e['punto'] == salida_elegida['punto'])
            if e['es_entrada']:
                col, etq, radio, grosor = (0, 255, 0), 'ENT', 8, 2
            elif es_eleg:
                col, etq, radio, grosor = (255, 255, 0), 'ELEG', 12, 3
            else:
                col, etq, radio, grosor = (0, 255, 255), 'S', 8, 2
            pd = e['punto']
            if   e['lado'] == 'abajo':     pd = (pd[0], h - 10)
            elif e['lado'] == 'arriba':    pd = (pd[0], 10)
            elif e['lado'] == 'izquierda': pd = (10, pd[1])
            elif e['lado'] == 'derecha':   pd = (wi - 10, pd[1])
            cv2.circle(out, pd, radio, col, grosor)
            tx, ty = pd
            if   e['lado'] == 'arriba':    ty += 22
            elif e['lado'] == 'abajo':     ty -= 18
            elif e['lado'] == 'izquierda': tx += 18
            else:                          tx -= 64
            self._txt(out, etq, (tx, ty), col)

        if flecha_cruda is not None:
            col_f = (0, 255, 255) if flecha_conf else (0, 165, 255)
            cv2.drawContours(out, [flecha_cruda['contorno']], -1, col_f, 1)
            cf = tuple(map(int, flecha_cruda['centro']))
            pf = tuple(map(int, flecha_cruda['punta']))
            cv2.arrowedLine(out, cf, pf, col_f, 2, tipLength=0.30)

        if marca_conf is not None:
            clase, conf, (mx, my, mw, mh) = marca_conf
            cv2.rectangle(out, (mx, my), (mx + mw, my + mh), (255, 0, 200), 2)
            self._txt(out, '%s %.0f%%' % (clase, conf * 100),
                      (mx, max(12, my - 4)), (255, 0, 200), scale=0.45)

        dir_s = ('--' if memoria_dir is None
                 else '%s (TTL %d/%d)' % (memoria_dir['direccion'],
                                           memoria_dir['ttl_left'],
                                           memoria_dir['ttl_max']))
        if marca_conf is not None:
            marca_s = '%s %.0f%%' % (marca_conf[0], marca_conf[1] * 100)
        elif marca_hud is not None:
            marca_s = '%s (mem)' % marca_hud[0]
        else:
            marca_s = '--'
        for i, t in enumerate((
                'Estado: ' + estado,
                'Memoria dir: ' + dir_s,
                'SALIDA ELEG: ' + (salida_elegida['lado'] if salida_elegida else '--'),
                'MARCA      : ' + marca_s,
                'Error: ' + ('%+.2f' % error if error is not None else '--'),
                'v=%+.2f w=%+.2f' % (v, w),
        )):
            self._txt(out, t, (4, 14 + i * 13), (255, 255, 255))
        return out

    @staticmethod
    def _txt(img, txt, pos, color, scale=0.4):
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 1, cv2.LINE_AA)

    def _mostrar(self, img, title='BrainFollowLine'):
        if not self._gui_ok:
            return
        try:
            cv2.imshow(title, img)
            cv2.waitKey(1)
        except Exception:
            print('AVISO: GUI no disponible. Desactivo display.')
            self._gui_ok = False

    # ----- Step principal -----
    def step(self):
        ok, bgr = self.capture.read()
        if not ok or bgr is None:
            self.move(0.0, 0.0)
            return

        h_img, w_img = bgr.shape[:2]
        cx_img = w_img / 2.0
        self._frame_idx += 1

        # --- Percepcion ---
        m_lin, m_rojo = segmentar_qda(self.clf_qda, bgr,
                                       area_lin_min=self.AREA_MIN_LINEA,
                                       area_mar_min=self.AREA_MIN_ROJO)
        min_front, min_left = self.leer_distancias()
        if self.post_avoid_grace:
            self.post_avoid_grace -= 1

        extremos = detectar_extremos(m_lin, self.BANDA_BORDE,
                                      self.MIN_SEGMENTO, self.FUSION_GAP)
        salidas = [e for e in extremos if not e['es_entrada']]
        error = self.calcular_error_linea(m_lin, cx_img)

        flecha_cruda = detectar_flecha(m_rojo, area_min=self.AREA_MIN_FLECHA)
        flecha_conf  = self.confirmador_flecha.update(flecha_cruda)
        memoria_dir  = self.memoria_direccion.update(flecha_conf, self._frame_idx)

        marca_cruda = None
        if flecha_cruda is None:
            marca_cruda = predecir_marca(m_rojo, self.modelo_marcas,
                                          area_min=self.AREA_MIN_MARCA,
                                          umbral_conf=self.MARCA_UMBRAL_CONF,
                                          maha_max=self.MARCA_MAHA_MAX,
                                          h_img=h_img,
                                          y_top_max_frac=self.MARCA_Y_TOP_MAX_FRAC)
        marca_conf_val = self.confirmador_marca.update(marca_cruda,
                                                       self._frame_idx)
        if flecha_cruda is not None:
            self.confirmador_marca.cancelar_hud()
        if marca_conf_val is not None:
            self.marcas_vistas.append((self._frame_idx, marca_conf_val[0],
                                       marca_conf_val[1]))
        marca_hud = None
        if marca_conf_val is None and self.confirmador_marca.hud_ttl > 0:
            marca_hud = (self.confirmador_marca.hud_clase,
                         self.confirmador_marca.hud_conf)

        salida_elegida  = elegir_salida_por_direccion(extremos, memoria_dir)
        cruce_inminente = self.hay_cruce_inminente(m_lin, salidas, h_img)

        # --- Control de motores ---
        v_cmd, w_cmd, estado = 0.0, 0.0, ''

        if min_front < self.DIST_FRONTAL_OBST and not self.avoiding:
            self.avoiding   = True
            self.avoid_ticks = 0
            self.prev_error  = None

        if self.avoiding:
            self.avoid_ticks += 1
            linea_centrada = (error is not None
                              and abs(error) < self.AVOID_EXIT_ERR_MAX)
            timeout = self.avoid_ticks > self.AVOID_TICKS_MAX
            if (self.avoid_ticks > self.AVOID_TICKS_MIN
                    and min_front > self.DIST_FRENTE_LIBRE
                    and (linea_centrada or timeout)):
                self.avoiding   = False
                self.last_error = self.AVOID_FLAG
                self.prev_error = None
                self.post_avoid_grace = self.POST_AVOID_GRACE
                self.move(0.0, 0.0)
                return
            if min_front < self.DIST_FRONTAL_OBST:
                v_cmd, w_cmd, estado = 0.0, -1.0, 'AVOID-FRONT'
            elif min_left > 0.5:
                v_cmd, w_cmd, estado = 0.0, 1.0, 'AVOID-CORNER'
            else:
                w_cmd = _clamp(-2.5 * (self.DIST_OBJETIVO_PARED - min_left))
                v_cmd, estado = 0.15, 'AVOID-WALL'

        elif cruce_inminente and salida_elegida is not None:
            # Giro al lado que indico la flecha
            err_x = (salida_elegida['punto'][0] - cx_img) / cx_img
            w_cmd = _clamp(-self.CRUCE_KP * err_x)
            v_cmd = max(self.SLOW_FORWARD,
                        self.FULL_FORWARD * (1.0 - 0.9 * abs(w_cmd)))
            self.last_error = err_x
            self.prev_error = None
            estado = 'CRUCE -> ' + salida_elegida['lado']

        elif error is None:
            self.prev_error = None
            w_cmd = -0.8 if self.last_error > 0 else 0.8
            v_cmd, estado = 0.0, 'BUSCAR (spin)'

        else:
            if abs(error) > 0.15:
                self.last_error = error
            d_err = 0.0 if self.prev_error is None else (error - self.prev_error)
            self.prev_error = error
            w_cmd = _clamp(-(self.KP * error + 0.6 * d_err))
            v_cmd = max(self.SLOW_FORWARD,
                        self.FULL_FORWARD * (1.0 - 0.8 * abs(w_cmd)))
            estado = ('FOLLOW (grace=%d)' % self.post_avoid_grace
                      if self.post_avoid_grace else 'FOLLOW')

        self.move(v_cmd, w_cmd)

        if self._frame_idx % self.PRINT_EVERY_N_FRAMES == 0:
            print('[%5d] %-18s err=%s v=%+.2f w=%+.2f sonar_f=%.2f'
                  % (self._frame_idx, estado,
                     ('%+.2f' % error) if error is not None else '  --',
                     v_cmd, w_cmd, min_front))

        out = self.overlay(bgr, m_lin, m_rojo, extremos, salida_elegida,
                            flecha_cruda, flecha_conf, memoria_dir,
                            marca_conf_val, marca_hud,
                            error, v_cmd, w_cmd, estado)
        self._mostrar(out)


def INIT(engine):
    assert (engine.robot.requires('range-sensor')
            and engine.robot.requires('continuous-movement'))
    return BrainFollowLine('BrainFollowLine', engine)
