from pyrobot.brain import Brain

import os
import re
import glob
import math

import cv2
import numpy as np

from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)


# ======================================================================
# Rutas de los datos (relativas al directorio del brain)
# ======================================================================
_AQUI = os.path.dirname(os.path.abspath(__file__))
RUTA_IMG_ORIGINAL   = os.path.join(_AQUI, 'imagen_original.png')
RUTA_IMG_MARCADA    = os.path.join(_AQUI, 'imagen_marcada.png')
RUTA_DATASET_MARCAS = os.path.join(_AQUI, 'marcas-capturasStage')

CLASES_MARCAS = ('man', 'stairs', 'telephone', 'woman')


# ======================================================================
# Segmentacion QDA (linea, marca, fondo) — entrenado una vez en setup()
# ======================================================================
def _features_pixel(rgb):
    """9 features por pixel: HSV norm + ab(Lab) + RGB norm + L."""
    if rgb.ndim == 3:
        rgb = rgb.reshape(-1, 3)
    img = rgb.reshape(1, -1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    hsv[:, 0] /= 179.0
    hsv[:, 1] /= 255.0
    hsv[:, 2] /= 255.0
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    L = lab[:, 0:1] / 255.0
    a = (lab[:, 1:2] - 128.0) / 128.0
    b = (lab[:, 2:3] - 128.0) / 128.0
    rgb_f = rgb.astype(np.float32)
    s = rgb_f.sum(axis=1, keepdims=True).clip(min=1.0)
    rgb_n = rgb_f / s
    return np.hstack([hsv, a, b, rgb_n, L]).astype(np.float32)


def entrenar_qda_linea():
    """Carga imagen_original + imagen_marcada y entrena un QDA 3-clases."""
    orig = cv2.imread(RUTA_IMG_ORIGINAL)
    marc = cv2.imread(RUTA_IMG_MARCADA)
    if orig is None or marc is None:
        raise FileNotFoundError(
            'No se encuentran imagen_original.png / imagen_marcada.png en ' + _AQUI)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    marc_rgb = cv2.cvtColor(marc, cv2.COLOR_BGR2RGB)
    # Paleta:  rojo=marca  verde=fondo  azul=linea
    m_marca = (marc_rgb[:, :, 0] == 255) & (marc_rgb[:, :, 1] == 0) & (marc_rgb[:, :, 2] == 0)
    m_fondo = (marc_rgb[:, :, 0] == 0) & (marc_rgb[:, :, 1] == 255) & (marc_rgb[:, :, 2] == 0)
    m_linea = (marc_rgb[:, :, 0] == 0) & (marc_rgb[:, :, 1] == 0) & (marc_rgb[:, :, 2] == 255)
    X = np.vstack([
        _features_pixel(orig_rgb[m_fondo]),
        _features_pixel(orig_rgb[m_marca]),
        _features_pixel(orig_rgb[m_linea]),
    ])
    y = np.hstack([
        np.zeros(int(m_fondo.sum()), dtype=int),
        np.ones(int(m_marca.sum()), dtype=int),
        np.full(int(m_linea.sum()), 2, dtype=int),
    ])
    return QuadraticDiscriminantAnalysis(reg_param=0.01).fit(X, y)


def segmentar_qda(clf, bgr):
    """Devuelve (mask_linea, mask_marca) como bool arrays (H, W)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    feats = _features_pixel(rgb)
    pred = clf.predict(feats).reshape(h, w)
    m_lin = (pred == 2).astype(np.uint8) * 255
    m_mar = (pred == 1).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_OPEN, k, iterations=1)
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_CLOSE, k, iterations=3)
    m_mar = cv2.morphologyEx(m_mar, cv2.MORPH_OPEN, k, iterations=1)
    m_mar = cv2.morphologyEx(m_mar, cv2.MORPH_CLOSE, k, iterations=2)
    return m_lin > 0, m_mar > 0


# ======================================================================
# Clasificador LDA de marcas (man / stairs / telephone / woman)
# ======================================================================
_RX_FNAME = re.compile(r'^([a-z]+)[-_]\d+\.png$', re.IGNORECASE)


def _log_hu(sil):
    M = cv2.moments(sil, binaryImage=True)
    hu = cv2.HuMoments(M).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


def _ratios_forma(sil):
    cnts, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros(4, dtype=np.float32)
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, closed=True)
    if perim <= 1 or area <= 1:
        return np.zeros(4, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = h / w if w else 0.0
    extent = area / (w * h) if w and h else 0.0
    hull = cv2.convexHull(cnt)
    ha = cv2.contourArea(hull)
    solidity = area / ha if ha else 0.0
    circ = 4 * math.pi * area / (perim * perim)
    return np.array([aspect, extent, solidity, circ], dtype=np.float32)


def _descriptor_marca(sil):
    return np.concatenate([_log_hu(sil), _ratios_forma(sil)]).astype(np.float32)


def _silueta_de_bgr(bgr, area_min=80):
    """Extrae silueta de marca roja (HSV interno solo para recortar bbox)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (cv2.inRange(hsv, np.array([0, 100, 70]),  np.array([12, 255, 255])) |
         cv2.inRange(hsv, np.array([165, 100, 70]), np.array([179, 255, 255])))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    if not m.any():
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    areas = st[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    if st[idx, cv2.CC_STAT_AREA] < area_min:
        return None
    x = int(st[idx, cv2.CC_STAT_LEFT])
    y = int(st[idx, cv2.CC_STAT_TOP])
    w = int(st[idx, cv2.CC_STAT_WIDTH])
    h = int(st[idx, cv2.CC_STAT_HEIGHT])
    sil = (lab[y:y+h, x:x+w] == idx).astype(np.uint8) * 255
    return sil, (x, y, w, h)


def entrenar_lda_marcas():
    files = sorted(glob.glob(os.path.join(RUTA_DATASET_MARCAS, '*.png')))
    if not files:
        raise FileNotFoundError(
            'Carpeta marcas-capturasStage vacia en ' + RUTA_DATASET_MARCAS)
    X, y, rangos_raw = [], [], {c: [] for c in range(len(CLASES_MARCAS))}
    for f in files:
        m = _RX_FNAME.match(os.path.basename(f))
        if not m:
            continue
        clase = m.group(1).lower()
        if clase not in CLASES_MARCAS:
            continue
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        out = _silueta_de_bgr(bgr)
        if out is None:
            continue
        sil, _ = out
        d = _descriptor_marca(sil)
        X.append(d)
        idx = CLASES_MARCAS.index(clase)
        y.append(idx)
        rangos_raw[idx].append(d[7:11])
    X = np.stack(X)
    y = np.array(y)
    clf = LinearDiscriminantAnalysis(solver='svd').fit(X, y)
    margen = 0.05
    rangos = {}
    for k, lst in rangos_raw.items():
        if lst:
            arr = np.stack(lst)
            rangos[k] = np.column_stack([arr.min(axis=0) - margen,
                                         arr.max(axis=0) + margen])
    return clf, rangos


def predecir_marca(bgr, clf, rangos, area_min=300, umbral_conf=0.55):
    out = _silueta_de_bgr(bgr, area_min=area_min)
    if out is None:
        return None
    sil, bbox = out
    feat = _descriptor_marca(sil).reshape(1, -1)
    probs = clf.predict_proba(feat)[0] if hasattr(clf, 'predict_proba') else None
    pred = int(clf.predict(feat)[0])
    conf = float(probs[pred]) if probs is not None else 1.0
    if conf < umbral_conf:
        return None
    rng = rangos.get(pred) if rangos else None
    if rng is not None:
        ratios = feat[0, 7:11]
        if np.any(ratios < rng[:, 0]) or np.any(ratios > rng[:, 1]):
            return None
    return CLASES_MARCAS[pred], conf, bbox


# ======================================================================
# Brain
# ======================================================================
class BrainFollowLine(Brain):
    """Integracion Practica 1 (control) + Practica 2 (percepcion).

    - Linea azul: segmentacion QDA (entrenada de imagen_original + imagen_marcada)
    - Marcas: clasificador LDA (entrenado con marcas-capturasStage/)
    - Flecha: PCA sobre blob rojo alargado; el ramal con angulo mas parecido
      al de la flecha se elige como salida.
    - Esquiva: lectura sonar frontal/izquierdo, maquina de estados
    - Overlay: linea/marca, entradas (verdes), salidas (amarillas),
      salida elegida (cyan), flecha (vector amarillo), bbox de marca, error
    """

    # ---- Velocidades ------------------------------------------------
    NO_FORWARD   = 0.0
    SLOW_FORWARD = 0.05   # piso de avance al girar (antes 0.10)
    MED_FORWARD  = 0.20
    FULL_FORWARD = 0.40   # techo en recta (antes 0.55)

    NO_TURN    = 0.0
    SOFT_LEFT  = 0.30
    MED_LEFT   = 0.60
    HARD_LEFT  = 1.00
    HARD_RIGHT = -1.00

    # ---- Control PD -------------------------------------------------
    KP      = 1.5    # ganancia proporcional
    KD      = 0.6    # mas damping para evitar rebotes en curvas cerradas
    # Pendientes de frenado: con omega=1 el factor cae al 3%, con |error|=1
    # cae al 10%. Tomamos el MINIMO -> el frenado mas agresivo manda. Asi
    # incluso si el omega no esta saturado todavia, un error grande (linea
    # casi fuera del campo) ya frena el robot.
    ALPHA_V_W = 0.97   # peso de |omega| en el frenado
    ALPHA_V_E = 0.90   # peso de |error| en el frenado

    # ---- Cruces -----------------------------------------------------
    BANDA_BORDE    = 4    # ancho de la banda perimetral (px)
    MIN_SEGMENTO   = 5    # longitud minima de un segmento en el borde
    FUSION_GAP     = 8    # gap maximo para fusionar segmentos
    FRANJA_ERROR   = 40   # franja inferior para medir error
    AREA_MIN_LINEA = 120

    # ---- Flechas y marcas ------------------------------------------
    AREA_MIN_FLECHA   = 120
    FLECHA_CIRC_MAX   = 0.45   # circularidad <= -> puede ser flecha
    FLECHA_ELONG_MIN  = 2.5    # sqrt(eigval_max/eigval_min) >= -> es flecha
    MARCA_AREA_MIN    = 300
    MARCA_COOLDOWN    = 25

    # ---- Obstaculo --------------------------------------------------
    DIST_FRONTAL_OBST   = 0.45
    DIST_FRENTE_LIBRE   = 0.40
    DIST_OBJETIVO_PARED = 0.30
    AVOID_TICKS_MIN     = 40

    # ---- Cache flecha ----------------------------------------------
    ARROW_TTL  = 60     # frames que una flecha vista sigue valida (6 s @ 10fps)
    CRUCE_KP   = 1.6    # ganancia omega = -KP * (x_extremo - W/2)/(W/2)

    # ---- Cruce inminente vs cruce visible --------------------------
    # cruce_visible: hay >=2 salidas en algun borde (puede estar lejos)
    # cruce_imminente: la BIFURCACION esta en la mitad-baja de la imagen
    #                  (la linea ya se separa cerca del robot)
    # Hasta que sea imminente, el robot sigue la linea tronco con PD.
    JUNCTION_Y_FRAC  = 0.55   # umbral: junc_y >= H * frac -> imminente
    JUNCTION_MIN_RUN = 4      # longitud minima de un "run" en cada fila

    # ==================================================================
    def setup(self):
        print('Entrenando QDA de segmentacion...')
        self.clf_qda = entrenar_qda_linea()
        print('Entrenando LDA de marcas...')
        self.clf_lda, self.rangos_marcas = entrenar_lda_marcas()
        print('Listo. Entrenamiento completo.')

        self.prev_error      = None
        self.last_error      = 0.0
        self.avoiding        = False
        self.avoid_ticks     = 0
        self.arrow_cache     = None     # dict con angulo, centro, contorno, punta
        self.arrow_ttl_left  = 0
        self.last_marca      = None
        self.cooldown_marca  = 0
        self.step_count      = 0
        self.marcas_vistas   = []

    def destroy(self):
        if self.marcas_vistas:
            print('=== Marcas detectadas durante la ejecucion ===')
            for i, info in enumerate(self.marcas_vistas, 1):
                print('  %d. %s (conf %.2f)  imagen=(%d,%d)  robot=%s'
                      % (i, info[0], info[1], info[2], info[3], info[4]))
        cv2.destroyAllWindows()

    # ==================================================================
    # Linea: error y deteccion de extremos (entradas/salidas)
    # ==================================================================
    def error_seguimiento(self, m_lin):
        h, w = m_lin.shape
        franja = m_lin[-self.FRANJA_ERROR:, :]
        if not franja.any():
            return None
        _, xs = np.where(franja)
        cx = xs.mean()
        return float((cx - w / 2.0) / (w / 2.0))

    def _segmentos(self, perfil):
        p = perfil.astype(np.int8)
        d = np.diff(np.concatenate([[0], p, [0]]))
        ini = np.where(d == 1)[0]
        fin = np.where(d == -1)[0]
        return [(int(a), int(b)) for a, b in zip(ini, fin)
                if (b - a) >= self.MIN_SEGMENTO]

    def _fusionar(self, segs):
        if not segs:
            return segs
        out = [segs[0]]
        for s, e in segs[1:]:
            ps, pe = out[-1]
            if s - pe <= self.FUSION_GAP:
                out[-1] = (ps, e)
            else:
                out.append((s, e))
        return out

    def junction_y(self, m_lin):
        """Devuelve la fila mas baja donde la linea tiene >=2 segmentos.

        Recorre las filas de la imagen de abajo a arriba; la primera fila
        donde encuentra al menos 2 "runs" de longitud minima es el punto
        de bifurcacion. Devuelve None si la linea no se bifurca dentro
        del campo. Y grande = bifurcacion CERCA (cerca del robot);
        y pequeno = bifurcacion LEJOS (arriba del campo).
        """
        h, w = m_lin.shape
        for y in range(h - 1, -1, -1):
            row = m_lin[y].astype(np.uint8)
            d = np.diff(np.concatenate([[0], row, [0]]))
            ini = np.where(d == 1)[0]
            fin = np.where(d == -1)[0]
            n_long = sum(1 for s, e in zip(ini, fin)
                         if (e - s) >= self.JUNCTION_MIN_RUN)
            if n_long >= 2:
                return int(y)
        return None

    def detectar_extremos(self, m_lin):
        """Lista de dicts con lado, punto (x,y), longitud, es_entrada, angulo."""
        h, w = m_lin.shape
        b = self.BANDA_BORDE
        cx, cy = w / 2.0, h / 2.0
        bordes = {
            'abajo'    : (m_lin[-b:, :].any(axis=0), 'x'),
            'arriba'   : (m_lin[:b, :].any(axis=0),  'x'),
            'izquierda': (m_lin[:, :b].any(axis=1),  'y'),
            'derecha'  : (m_lin[:, -b:].any(axis=1), 'y'),
        }
        out = []
        for lado, (perfil, eje) in bordes.items():
            for s, e in self._fusionar(self._segmentos(perfil)):
                pos = (s + e) // 2
                if eje == 'x':
                    px, py = pos, (h - 1 if lado == 'abajo' else 0)
                else:
                    px, py = (0 if lado == 'izquierda' else w - 1), pos
                ang = math.degrees(math.atan2(cy - py, px - cx))
                out.append({
                    'lado': lado, 'punto': (int(px), int(py)),
                    'longitud': e - s, 'es_entrada': (lado == 'abajo'),
                    'angulo': ang,
                })
        return out

    # ==================================================================
    # Flechas
    # ==================================================================
    def detectar_flecha(self, m_rojo):
        """Devuelve dict con angulo, centro, contorno, punta. None si no la hay."""
        if not m_rojo.any():
            return None
        m_u8 = m_rojo.astype(np.uint8)
        cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < self.AREA_MIN_FLECHA or len(cnt) < 5:
            return None
        perim = cv2.arcLength(cnt, closed=True)
        circ = 4.0 * math.pi * area / max(perim * perim, 1e-6)
        if circ > self.FLECHA_CIRC_MAX:
            return None

        # PCA sobre la mascara del contorno mayor
        comp_mask = np.zeros_like(m_u8)
        cv2.drawContours(comp_mask, [cnt], -1, 255, thickness=-1)
        ys, xs = np.where(comp_mask > 0)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        cov = np.cov(pts.T)
        eigval, eigvec = np.linalg.eigh(cov)
        # Filtro de elongacion: una flecha tiene un eje muy dominante
        ev_max = float(max(eigval))
        ev_min = float(max(min(eigval), 1e-6))
        elong = math.sqrt(ev_max / ev_min)
        if elong < self.FLECHA_ELONG_MIN:
            return None
        eje  = eigvec[:, int(np.argmax(eigval))]
        perp = np.array([-eje[1], eje[0]])
        proy_a = (pts[:, 0] - cx) * eje[0]  + (pts[:, 1] - cy) * eje[1]
        proy_p = (pts[:, 0] - cx) * perp[0] + (pts[:, 1] - cy) * perp[1]
        pmin, pmax = float(proy_a.min()), float(proy_a.max())
        longitud = pmax - pmin
        if longitud < 5:
            return None

        # Determinar la "cabeza" combinando dos pistas:
        #   (a) lado con mayor area pasada el centroide
        #       (la cabeza triangular pesa mas que la cola)
        #   (b) lado con mayor span perpendicular en la franja extrema
        #       (el triangulo se ensancha cerca de la base)
        n_pos = int(np.sum(proy_a > 0))
        n_neg = int(np.sum(proy_a < 0))
        sentido_a = +1 if n_pos > n_neg else -1
        franja_a = 0.25 * longitud
        sel_pos = proy_a > (pmax - franja_a)
        sel_neg = proy_a < (pmin + franja_a)

        def _span(p):
            return float(p.max() - p.min()) if len(p) >= 5 else 0.0

        s_pos = _span(proy_p[sel_pos])
        s_neg = _span(proy_p[sel_neg])
        sentido_b = +1 if s_pos > s_neg else -1

        # Requerimos asimetria clara: la cabeza debe ser >= 30% mas ancha
        # que la cola en los extremos. Asi rechazamos siluetas simetricas
        # (man, stairs) que no son flechas.
        max_s = max(s_pos, s_neg)
        if max_s > 1e-3:
            asim = abs(s_pos - s_neg) / max_s
            if asim < 0.30:
                return None

        # La asimetria de span es la senal mas fiable (n_pos/n_neg suelen
        # estar empatados porque la centroide divide la masa por la mitad).
        sentido = sentido_b
        idx = int(np.argmax(proy_a)) if sentido > 0 else int(np.argmin(proy_a))
        px, py = float(pts[idx, 0]), float(pts[idx, 1])
        # Angulo en convencion imagen (eje Y invertido):
        #   0=derecha  90=arriba  +-180=izquierda  -90=abajo
        ang = math.degrees(math.atan2(cy - py, px - cx))
        return {'angulo': ang, 'centro': (cx, cy),
                'contorno': cnt, 'punta': (px, py), 'area': float(area)}

    def elegir_salida(self, extremos, flecha):
        """Elige el extremo (no entrada) cuyo angulo cuadra con la flecha."""
        salidas = [e for e in extremos if not e['es_entrada']]
        if not salidas:
            return None
        if flecha is None:
            # Sin flecha: preferir el extremo superior centrado
            return min(salidas,
                       key=lambda e: abs(e['angulo'] - 90))
        ang_fl = flecha['angulo']

        def dif(a, b):
            d = (a - b + 180) % 360 - 180
            return abs(d)

        return min(salidas, key=lambda e: dif(e['angulo'], ang_fl))

    def omega_para_salida(self, salida, w_img):
        """Omega proporcional al desplazamiento horizontal del extremo
        elegido. Funciona igual para 'arriba', 'izquierda' y 'derecha'.

            extremo izquierdo (px=0)       -> err=-1 -> omega=+KP -> giro izquierda
            extremo derecho   (px=W-1)     -> err=+1 -> omega=-KP -> giro derecha
            extremo arriba centrado        -> err=0  -> omega=0   -> recto
        """
        if salida is None:
            return 0.0
        px = salida['punto'][0]
        err = (px - w_img / 2.0) / (w_img / 2.0)
        return max(-1.0, min(1.0, -self.CRUCE_KP * err))

    # ==================================================================
    # Control PD
    # ==================================================================
    def velocidad_para_giro(self, omega, error_abs=0.0):
        """Velocidad limitada por dos canales (gana el mas restrictivo).

        - Canal omega: cuando el robot gira fuerte, frena.
        - Canal error: cuando la linea esta lejos del centro (curva muy
          cerrada / linea a punto de salirse del campo), frena tambien
          aunque omega todavia no haya saturado.

        El piso es SLOW_FORWARD (0.05) -> el robot sigue avanzando un
        minimo para mantener la linea en el campo de vision en vez de
        clavarse y perderla.
        """
        factor_w = max(0.0, 1.0 - self.ALPHA_V_W * abs(omega))
        factor_e = max(0.0, 1.0 - self.ALPHA_V_E * abs(error_abs))
        factor = min(factor_w, factor_e)
        return max(self.SLOW_FORWARD, self.FULL_FORWARD * factor)

    def control_pd(self, error):
        d = 0.0 if self.prev_error is None else (error - self.prev_error)
        self.prev_error = error
        omega = -(self.KP * error + self.KD * d)
        omega = max(-1.0, min(1.0, omega))
        v = self.velocidad_para_giro(omega, abs(error))
        return v, omega

    # ==================================================================
    # Pose del robot
    # ==================================================================
    def _pose_robot(self):
        try:
            return '(x=%.2f, y=%.2f, th=%.1f)' % (
                float(self.robot.x), float(self.robot.y), float(self.robot.th))
        except Exception:
            return '(pose no disponible)'

    # ==================================================================
    # Overlay
    # ==================================================================
    def dibujar_overlay(self, bgr, m_lin, m_rojo, extremos, salida_elegida,
                        flecha, marca_info, error, v, omega, estado,
                        junc_y=None):
        h, w = bgr.shape[:2]
        out = bgr.copy()

        # Mascaras translucidas
        overlay = out.copy()
        overlay[m_lin]  = (255, 0, 0)   # azul intenso
        overlay[m_rojo] = (0, 0, 255)   # rojo intenso
        out = cv2.addWeighted(overlay, 0.30, out, 0.70, 0)

        # Franja inferior + error
        cv2.rectangle(out, (0, h - self.FRANJA_ERROR),
                      (w - 1, h - 1), (180, 180, 180), 1)
        cv2.line(out, (w // 2, h - self.FRANJA_ERROR),
                 (w // 2, h - 1), (180, 180, 180), 1)
        if error is not None:
            x_err = int(w / 2 + error * (w / 2))
            cv2.line(out, (x_err, h - self.FRANJA_ERROR),
                     (x_err, h - 1), (0, 255, 255), 2)

        # Entradas (verde) / Salidas (amarillo) / Elegida (cyan)
        for e in extremos:
            es_eleg = (salida_elegida is not None and
                       e['lado'] == salida_elegida['lado'] and
                       e['punto'] == salida_elegida['punto'])
            if e['es_entrada']:
                col, etq, gr = (0, 255, 0), 'ENT', 2
            elif es_eleg:
                col, etq, gr = (255, 255, 0), 'ELEG', 3
            else:
                col, etq, gr = (0, 255, 255), 'S', 2
            cv2.circle(out, e['punto'], 8, col, gr)
            tx, ty = e['punto']
            if   e['lado'] == 'arriba':    ty += 16
            elif e['lado'] == 'abajo':     ty -= 6
            elif e['lado'] == 'izquierda': tx += 12
            else:                          tx -= 36
            cv2.putText(out, etq, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, etq, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, col, 1, cv2.LINE_AA)

        # Flecha (contorno + vector centroide -> punta)
        if flecha is not None:
            cv2.drawContours(out, [flecha['contorno']], -1, (255, 255, 0), 1)
            cxf, cyf = map(int, flecha['centro'])
            pxf, pyf = map(int, flecha['punta'])
            cv2.circle(out, (cxf, cyf), 3, (255, 255, 0), -1)
            cv2.circle(out, (pxf, pyf), 4, (0, 255, 255), -1)
            cv2.arrowedLine(out, (cxf, cyf), (pxf, pyf),
                            (0, 255, 255), 2, tipLength=0.30)

        # Marca: bbox + clase
        if marca_info is not None:
            clase, conf, bbox = marca_info
            x0, y0b, ww, hh = bbox
            cv2.rectangle(out, (x0, y0b), (x0 + ww, y0b + hh),
                          (0, 200, 255), 2)
            txt = '%s (%.2f)' % (clase, conf)
            cv2.putText(out, txt, (x0, max(12, y0b - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, txt, (x0, max(12, y0b - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 200, 255), 1, cv2.LINE_AA)

        # Panel de texto
        n_ent = sum(1 for e in extremos if e['es_entrada'])
        n_sal = sum(1 for e in extremos if not e['es_entrada'])
        err_txt   = ('%+.2f' % error) if error is not None else '  --'
        fle_txt   = ('%+5.0f deg' % flecha['angulo']) if flecha is not None else '--'
        eleg_txt  = salida_elegida['lado'] if salida_elegida is not None else '--'
        marca_txt = ('%s (%.2f)' % (marca_info[0], marca_info[1])
                     if marca_info is not None else '--')
        # Indicador de proximidad de la bifurcacion: '----' lejos, '====' cerca
        if junc_y is not None:
            frac = junc_y / float(h)
            n_bar = max(1, int(frac * 8))
            jy_txt = '%-8s y=%d' % ('=' * n_bar + '-' * (8 - n_bar), junc_y)
        else:
            jy_txt = '--'
        lineas = [
            'Estado : %s'    % estado,
            'Entr=%d  Sal=%d' % (n_ent, n_sal),
            'Eleg   : %s'    % eleg_txt,
            'Flecha : %s'    % fle_txt,
            'JuncY  : %s'    % jy_txt,
            'Marca  : %s'    % marca_txt,
            'Error  : %s'    % err_txt,
            'v=%+.2f w=%+.2f' % (v, omega),
        ]

        # Linea horizontal del punto de bifurcacion
        if junc_y is not None:
            color = (0, 200, 255) if junc_y >= h * self.JUNCTION_Y_FRAC \
                                  else (120, 120, 120)
            cv2.line(out, (0, junc_y), (w - 1, junc_y), color, 1)
        # Marca el umbral de "cruce imminente"
        y_thr = int(h * self.JUNCTION_Y_FRAC)
        cv2.line(out, (0, y_thr), (w - 1, y_thr), (60, 60, 60), 1)
        y0 = 14
        for i, line in enumerate(lineas):
            yy = y0 + i * 13
            cv2.putText(out, line, (4, yy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, line, (4, yy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    # ==================================================================
    # Bucle principal
    # ==================================================================
    def step(self):
        self.step_count += 1
        bgr = self.robot.getImage()
        if bgr is None:
            self.move(self.NO_FORWARD, self.NO_TURN)
            return

        m_lin, m_rojo = segmentar_qda(self.clf_qda, bgr)

        # Sonar
        try:
            front = [self.robot.range[i].distance() for i in range(2, 6)]
            left  = [self.robot.range[i].distance() for i in range(0, 3)]
        except Exception:
            front, left = [99.0], [99.0]
        min_front = min(front)
        min_left  = min(left)

        self.cooldown_marca = max(0, self.cooldown_marca - 1)

        # Extremos siempre, para visualizar
        extremos = self.detectar_extremos(m_lin)
        salidas = [e for e in extremos if not e['es_entrada']]
        # "cruce_visible": hay topologia de cruce en algun borde
        cruce_visible = len(salidas) >= 2
        # "cruce_imminente": ademas el punto de bifurcacion esta en la
        # mitad inferior de la imagen -> el robot esta llegando al cruce
        h_img, w_img = m_lin.shape
        junc_y = self.junction_y(m_lin) if cruce_visible else None
        en_cruce = (cruce_visible and junc_y is not None
                    and junc_y >= h_img * self.JUNCTION_Y_FRAC)

        # Primero: probar si el blob rojo es una marca conocida.
        # Si lo es, no buscamos flecha (la silueta de la marca podria pasar
        # los filtros de flecha y contaminar la cache).
        marca_actual = None
        if m_rojo.any():
            marca_actual = predecir_marca(bgr, self.clf_lda, self.rangos_marcas,
                                          area_min=self.MARCA_AREA_MIN)

        # Reportar marca solo si es nueva y fuera de cooldown
        marca_info = marca_actual
        if (marca_actual is not None and self.cooldown_marca == 0
                and marca_actual[0] != self.last_marca):
            clase, conf, bbox = marca_actual
            cx_m, cy_m = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
            pose = self._pose_robot()
            print('>>> MARCA: %s (conf %.2f)  img=(%d,%d)  robot=%s'
                  % (clase, conf, cx_m, cy_m, pose))
            self.marcas_vistas.append((clase, conf, cx_m, cy_m, pose))
            self.last_marca = clase
            self.cooldown_marca = self.MARCA_COOLDOWN

        # Solo buscamos flecha si NO hay marca clasificada. Cacheamos.
        flecha = None
        if marca_actual is None:
            flecha = self.detectar_flecha(m_rojo)
        if flecha is not None:
            self.arrow_cache = flecha
            self.arrow_ttl_left = self.ARROW_TTL
        elif self.arrow_ttl_left > 0:
            self.arrow_ttl_left -= 1
            flecha = self.arrow_cache

        error = self.error_seguimiento(m_lin)

        # ---------------- Decision de control ------------------------
        # Cuatro estados puros, SIN ningun "lock" ni omega congelada de
        # frames anteriores. Cada paso reacciona a la imagen actual:
        #   1) AVOID    - obstaculo delante
        #   2) CRUCE    - bifurcacion imminente: usar flecha para elegir
        #                 ramal; omega proporcional al x del extremo elegido
        #   3) FOLLOW   - PD sobre el error de la franja inferior
        #   4) BUSCAR   - linea perdida del todo
        estado = ''
        salida_elegida = None
        v_cmd, w_cmd = self.NO_FORWARD, self.NO_TURN

        # 1) Obstaculo
        if min_front < self.DIST_FRONTAL_OBST and not self.avoiding:
            self.avoiding = True
            self.avoid_ticks = 0
            self.prev_error = None

        if self.avoiding:
            self.avoid_ticks += 1
            found = m_lin.any()
            if (found and min_front > self.DIST_FRENTE_LIBRE
                    and self.avoid_ticks > self.AVOID_TICKS_MIN):
                self.avoiding = False
                self.last_error = 1.0
                v_cmd, w_cmd = self.NO_FORWARD, self.NO_TURN
                estado = 'POST-AVOID'
            elif min_front < self.DIST_FRONTAL_OBST:
                v_cmd, w_cmd = self.NO_FORWARD, self.HARD_RIGHT
                estado = 'AVOID-FRONT'
            elif min_left > 0.5:
                v_cmd, w_cmd = self.NO_FORWARD, self.HARD_LEFT
                estado = 'AVOID-CORNER'
            else:
                err_pared = self.DIST_OBJETIVO_PARED - min_left
                w_cmd = max(-1.0, min(1.0, -2.5 * err_pared))
                v_cmd = self.SLOW_FORWARD
                estado = 'AVOID-WALL'

        # 2) CRUCE imminente: solo cuando la bifurcacion esta cerca del robot
        #    (junc_y en la mitad-baja de la imagen). La flecha (de la cache)
        #    elige el ramal; omega = -KP * (x_extremo - W/2)/(W/2).
        #    Recalculamos CADA FRAME: si el robot rota, x_extremo se mueve
        #    hacia el centro y omega baja suavemente sin overshoot.
        elif en_cruce:
            salida_elegida = self.elegir_salida(extremos, flecha)
            if salida_elegida is not None:
                px = salida_elegida['punto'][0]
                err_x = (px - w_img / 2.0) / (w_img / 2.0)
                w_cmd = max(-1.0, min(1.0, -self.CRUCE_KP * err_x))
                v_cmd = self.velocidad_para_giro(w_cmd, abs(err_x))
                self.last_error = err_x   # recordar sentido por si perdemos la linea despues
                self.prev_error = None    # PD se reinicia al salir de cruce
                estado = ('CRUCE %d sal -> %s (errX=%+.2f, w=%+.2f, v=%.2f)'
                          % (len(salidas), salida_elegida['lado'],
                             err_x, w_cmd, v_cmd))
                if flecha is not None and self.step_count % 5 == 0:
                    print('>>> CRUCE: sal=%d eleg=%s ang_fl=%+.0f '
                          'errX=%+.2f w=%+.2f v=%.2f'
                          % (len(salidas), salida_elegida['lado'],
                             flecha['angulo'], err_x, w_cmd, v_cmd))
            elif error is not None:
                # Cruce visible pero sin flecha NI extremo elegible:
                # caer a PD para no quedarnos parados.
                if abs(error) > 0.10:
                    self.last_error = error
                v_cmd, w_cmd = self.control_pd(error)
                estado = 'CRUCE sin flecha -> PD'
            else:
                v_cmd, w_cmd = self.SLOW_FORWARD, self.NO_TURN
                estado = 'CRUCE sin info'

        # 3) FOLLOW: PD sobre la franja inferior. Cubre rectas, curvas y
        #    el caso "cruce visible pero todavia lejos": seguimos la linea
        #    tronco hasta que la bifurcacion entre en la zona imminente.
        elif error is not None:
            if abs(error) > 0.10:
                self.last_error = error
            v_cmd, w_cmd = self.control_pd(error)
            if cruce_visible:
                estado = ('FOLLOW (cruce arriba, juncY=%s/%d)'
                          % (junc_y if junc_y is not None else '?',
                             int(h_img * self.JUNCTION_Y_FRAC)))
            else:
                estado = 'FOLLOW (err=%+.2f, w=%+.2f, v=%.2f)' % (
                    error, w_cmd, v_cmd)

        # 4) Linea perdida del todo: avanzar muy lento mientras gira suave
        #    hacia donde estaba la linea por ultima vez.
        else:
            self.prev_error = None
            v_cmd = self.SLOW_FORWARD
            w_cmd = -0.6 if self.last_error > 0 else 0.6
            estado = 'BUSCAR LINEA (ult_err=%+.2f)' % self.last_error

        # ---------------- Ejecucion + overlay ------------------------
        self.move(v_cmd, w_cmd)

        out = self.dibujar_overlay(
            bgr, m_lin, m_rojo, extremos, salida_elegida, flecha,
            marca_info, error, v_cmd, w_cmd, estado, junc_y=junc_y)
        cv2.imshow('Stage Camera Image', out)
        cv2.waitKey(1)


def INIT(engine):
    assert (engine.robot.requires('range-sensor') and
            engine.robot.requires('continuous-movement'))
    return BrainFollowLine('BrainFollowLine', engine)
