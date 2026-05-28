"""BrainFollowLine.py - Cerebro final para el robot real (Pioneer + C920).

Pipeline integrado:
  - Segmentacion HSV (linea azul + zonas rojas/magenta).
  - Deteccion de extremos del camino (ENT / S).
  - Deteccion geometrica de FLECHA + memoria con TTL.
  - Clasificacion de MARCAS (man/stairs/telephone/woman) con LDA.
  - Eleccion de la salida del cruce segun la flecha + LOCK.
  - Evasion frontal con sonar (3 estados: FRONT / CORNER / WALL).
  - Seguimiento PD de la franja inferior + arco para reencontrar la linea.
"""

from pyrobot.brain import Brain

import os
import re
import glob
import math

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# =====================================================================
# RUTAS Y CONSTANTES GLOBALES
# =====================================================================
_AQUI = os.path.dirname(os.path.abspath(__file__))
RUTA_DATASET_MARCAS = os.path.join(_AQUI, 'marcas-capturasStage')
CLASES_MARCAS = ('man', 'stairs', 'telephone', 'woman')
_RX_MARCA = re.compile(r'^([a-z]+)[-_]\d+\.png$', re.IGNORECASE)

# Kernels morfologicos (una sola instancia compartida).
_KERNEL_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_KERNEL_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# --- HSV: linea azul del piso real ---
_HSV_AZUL_LO  = np.array([ 90,  60,  40], dtype=np.uint8)
_HSV_AZUL_HI  = np.array([130, 255, 255], dtype=np.uint8)

# --- HSV: rojo + magenta (flechas y marcas en el frame del robot) ---
_HSV_ROJO_LO1 = np.array([  0,  80,  50], dtype=np.uint8)
_HSV_ROJO_HI1 = np.array([ 12, 255, 255], dtype=np.uint8)
_HSV_ROJO_LO2 = np.array([165,  80,  50], dtype=np.uint8)
_HSV_ROJO_HI2 = np.array([179, 255, 255], dtype=np.uint8)
_HSV_MAG_LO   = np.array([130,  50,  40], dtype=np.uint8)
_HSV_MAG_HI   = np.array([165, 255, 255], dtype=np.uint8)

# --- HSV: rojo estricto (solo para el descriptor del LDA de marcas) ---
_HSV_ROJO_MARCA_LO1 = np.array([  0, 100,  70], dtype=np.uint8)
_HSV_ROJO_MARCA_HI1 = np.array([ 12, 255, 255], dtype=np.uint8)
_HSV_ROJO_MARCA_LO2 = np.array([165, 100,  70], dtype=np.uint8)
_HSV_ROJO_MARCA_HI2 = np.array([179, 255, 255], dtype=np.uint8)


def _clamp(valor, minimo=-1.0, maximo=1.0):
    return max(minimo, min(maximo, valor))


# =====================================================================
# SEGMENTACION HSV (linea azul + zonas rojas/magenta)
# =====================================================================
def _filtrar_componentes(mask, area_min=80):
    """Quita blobs con area menor a `area_min`. Devuelve mascara booleana."""
    mask = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            out[labels == i] = 255
    return out > 0


def segmentar_hsv(bgr):
    """Devuelve (m_lin, m_rojo) booleanas a partir de un frame BGR.

      m_lin  : linea azul del piso.
      m_rojo : flechas y marcas rojas/magenta.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    m_lin = cv2.inRange(hsv, _HSV_AZUL_LO, _HSV_AZUL_HI)
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_OPEN,  _KERNEL_3, iterations=1)
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_CLOSE, _KERNEL_5, iterations=2)
    m_lin = _filtrar_componentes(m_lin, area_min=750)

    m_rojo = (cv2.inRange(hsv, _HSV_ROJO_LO1, _HSV_ROJO_HI1) |
              cv2.inRange(hsv, _HSV_ROJO_LO2, _HSV_ROJO_HI2) |
              cv2.inRange(hsv, _HSV_MAG_LO,   _HSV_MAG_HI))
    m_rojo = cv2.morphologyEx(m_rojo, cv2.MORPH_OPEN,  _KERNEL_3, iterations=1)
    m_rojo = cv2.morphologyEx(m_rojo, cv2.MORPH_CLOSE, _KERNEL_3, iterations=1)
    m_rojo = _filtrar_componentes(m_rojo, area_min=350)

    return m_lin, m_rojo


# =====================================================================
# DESCRIPTOR DE FORMA + LDA DE MARCAS
# =====================================================================
def _silueta_y_descriptor(bgr, area_min=80):
    """Extrae descriptor 11-D (7 log-Hu + 4 ratios) de la mayor silueta
    roja del bgr. Devuelve (desc, bbox) o None."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (cv2.inRange(hsv, _HSV_ROJO_MARCA_LO1, _HSV_ROJO_MARCA_HI1) |
         cv2.inRange(hsv, _HSV_ROJO_MARCA_LO2, _HSV_ROJO_MARCA_HI2))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  _KERNEL_3, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _KERNEL_3, iterations=2)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    idx = int(np.argmax(st[1:, cv2.CC_STAT_AREA])) + 1
    if st[idx, cv2.CC_STAT_AREA] < area_min:
        return None
    x = int(st[idx, cv2.CC_STAT_LEFT])
    y = int(st[idx, cv2.CC_STAT_TOP])
    w = int(st[idx, cv2.CC_STAT_WIDTH])
    h = int(st[idx, cv2.CC_STAT_HEIGHT])
    sil = (lab[y:y + h, x:x + w] == idx).astype(np.uint8) * 255

    hu = cv2.HuMoments(cv2.moments(sil, binaryImage=True)).flatten()
    log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    ratios = np.zeros(4, dtype=np.float32)
    cnts, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        if perim > 1 and area > 1:
            _, _, bw, bh = cv2.boundingRect(cnt)
            hull_area = cv2.contourArea(cv2.convexHull(cnt)) or 1.0
            ratios = np.array([
                bh / bw if bw else 0.0,
                area / (bw * bh) if (bw and bh) else 0.0,
                area / hull_area,
                4 * math.pi * area / (perim * perim),
            ], dtype=np.float32)

    desc = np.concatenate([log_hu, ratios]).astype(np.float32)
    return desc, (x, y, w, h)


def entrenar_lda_marcas():
    """Entrena LDA de 4 clases con los PNG de marcas-capturasStage/."""
    X, y = [], []
    rangos_raw = {c: [] for c in range(len(CLASES_MARCAS))}
    for f in sorted(glob.glob(os.path.join(RUTA_DATASET_MARCAS, '*.png'))):
        m = _RX_MARCA.match(os.path.basename(f))
        if not (m and m.group(1).lower() in CLASES_MARCAS):
            continue
        out = _silueta_y_descriptor(cv2.imread(f))
        if not out:
            continue
        d, _ = out
        clase = CLASES_MARCAS.index(m.group(1).lower())
        X.append(d)
        y.append(clase)
        rangos_raw[clase].append(d[7:11])
    clf = LinearDiscriminantAnalysis(solver='svd').fit(np.stack(X), np.array(y))
    rangos = {k: np.column_stack([np.stack(l).min(0) - 0.05,
                                   np.stack(l).max(0) + 0.05])
              for k, l in rangos_raw.items() if l}
    return clf, rangos


def predecir_marca(bgr, clf, rangos, area_min=300, umbral_conf=0.55,
                   usar_rangos=True):
    out = _silueta_y_descriptor(bgr, area_min)
    if not out:
        return None
    feat, bbox = out
    x, y, w, h = bbox

    # Rechazo si el bbox toca cualquier borde (silueta probablemente cortada)
    if (x <= 5 or y <= 5
            or (x + w) >= (bgr.shape[1] - 5)
            or (y + h) >= (bgr.shape[0] - 5)):
        return None

    # Escudo: figuras MUY alargadas son flechas, no marcas
    if feat[10] < 0.05:
        return None

    probs = clf.predict_proba(feat.reshape(1, -1))[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    if conf < umbral_conf:
        return None
    if usar_rangos:
        rng = rangos.get(pred) if rangos else None
        if rng is not None:
            r = feat[7:11]
            if np.any(r < rng[:, 0]) or np.any(r > rng[:, 1]):
                return None
    return CLASES_MARCAS[pred], conf, bbox


# =====================================================================
# CEREBRO PRINCIPAL
# =====================================================================
class BrainFollowLine(Brain):

    # ---- Camara real (C920) ----
    CAM_W, CAM_H = 640, 360

    # ---- Velocidades y PD ----
    SLOW_FORWARD, FULL_FORWARD = 0.08, 0.40
    KP                    = 1.2
    CRUCE_KP              = 1.8

    # ---- Percepcion ----
    FRANJA_ERROR, BANDA_BORDE = 60, 6
    MIN_SEGMENTO, FUSION_GAP  = 12, 18
    JUNCTION_Y_FRAC           = 0.75
    JUNCTION_MIN_RUN          = 6
    SIDE_EXIT_Y_FRAC          = 0.70

    AREA_MIN_FLECHA           = 350
    MARCA_AREA_MIN            = 450

    FLECHA_CIRC_MAX           = 0.45
    FLECHA_ELONG_MIN          = 2.5
    FLECHA_ASIM_MIN           = 0.30

    MARCA_UMBRAL_CONF_ALTO    = 0.55
    MARCA_UMBRAL_CONF_BAJO    = 0.35
    MARCA_FILTRO_RANGOS       = False

    MARCA_COOLDOWN, ARROW_TTL = 25, 300

    # ---- LOCK de la salida elegida en cruce ----
    CRUCE_LOCK_TICKS_MIN  = 30

    # ---- Evasion ----
    DIST_FRONTAL_OBST     = 0.35
    DIST_FRENTE_LIBRE     = 0.40
    DIST_OBJETIVO_PARED   = 0.40   # 40 cm de margen lateral
    AVOID_TICKS_MIN       = 80
    AVOID_TICKS_MAX       = 250
    AVOID_FLAG            = -1.0   # negativo: tras salir, BUSCAR gira IZQ
    POST_AVOID_GRACE      = 60
    PRINT_EVERY_N_FRAMES  = 15

    AVOID_CORNER_LEFT     = 0.60   # despeje minimo para detectar esquina
    AVOID_EXIT_LEFT       = 0.60   # despeje minimo para terminar el avoid
    AVOID_WALL_GAIN       = 2.5

    # =================================================================
    # SETUP / DESTROY
    # =================================================================
    def setup(self):
        print('Cargando LDA de marcas...')
        self.clf_lda, self.rangos_marcas = entrenar_lda_marcas()
        print('Listo. Todo preparado para la entrega final.')

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,  self.CAM_W)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_H)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        if not self.capture.isOpened():
            print("ERROR: no se pudo abrir la camara /dev/video0")
        else:
            print("Camara real abierta correctamente (%dx%d nativo)"
                  % (self.CAM_W, self.CAM_H))

        self.prev_error       = None
        self.last_error       = 0.0
        self.avoiding         = False
        self.avoid_ticks      = 0
        self.post_avoid_grace = 0
        self.arrow_cache      = None
        self.arrow_ttl_left   = 0
        self.last_marca       = None
        self.cooldown_marca   = 0
        self.marcas_vistas    = []
        self._frame_idx       = 0
        self._gui_ok          = True
        self.cruce_salida_lock     = None
        self.cruce_lock_ticks_left = 0

    def destroy(self):
        self.move(0.0, 0.0)
        if hasattr(self, "capture") and self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()

    # =================================================================
    # SONAR Y TEMPORIZADORES
    # =================================================================
    def leer_distancias(self):
        try:
            min_front = min(self.robot.range[i].distance() for i in range(2, 6))
            min_left  = min(self.robot.range[i].distance() for i in range(0, 3))
        except Exception:
            min_front, min_left = 99.0, 99.0
        return min_front, min_left

    def actualizar_temporizadores(self):
        if self.cooldown_marca:   self.cooldown_marca   -= 1
        if self.post_avoid_grace: self.post_avoid_grace -= 1

    def calcular_error_linea(self, m_lin, cx_img):
        franja = m_lin[-self.FRANJA_ERROR:, :].astype(np.uint8)
        M = cv2.moments(franja, binaryImage=True)
        if M['m00'] < 1:
            return None
        return float((M['m10'] / M['m00'] - cx_img) / cx_img)

    def _linea_cerca(self, m_lin):
        """True solo si hay linea en los ultimos 20 pixeles INFERIORES.

        Se usa durante el avoid como condicion de salida: evita que el
        robot abandone la maniobra al ver una linea LEJANA en la parte
        alta de la imagen.
        """
        return int(m_lin[-20:, :].sum()) > 200

    # =================================================================
    # PERCEPCION: EXTREMOS, CRUCES Y FLECHA
    # =================================================================
    def detectar_extremos(self, m_lin):
        h, w = m_lin.shape
        b, cx, cy = self.BANDA_BORDE, w / 2.0, h / 2.0
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
                    if (z - a) >= self.MIN_SEGMENTO]
            if not segs:
                continue
            fus = [segs[0]]
            for s, e in segs[1:]:
                if s - fus[-1][1] <= self.FUSION_GAP:
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

    def detectar_flecha(self, m_rojo):
        if not m_rojo.any():
            return None
        cnts, _ = cv2.findContours(m_rojo.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < self.AREA_MIN_FLECHA or len(cnt) < 5:
            return None
        perim = cv2.arcLength(cnt, True)
        if (4.0 * math.pi * area / max(perim * perim, 1e-6)) > self.FLECHA_CIRC_MAX:
            return None

        mask = np.zeros(m_rojo.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        ys, xs = np.where(mask > 0)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        eigval, eigvec = np.linalg.eigh(np.cov(pts.T))

        if math.sqrt(float(max(eigval)) /
                     float(max(min(eigval), 1e-6))) < self.FLECHA_ELONG_MIN:
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
        if max_s < 1e-3 or abs(s_pos - s_neg) / max_s < self.FLECHA_ASIM_MIN:
            return None

        idx = int(np.argmax(proy_a)) if s_pos > s_neg else int(np.argmin(proy_a))
        px, py = float(pts[idx, 0]), float(pts[idx, 1])

        return {
            'angulo': math.degrees(math.atan2(cy - py, px - cx)),
            'centro': (cx, cy), 'contorno': cnt, 'punta': (px, py),
        }

    def _reportar_marca(self, marca):
        if (marca is not None and self.cooldown_marca == 0
                and marca[0] != self.last_marca):
            clase, conf, _ = marca
            print('>>> MARCA: %s (%.2f)' % (clase, conf))
            self.last_marca     = clase
            self.cooldown_marca = self.MARCA_COOLDOWN

    def procesar_rojo(self, bgr, m_rojo):
        """Cascada: flecha primero, luego marca alta, luego marca baja."""
        if not m_rojo.any():
            return None, None

        flecha_visual = self.detectar_flecha(m_rojo)
        if flecha_visual is not None:
            return None, flecha_visual

        marca_alta = predecir_marca(bgr, self.clf_lda, self.rangos_marcas,
                                     area_min=self.MARCA_AREA_MIN,
                                     umbral_conf=self.MARCA_UMBRAL_CONF_ALTO,
                                     usar_rangos=self.MARCA_FILTRO_RANGOS)
        if marca_alta is not None:
            self._reportar_marca(marca_alta)
            return marca_alta, None

        marca_baja = predecir_marca(bgr, self.clf_lda, self.rangos_marcas,
                                     area_min=self.MARCA_AREA_MIN,
                                     umbral_conf=self.MARCA_UMBRAL_CONF_BAJO,
                                     usar_rangos=self.MARCA_FILTRO_RANGOS)
        if marca_baja is not None:
            self._reportar_marca(marca_baja)
        return marca_baja, None

    def actualizar_memoria_flecha(self, flecha_visual):
        if flecha_visual:
            self.arrow_cache    = flecha_visual
            self.arrow_ttl_left = self.ARROW_TTL
        elif self.arrow_ttl_left > 0:
            self.arrow_ttl_left -= 1
        return self.arrow_cache if self.arrow_ttl_left > 0 else None

    # =================================================================
    # ELECCION DE SALIDA + LOCK
    # =================================================================
    def elegir_salida(self, extremos, flecha):
        salidas = [e for e in extremos if not e['es_entrada']]
        if not salidas: return None
        if not flecha:
            return min(salidas, key=lambda e: abs((e['angulo'] - 90 + 180) % 360 - 180))

        # Mapeo del angulo de la flecha a un LADO discreto.
        # Convencion (atan2 estilo): 0=derecha, +90=arriba, +/-180=izquierda, -90=abajo
        a = ((flecha['angulo'] + 180.0) % 360.0) - 180.0
        if   abs(a) >= 135.0: lado_obj = 'izquierda'
        elif abs(a) <=  45.0: lado_obj = 'derecha'
        elif a > 0:           lado_obj = 'arriba'
        else:                 lado_obj = 'abajo'

        # 1) Si hay salida del lado exacto que indica la flecha, esa es.
        mismo_lado = [e for e in salidas if e['lado'] == lado_obj]
        if mismo_lado:
            return min(mismo_lado, key=lambda e: abs((e['angulo'] - flecha['angulo'] + 180) % 360 - 180))

        # 2) Fallback angular: la salida cuyo angulo este mas cerca al de la flecha
        return min(salidas, key=lambda e: abs((e['angulo'] - flecha['angulo'] + 180) % 360 - 180))

    def _actualizar_salida_lock(self, extremos):
        if self.cruce_salida_lock is None:
            return None
        lock = self.cruce_salida_lock
        candidatos = [e for e in extremos
                      if not e['es_entrada'] and e['lado'] == lock['lado']]
        if not candidatos:
            return lock
        return min(candidatos,
                   key=lambda e: abs(e['punto'][0] - lock['punto'][0]))

    # =================================================================
    # OVERLAY (GUI)
    # =================================================================
    def overlay(self, bgr, m_lin, m_rojo, ext, sel,
                flecha, marca, err, v, w, estado):
        h, wi = bgr.shape[:2]
        ov = bgr.copy()
        ov[m_lin]  = (255, 0, 0)
        ov[m_rojo] = (0, 0, 255)
        out = cv2.addWeighted(ov, 0.3, bgr, 0.7, 0)
        cv2.line(out, (0, h - self.FRANJA_ERROR),
                 (wi, h - self.FRANJA_ERROR), (180, 180, 180), 1)

        for e in ext:
            es_eleg = sel is not None and e['punto'] == sel['punto']
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
            self._txt(out, etq, (tx, ty), col)

        if flecha:
            cv2.drawContours(out, [flecha['contorno']], -1, (255, 255, 0), 1)
            cv2.arrowedLine(out, tuple(map(int, flecha['centro'])),
                            tuple(map(int, flecha['punta'])),
                            (0, 255, 255), 2, tipLength=0.30)

        if marca:
            x0, y0, bw, bh = marca[2]
            cv2.rectangle(out, (x0, y0), (x0 + bw, y0 + bh), (0, 200, 255), 2)
            self._txt(out, marca[0], (x0, max(12, y0 - 4)),
                      (0, 200, 255), scale=0.45)

        lineas = (
            "Estado: %s" % estado,
            "Eleg: %s" % (sel['lado'] if sel else '--'),
            "Flecha: %s" % ('%+5.0f deg' % flecha['angulo'] if flecha else '--'),
            "Error: %s" % ('%+.2f' % err if err is not None else '--'),
            "v=%+.2f w=%+.2f" % (v, w),
        )
        for i, t in enumerate(lineas):
            self._txt(out, t, (4, 14 + i * 13), (255, 255, 255))
        return out

    @staticmethod
    def _txt(img, txt, pos, color, scale=0.4):
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 1, cv2.LINE_AA)

    def _mostrar(self, img, title='Brain Camera'):
        if not self._gui_ok:
            return
        try:
            cv2.imshow(title, img)
            cv2.waitKey(1)
        except Exception:
            print("AVISO: no se puede mostrar ventana. Desactivo display.")
            self._gui_ok = False

    # =================================================================
    # BUCLE PRINCIPAL
    # =================================================================
    def step(self):
        ok, bgr = self.capture.read()
        if not ok or bgr is None:
            self.move(0.0, 0.0)
            return

        h_img, w_img = bgr.shape[:2]
        cx_img = w_img / 2.0
        self._frame_idx += 1

        # --- Percepcion ---
        m_lin, m_rojo = segmentar_hsv(bgr)
        min_front, min_left = self.leer_distancias()
        self.actualizar_temporizadores()

        extremos = self.detectar_extremos(m_lin)
        salidas  = [e for e in extremos if not e['es_entrada']]

        error = self.calcular_error_linea(m_lin, cx_img)
        marca_actual, flecha_visual = self.procesar_rojo(bgr, m_rojo)
        flecha_logica = self.actualizar_memoria_flecha(flecha_visual)
        cruce_inminente = self.hay_cruce_inminente(m_lin, salidas, h_img)

        # --- Salida elegida con LOCK ---
        if cruce_inminente and len(salidas) >= 2:
            if self.cruce_salida_lock is None:
                self.cruce_salida_lock = self.elegir_salida(extremos, flecha_logica)
                if self.cruce_salida_lock is not None:
                    print('>>> CRUCE: lock=%s  flecha=%s'
                          % (self.cruce_salida_lock['lado'],
                             ('%+.0f deg' % flecha_logica['angulo'])
                             if flecha_logica else '--'))
            self.cruce_lock_ticks_left = self.CRUCE_LOCK_TICKS_MIN
            actualizada = self._actualizar_salida_lock(extremos)
            if actualizada is not None:
                self.cruce_salida_lock = actualizada
            salida_elegida = self.cruce_salida_lock
        elif self.cruce_lock_ticks_left > 0 and self.cruce_salida_lock is not None:
            self.cruce_lock_ticks_left -= 1
            actualizada = self._actualizar_salida_lock(extremos)
            if actualizada is not None:
                self.cruce_salida_lock = actualizada
            salida_elegida = self.cruce_salida_lock
            if self.cruce_lock_ticks_left == 0:
                print('>>> CRUCE: lock liberado')
                self.cruce_salida_lock = None
        else:
            self.cruce_salida_lock = None
            salida_elegida = (self.elegir_salida(extremos, flecha_logica)
                              if len(salidas) >= 2 else None)

        # --- Control de motores ---
        v_cmd, w_cmd, estado = 0.0, 0.0, ''

        if min_front < self.DIST_FRONTAL_OBST and not self.avoiding:
            self.avoiding    = True
            self.avoid_ticks = 0
            self.prev_error  = None

        if self.avoiding:
            self.avoid_ticks += 1
            # found_line ESTRICTO: solo cuenta si la linea esta REALMENTE
            # abajo (cerca del robot), no una lejana en la parte alta.
            found_line = self._linea_cerca(m_lin)
            timeout    = self.avoid_ticks > self.AVOID_TICKS_MAX
            lateral_libre = min_left > self.AVOID_EXIT_LEFT

            if (self.avoid_ticks > self.AVOID_TICKS_MIN
                    and min_front > self.DIST_FRENTE_LIBRE
                    and lateral_libre
                    and (found_line or timeout)):
                self.avoiding   = False
                # AVOID_FLAG = -1.0 -> en BUSCAR girara IZQUIERDA
                self.last_error = self.AVOID_FLAG
                self.prev_error = None
                self.post_avoid_grace = self.POST_AVOID_GRACE
                self.move(0.0, 0.0)
                return

            # Maquina de estados del avoid (3 casos):
            if min_front < self.DIST_FRONTAL_OBST:
                # CASO A: caja al frente -> girar derecha, sin avanzar
                v_cmd, w_cmd, estado = 0.0, -1.0, 'AVOID-FRONT'
            elif min_left > self.AVOID_CORNER_LEFT:
                # CASO B: izquierda despejada -> envolver con curva amplia
                # v=0.15, w=0.6 -> radio 25 cm
                v_cmd, w_cmd, estado = 0.15, 0.6, 'AVOID-CORNER'
            else:
                # CASO C: bordear la caja con P-control sobre min_left
                w_cmd = _clamp(-self.AVOID_WALL_GAIN
                               * (self.DIST_OBJETIVO_PARED - min_left))
                v_cmd, estado = 0.15, 'AVOID-WALL'

        elif cruce_inminente and salida_elegida is not None:
            err_x = (salida_elegida['punto'][0] - cx_img) / cx_img
            w_cmd = _clamp(-self.CRUCE_KP * err_x)
            v_cmd = max(self.SLOW_FORWARD,
                        self.FULL_FORWARD * (1.0 - 0.9 * abs(w_cmd)))
            self.last_error = err_x
            self.prev_error = None
            estado = 'CRUCE -> ' + salida_elegida['lado']

        elif error is None:
            # Sin linea -> arco para reencontrarla
            self.prev_error = None
            w_cmd = -1.2 if self.last_error > 0 else 1.2
            v_cmd, estado = 0.1, 'BUSCAR (arc)'

        else:
            # Seguimiento PD normal
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
        out = self.overlay(bgr, m_lin, m_rojo, extremos, salida_elegida,
                           flecha_visual, marca_actual, error,
                           v_cmd, w_cmd, estado)
        self._mostrar(out)


def INIT(engine):
    assert (engine.robot.requires('range-sensor')
            and engine.robot.requires('continuous-movement'))
    return BrainFollowLine('BrainFollowLine', engine)
