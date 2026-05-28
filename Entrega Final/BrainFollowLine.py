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


_AQUI = os.path.dirname(os.path.abspath(__file__))
RUTA_IMG_ORIGINAL   = os.path.join(_AQUI, 'imagen_original.png')
RUTA_IMG_MARCADA    = os.path.join(_AQUI, 'imagen_marcada.png')
RUTA_DATASET_MARCAS = os.path.join(_AQUI, 'marcas-capturasStage')
CLASES_MARCAS = ('man', 'stairs', 'telephone', 'woman')

_HSV_ROJO_LO1 = np.array([0,   100,  70], dtype=np.uint8)
_HSV_ROJO_HI1 = np.array([12,  255, 255], dtype=np.uint8)
_HSV_ROJO_LO2 = np.array([165, 100,  70], dtype=np.uint8)
_HSV_ROJO_HI2 = np.array([179, 255, 255], dtype=np.uint8)
_KERNEL_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_RX_MARCA = re.compile(r'^([a-z]+)[-_]\d+\.png$', re.IGNORECASE)


def _clamp(valor, minimo=-1.0, maximo=1.0):
    return max(minimo, min(maximo, valor))


def _features_pixel(rgb):
    """Convierte pixeles RGB a rasgos de color para clasificacion."""
    if rgb.ndim == 3:
        rgb = rgb.reshape(-1, 3)
    img = rgb.reshape(1, -1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    hsv *= np.array([1/179.0, 1/255.0, 1/255.0], dtype=np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    L  = lab[:, 0:1] / 255.0
    ab = (lab[:, 1:3] - 128.0) / 128.0
    rgb_f = rgb.astype(np.float32)
    rgb_n = rgb_f / rgb_f.sum(axis=1, keepdims=True).clip(min=1.0)
    return np.hstack([hsv, ab, rgb_n, L]).astype(np.float32)


def entrenar_qda_linea():
    """Entrena el clasificador de fondo, marca roja y linea azul."""
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


def _filtrar_componentes(mask, area_min=80):
    mask = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= area_min:
            out[labels == i] = 255
    return out > 0


def segmentar_qda(clf, bgr):
    """Segmentacion para robot real por umbrales HSV robustos."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # LINEA AZUL REAL
    azul_lo = np.array([90, 60, 40], dtype=np.uint8)
    azul_hi = np.array([130, 255, 255], dtype=np.uint8)
    m_lin = cv2.inRange(hsv, azul_lo, azul_hi)

    # ROJO / MAGENTA PARA FLECHAS Y MARCAS
    rojo_lo1 = np.array([0, 80, 50], dtype=np.uint8)
    rojo_hi1 = np.array([12, 255, 255], dtype=np.uint8)
    rojo_lo2 = np.array([165, 80, 50], dtype=np.uint8)
    rojo_hi2 = np.array([179, 255, 255], dtype=np.uint8)
    mag_lo = np.array([130, 50, 40], dtype=np.uint8)
    mag_hi = np.array([165, 255, 255], dtype=np.uint8)

    m_mar = (
        cv2.inRange(hsv, rojo_lo1, rojo_hi1) |
        cv2.inRange(hsv, rojo_lo2, rojo_hi2) |
        cv2.inRange(hsv, mag_lo, mag_hi)
    )

    # Limpieza morfologica
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_OPEN, kernel3, iterations=1)
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_CLOSE, kernel5, iterations=2)

    m_mar = cv2.morphologyEx(m_mar, cv2.MORPH_OPEN, kernel3, iterations=1)
    m_mar = cv2.morphologyEx(m_mar, cv2.MORPH_CLOSE, kernel3, iterations=1)

    m_lin = _filtrar_componentes(m_lin, area_min=750)
    m_mar = _filtrar_componentes(m_mar, area_min=350)

    return m_lin, m_mar


def _silueta_y_descriptor(bgr, area_min=80):
    """Extrae descriptor de forma de la mayor silueta roja."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (cv2.inRange(hsv, _HSV_ROJO_LO1, _HSV_ROJO_HI1) |
         cv2.inRange(hsv, _HSV_ROJO_LO2, _HSV_ROJO_HI2))
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
    sil = (lab[y:y+h, x:x+w] == idx).astype(np.uint8) * 255

    hu = cv2.HuMoments(cv2.moments(sil, binaryImage=True)).flatten()
    log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    cnts, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ratios = np.zeros(4, dtype=np.float32)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        if perim > 1 and area > 1:
            bx, by, bw, bh = cv2.boundingRect(cnt)
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
    X, y, rangos_raw = [], [], {c: [] for c in range(len(CLASES_MARCAS))}
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
    if (bbox[1] + bbox[3]) >= (bgr.shape[0] - 5):
        return None
        
    # --- ESCUDO MUTUAMENTE EXCLUSIVO ---
    # La circularidad es el índice 10 del descriptor.
    # Si la forma es MUY alargada (circ < 0.20) es una flecha y no una marca.
    # Bajado de 0.45 a 0.20 para que telephone/stairs (mas alargadas)
    # sigan siendo clasificadas como marca.
    if feat[10] < 0.20:
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


class BrainFollowLine(Brain):

    # ---- Parametros de camara real (C920, 640x360 nativo) ----
    CAM_W, CAM_H = 640, 360

    # ---- Velocidades y control de seguimiento ----
    SLOW_FORWARD, FULL_FORWARD = 0.08, 0.40
    KP                    = 1.2
    CRUCE_KP              = 1.8 # Un poco más agresivo para cruces

    # ---- Parametros de percepcion ----
    FRANJA_ERROR, BANDA_BORDE = 60, 6
    MIN_SEGMENTO, FUSION_GAP  = 12, 18
    JUNCTION_Y_FRAC           = 0.75
    JUNCTION_MIN_RUN          = 6
    SIDE_EXIT_Y_FRAC          = 0.70 # Activamos el cruce un pelín antes para que el robot gire a tiempo

    AREA_MIN_FLECHA           = 350
    MARCA_AREA_MIN            = 900

    # Filtros para que no confunda flechas con marcas
    FLECHA_CIRC_MAX           = 0.45 # Límite estricto complementario a las marcas
    FLECHA_ELONG_MIN          = 2.5  # subido: telephone/stairs NO son tan alargados
    FLECHA_ASIM_MIN           = 0.30 # subido: telephone/stairs son simetricos

    MARCA_UMBRAL_CONF_ALTO    = 0.55
    MARCA_UMBRAL_CONF_BAJO    = 0.35
    MARCA_FILTRO_RANGOS       = False

    MARCA_COOLDOWN, ARROW_TTL = 25, 300

    # ---- Lock de la salida elegida en cruce ----
    # Una vez decidida la salida durante un cruce, el lock se mantiene
    # por al menos CRUCE_LOCK_TICKS_MIN frames despues de que el cruce
    # deje de detectarse. Asi el robot no cambia de opinion a mitad
    # del giro por culpa de un flicker visual.
    CRUCE_LOCK_TICKS_MIN  = 30

    # Umbrales de evasion alineados con Practica 1 - Control
    DIST_FRONTAL_OBST     = 0.35   # antes 0.50 (de Practica 1)
    DIST_FRENTE_LIBRE     = 0.40   # antes 0.45 (de Practica 1)
    DIST_OBJETIVO_PARED   = 0.30
    AVOID_TICKS_MIN       = 40
    AVOID_TICKS_MAX       = 250
    AVOID_EXIT_ERR_MAX    = 0.50
    AVOID_FLAG            = 1.0
    POST_AVOID_GRACE      = 60
    PRINT_EVERY_N_FRAMES  = 15

    def setup(self):
        print('Cargando modelos QDA y LDA...')
        self.clf_qda = entrenar_qda_linea()
        self.clf_lda, self.rangos_marcas = entrenar_lda_marcas()
        print('Listo. Todo preparado para la entrega final.')

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,  self.CAM_W)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_H)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        if not self.capture.isOpened():
            print("ERROR: no se pudo abrir la camara /dev/video0")
        else:
            print("Camara real abierta correctamente (%dx%d nativo)" % (self.CAM_W, self.CAM_H))

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

        # Lock de salida elegida en cruce: una vez fijada por la flecha
        # se queda ahi hasta que el cruce termine + ticks de gracia.
        self.cruce_salida_lock     = None
        self.cruce_lock_ticks_left = 0

    def destroy(self):
        self.move(0.0, 0.0)
        if hasattr(self, "capture") and self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()

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
            segs = [(int(a), int(z)) for a, z in zip(ini, fin) if (z - a) >= self.MIN_SEGMENTO]
            if not segs: continue
            fus = [segs[0]]
            for s, e in segs[1:]:
                if s - fus[-1][1] <= self.FUSION_GAP: fus[-1] = (fus[-1][0], e)
                else: fus.append((s, e))
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
                    if n >= 2: return int(y)
        return None

    def detectar_flecha(self, m_rojo):
        if not m_rojo.any(): return None
        cnts, _ = cv2.findContours(m_rojo.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: return None
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < self.AREA_MIN_FLECHA or len(cnt) < 5: return None
        
        perim = cv2.arcLength(cnt, True)
        # Validación complementaria a la marca
        if (4.0 * math.pi * area / max(perim * perim, 1e-6)) > self.FLECHA_CIRC_MAX: return None

        mask = np.zeros(m_rojo.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        ys, xs = np.where(mask > 0)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        eigval, eigvec = np.linalg.eigh(np.cov(pts.T))
        
        if math.sqrt(float(max(eigval)) / float(max(min(eigval), 1e-6))) < self.FLECHA_ELONG_MIN: return None

        eje  = eigvec[:, int(np.argmax(eigval))]
        perp = np.array([-eje[1], eje[0]])
        proy_a = (pts[:, 0] - cx) * eje[0]  + (pts[:, 1] - cy) * eje[1]
        proy_p = (pts[:, 0] - cx) * perp[0] + (pts[:, 1] - cy) * perp[1]
        pmin, pmax = float(proy_a.min()), float(proy_a.max())
        L = pmax - pmin
        if L < 5: return None
        
        # --- LÓGICA DE PUNTA CORREGIDA (Puramente matemática) ---
        franja = 0.25 * L
        pp_pos = proy_p[proy_a > pmax - franja]
        pp_neg = proy_p[proy_a < pmin + franja]
        s_pos = float(pp_pos.max() - pp_pos.min()) if len(pp_pos) >= 5 else 0.0
        s_neg = float(pp_neg.max() - pp_neg.min()) if len(pp_neg) >= 5 else 0.0
        max_s = max(s_pos, s_neg)
        if max_s < 1e-3 or abs(s_pos - s_neg) / max_s < self.FLECHA_ASIM_MIN: return None

        # La punta siempre es el lado donde la figura es más ancha
        idx = int(np.argmax(proy_a)) if s_pos > s_neg else int(np.argmin(proy_a))
        px, py = float(pts[idx, 0]), float(pts[idx, 1])
        
        return {
            'angulo': math.degrees(math.atan2(cy - py, px - cx)),
            'centro': (cx, cy), 'contorno': cnt, 'punta': (px, py),
        }

    def elegir_salida(self, extremos, flecha):
        # --- SELECCIÓN DE SALIDA CORREGIDA ---
        # Se elimina la lógica de los cuadrantes que funcionaba mal.
        # Ahora elije estrictamente la salida que esté más cerca angularmente de la flecha.
        salidas = [e for e in extremos if not e['es_entrada']]
        if not salidas: return None
        if not flecha:
            # Si no hay flecha, por defecto elegimos la que apunte hacia adelante (90 grados)
            return min(salidas, key=lambda e: abs((e['angulo'] - 90 + 180) % 360 - 180))

        return min(salidas, key=lambda e: abs((e['angulo'] - flecha['angulo'] + 180) % 360 - 180))

    def leer_distancias(self):
        try:
            min_front = min(self.robot.range[i].distance() for i in range(2, 6))
            min_left = min(self.robot.range[i].distance() for i in range(0, 3))
        except Exception:
            min_front, min_left = 99.0, 99.0
        return min_front, min_left

    def actualizar_temporizadores(self):
        if self.cooldown_marca: self.cooldown_marca -= 1
        if self.post_avoid_grace: self.post_avoid_grace -= 1

    def calcular_error_linea(self, m_lin, cx_img):
        franja = m_lin[-self.FRANJA_ERROR:, :].astype(np.uint8)
        M = cv2.moments(franja, binaryImage=True)
        if M['m00'] < 1: return None
        return float((M['m10'] / M['m00'] - cx_img) / cx_img)

    def _reportar_marca(self, marca):
        if marca is not None and self.cooldown_marca == 0 and marca[0] != self.last_marca:
            clase, conf, bbox = marca
            print('>>> MARCA: %s (%.2f)' % (clase, conf))
            self.last_marca = clase
            self.cooldown_marca = self.MARCA_COOLDOWN

    def procesar_rojo(self, bgr, m_rojo):
        if not m_rojo.any(): return None, None

        marca_alta = predecir_marca(bgr, self.clf_lda, self.rangos_marcas, area_min=self.MARCA_AREA_MIN, umbral_conf=self.MARCA_UMBRAL_CONF_ALTO, usar_rangos=self.MARCA_FILTRO_RANGOS)
        if marca_alta is not None:
            self._reportar_marca(marca_alta)
            return marca_alta, None

        flecha_visual = self.detectar_flecha(m_rojo)
        if flecha_visual is not None:
            return None, flecha_visual

        marca_baja = predecir_marca(bgr, self.clf_lda, self.rangos_marcas, area_min=self.MARCA_AREA_MIN, umbral_conf=self.MARCA_UMBRAL_CONF_BAJO, usar_rangos=self.MARCA_FILTRO_RANGOS)
        if marca_baja is not None:
            self._reportar_marca(marca_baja)
        return marca_baja, None

    def actualizar_memoria_flecha(self, flecha_visual):
        if flecha_visual:
            self.arrow_cache = flecha_visual
            self.arrow_ttl_left = self.ARROW_TTL
        elif self.arrow_ttl_left > 0:
            self.arrow_ttl_left -= 1
        return self.arrow_cache if self.arrow_ttl_left > 0 else None

    def _actualizar_salida_lock(self, extremos):
        """Si hay un lock activo, lo refresca con la salida ACTUAL del
        mismo lado mas cercana en X. Si ya no hay ninguna salida con ese
        lado, conserva la coordenada vieja (no oscila durante el giro).
        """
        if self.cruce_salida_lock is None:
            return None
        lock = self.cruce_salida_lock
        candidatos = [e for e in extremos
                      if not e['es_entrada'] and e['lado'] == lock['lado']]
        if not candidatos:
            return lock
        return min(candidatos,
                   key=lambda e: abs(e['punto'][0] - lock['punto'][0]))

    def hay_cruce_inminente(self, m_lin, salidas, h_img):
        if self.post_avoid_grace != 0 or len(salidas) < 2: return False
        jy = self.junction_y(m_lin)
        if jy is not None and jy >= h_img * self.JUNCTION_Y_FRAC: return True
        
        lim_y = h_img * self.SIDE_EXIT_Y_FRAC
        for salida in salidas:
            if salida['lado'] in ('izquierda', 'derecha') and salida['punto'][1] >= lim_y: return True
        return False

    def overlay(self, bgr, m_lin, m_rojo, ext, sel, flecha, marca, err, v, w, estado):
        h, wi = bgr.shape[:2]
        ov = bgr.copy()
        ov[m_lin]  = (255, 0, 0)
        ov[m_rojo] = (0, 0, 255)
        out = cv2.addWeighted(ov, 0.3, bgr, 0.7, 0)
        cv2.line(out, (0, h - self.FRANJA_ERROR), (wi, h - self.FRANJA_ERROR), (180, 180, 180), 1)

        for e in ext:
            es_eleg = sel is not None and e['punto'] == sel['punto']
            col, etq, gr = ((0, 255, 0), 'ENT', 2) if e['es_entrada'] else ((255, 255, 0), 'ELEG', 3) if es_eleg else ((0, 255, 255), 'S', 2)
            cv2.circle(out, e['punto'], 8, col, gr)
            tx, ty = e['punto']
            if e['lado'] == 'arriba': ty += 16
            elif e['lado'] == 'abajo': ty -= 6
            elif e['lado'] == 'izquierda': tx += 12
            else: tx -= 36
            cv2.putText(out, etq, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, etq, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

        if flecha:
            cv2.drawContours(out, [flecha['contorno']], -1, (255, 255, 0), 1)
            cv2.arrowedLine(out, tuple(map(int, flecha['centro'])), tuple(map(int, flecha['punta'])), (0, 255, 255), 2, tipLength=0.30)

        if marca:
            x0, y0, bw, bh = marca[2]
            cv2.rectangle(out, (x0, y0), (x0 + bw, y0 + bh), (0, 200, 255), 2)
            cv2.putText(out, marca[0], (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        for i, txt in enumerate((f"Estado: {estado}", f"Eleg: {sel['lado'] if sel else '--'}", f"Flecha: {'%+5.0f deg' % flecha['angulo'] if flecha else '--'}", f"Error: {'%+.2f' % err if err is not None else '--'}", f"v={v:+.2f} w={w:+.2f}")):
            cv2.putText(out, txt, (4, 14 + i * 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, txt, (4, 14 + i * 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _mostrar(self, img, title='Brain Camera'):
        if not self._gui_ok: return
        try:
            cv2.imshow(title, img)
            cv2.waitKey(1)
        except Exception as exc:
            print("AVISO: no se puede mostrar ventana. Desactivo display.")
            self._gui_ok = False

    def step(self):
        ok, bgr = self.capture.read()
        if not ok or bgr is None:
            self.move(0.0, 0.0)
            return

        h_img, w_img = bgr.shape[:2]
        cx_img = w_img / 2.0
        self._frame_idx += 1

        m_lin, m_rojo = segmentar_qda(self.clf_qda, bgr)
        min_front, min_left = self.leer_distancias()
        self.actualizar_temporizadores()

        extremos = self.detectar_extremos(m_lin)
        salidas  = [e for e in extremos if not e['es_entrada']]

        error = self.calcular_error_linea(m_lin, cx_img)
        marca_actual, flecha_visual = self.procesar_rojo(bgr, m_rojo)
        flecha_logica = self.actualizar_memoria_flecha(flecha_visual)
        cruce_inminente = self.hay_cruce_inminente(m_lin, salidas, h_img)

        # --- Salida elegida con LOCK ---
        # Cuando hay cruce inminente y >=2 salidas, COMMIT la decision
        # con la flecha y refresca el TTL del lock. Mientras el cruce
        # siga visible, refrescamos la posicion (mismo lado, x mas cerca).
        # Cuando el cruce ya no se ve, el lock sigue activo CRUCE_LOCK_TICKS_MIN
        # frames mas para que el robot complete el giro sin reconsiderar.
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
            # Lock aun activo aunque el cruce ya no se vea
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

        v_cmd, w_cmd, estado = 0.0, 0.0, ''

        if min_front < self.DIST_FRONTAL_OBST and not self.avoiding:
            self.avoiding   = True
            self.avoid_ticks = 0
            self.prev_error  = None

        if self.avoiding:
            self.avoid_ticks += 1
            # Condicion de salida tipo Practica 1: linea visible + frente libre
            # + minimo de ticks (deja al robot rodear bien antes de aceptar).
            found_line = (error is not None)
            timeout    = self.avoid_ticks > self.AVOID_TICKS_MAX
            if (self.avoid_ticks > self.AVOID_TICKS_MIN
                    and min_front > self.DIST_FRENTE_LIBRE
                    and (found_line or timeout)):
                self.avoiding   = False
                # Inyectamos last_error = +1.0 para forzar giro a la derecha
                # cuando pierda la linea al cruzarla (logica de Practica 1).
                self.last_error = self.AVOID_FLAG
                self.prev_error = None
                self.post_avoid_grace = self.POST_AVOID_GRACE
                self.move(0.0, 0.0)
                return

            if min_front < self.DIST_FRONTAL_OBST: v_cmd, w_cmd, estado = 0.0, -1.0, 'AVOID-FRONT'
            elif min_left > 0.5: v_cmd, w_cmd, estado = 0.0, 1.0, 'AVOID-CORNER'
            else:
                w_cmd = _clamp(-2.5 * (self.DIST_OBJETIVO_PARED - min_left))
                v_cmd, estado = 0.15, 'AVOID-WALL'

        # --- SEGUIMIENTO DE LÍNEA EN CRUCES CORREGIDO ---
        elif cruce_inminente and salida_elegida is not None:
            err_x = (salida_elegida['punto'][0] - cx_img) / cx_img

            # Giro puro proporcional (sin tanh que lo limitaba). Si el error es grande, el giro es fuerte
            w_cmd = _clamp(-self.CRUCE_KP * err_x)

            # Frenado dinámico: Si el robot tiene que girar muy fuerte, avanza extremadamente lento
            # para no salirse de la línea por culpa de la inercia física del Pioneer.
            v_cmd = max(self.SLOW_FORWARD, self.FULL_FORWARD * (1.0 - 0.9 * abs(w_cmd)))

            self.last_error = err_x
            self.prev_error = None
            estado = 'CRUCE -> ' + salida_elegida['lado']

        elif error is None:
            self.prev_error = None
            w_cmd = -0.8 if self.last_error > 0 else 0.8
            v_cmd, estado = 0.0, 'BUSCAR (spin)'

        else:
            if abs(error) > 0.15: self.last_error = error
            
            # Frenado dinámico también en el seguimiento recto con control Derivativo para evitar tambaleos
            d_err = 0.0 if self.prev_error is None else (error - self.prev_error)
            self.prev_error = error
            
            w_cmd = _clamp(-(self.KP * error + 0.6 * d_err))
            v_cmd = max(self.SLOW_FORWARD, self.FULL_FORWARD * (1.0 - 0.8 * abs(w_cmd)))
            estado = ('FOLLOW (grace=%d)' % self.post_avoid_grace if self.post_avoid_grace else 'FOLLOW')

        self.move(v_cmd, w_cmd)
        out = self.overlay(bgr, m_lin, m_rojo, extremos, salida_elegida, flecha_visual, marca_actual, error, v_cmd, w_cmd, estado)
        self._mostrar(out)

def INIT(engine):
    assert (engine.robot.requires('range-sensor') and engine.robot.requires('continuous-movement'))
    return BrainFollowLine('BrainFollowLine', engine)