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


def segmentar_qda(clf, bgr):
    """Devuelve mascaras booleanas para linea azul y objetos rojos."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pred = clf.predict(_features_pixel(rgb)).reshape(rgb.shape[:2])
    m_lin = (pred == 2).astype(np.uint8) * 255
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_OPEN,  _KERNEL_3, iterations=1)
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_CLOSE, _KERNEL_3, iterations=3)
    m_mar = (pred == 1).astype(np.uint8) * 255
    m_mar = cv2.morphologyEx(m_mar, cv2.MORPH_OPEN,  _KERNEL_3, iterations=1)
    m_mar = cv2.morphologyEx(m_mar, cv2.MORPH_CLOSE, _KERNEL_3, iterations=2)
    return m_lin > 0, m_mar > 0


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
    """Entrena el clasificador de marcas a partir de capturas etiquetadas."""
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


def predecir_marca(bgr, clf, rangos, area_min=300, umbral_conf=0.55):
    """Clasifica una marca roja si su forma y confianza son consistentes."""
    out = _silueta_y_descriptor(bgr, area_min)
    if not out:
        return None
    feat, bbox = out
    if (bbox[1] + bbox[3]) >= (bgr.shape[0] - 5):
        return None
    probs = clf.predict_proba(feat.reshape(1, -1))[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    if conf < umbral_conf:
        return None
    rng = rangos.get(pred) if rangos else None
    if rng is not None:
        r = feat[7:11]
        if np.any(r < rng[:, 0]) or np.any(r > rng[:, 1]):
            return None
    return CLASES_MARCAS[pred], conf, bbox


class BrainFollowLine(Brain):
    """Cerebro que sigue la linea, interpreta flechas/marcas y evita obstaculos."""

    # Velocidades y control de seguimiento.
    SLOW_FORWARD, FULL_FORWARD = 0.05, 0.40
    KP                    = 1.2
    CRUCE_KP              = 1.6

    # Parametros de percepcion.
    FRANJA_ERROR, BANDA_BORDE = 40, 4
    MIN_SEGMENTO, FUSION_GAP  = 5, 8
    JUNCTION_Y_FRAC           = 0.75
    JUNCTION_MIN_RUN          = 3
    SIDE_EXIT_Y_FRAC          = 0.80
    AREA_MIN_FLECHA           = 120
    FLECHA_CIRC_MAX           = 0.35
    FLECHA_ELONG_MIN          = 3.0
    FLECHA_ASIM_MIN           = 0.40
    MARCA_AREA_MIN            = 300
    MARCA_COOLDOWN, ARROW_TTL = 25, 300

    # Parametros de evasion de obstaculos.
    DIST_FRONTAL_OBST     = 0.40
    DIST_FRENTE_LIBRE     = 0.40
    DIST_OBJETIVO_PARED   = 0.30
    AVOID_TICKS_MIN       = 40
    AVOID_TICKS_MAX       = 250
    AVOID_EXIT_ERR_MAX    = 0.50
    AVOID_FLAG            = 1.0
    POST_AVOID_GRACE      = 60

def setup(self):
    print('Cargando modelos QDA y LDA...')
    self.clf_qda = entrenar_qda_linea()
    self.clf_lda, self.rangos_marcas = entrenar_lda_marcas()
    print('Listo. Todo preparado para la entrega final.')

    # CAMARA REAL
    self.capture = cv2.VideoCapture(0)

    # Resolucion recomendada para no ir lentisimo
    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    self.capture.set(cv2.CAP_PROP_FPS, 30)

    if not self.capture.isOpened():
        print("ERROR: no se pudo abrir la camara real")
    else:
        print("Camara real abierta correctamente")

    self.prev_error = None
    self.last_error = 0.0
    self.avoiding = False
    self.avoid_ticks = 0
    self.post_avoid_grace = 0
    self.arrow_cache = None
    self.arrow_ttl_left = 0
    self.last_marca = None
    self.cooldown_marca = 0
    self.marcas_vistas = []

    def destroy(self):
        self.move(0.0, 0.0)

        if hasattr(self, "capture") and self.capture is not None:
            self.capture.release()

        if self.marcas_vistas:
            print('=== Marcas detectadas ===')
            for i, (c, p, x, y) in enumerate(self.marcas_vistas, 1):
                print('  %d. %s (%.2f) en img=(%d,%d)' % (i, c, p, x, y))

        cv2.destroyAllWindows()

    def detectar_extremos(self, m_lin):
        """Busca entradas y salidas de la linea en los bordes de la imagen."""
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
        """Localiza la fila inferior donde aparecen dos segmentos de linea."""
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

    def detectar_flecha(self, m_rojo):
        """Detecta una flecha roja alargada y estima su direccion."""
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
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        eigval, eigvec = np.linalg.eigh(np.cov(pts.T))
        ev_max = float(max(eigval))
        ev_min = float(max(min(eigval), 1e-6))
        if math.sqrt(ev_max / ev_min) < self.FLECHA_ELONG_MIN:
            return None

        eje  = eigvec[:, int(np.argmax(eigval))]
        perp = np.array([-eje[1], eje[0]])
        proy_a = (pts[:, 0] - cx) * eje[0]  + (pts[:, 1] - cy) * eje[1]
        proy_p = (pts[:, 0] - cx) * perp[0] + (pts[:, 1] - cy) * perp[1]
        pmin, pmax = float(proy_a.min()), float(proy_a.max())
        L = pmax - pmin
        if L < 5:
            return None
        f = 0.25 * L
        pp_pos = proy_p[proy_a > pmax - f]
        pp_neg = proy_p[proy_a < pmin + f]
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

    def elegir_salida(self, extremos, flecha):
        """Elige la salida cuyo angulo coincide mejor con la flecha."""
        salidas = [e for e in extremos if not e['es_entrada']]
        if not salidas:
            return None
        ref = flecha['angulo'] if flecha else 90.0
        return min(salidas,
                   key=lambda e: abs((e['angulo'] - ref + 180) % 360 - 180))

    def leer_distancias(self):
        """Lee sensores de rango; si fallan, devuelve distancias seguras."""
        try:
            min_front = min(self.robot.range[i].distance() for i in range(2, 6))
            min_left = min(self.robot.range[i].distance() for i in range(0, 3))
        except Exception:
            min_front, min_left = 99.0, 99.0
        return min_front, min_left

    def actualizar_temporizadores(self):
        if self.cooldown_marca:
            self.cooldown_marca -= 1
        if self.post_avoid_grace:
            self.post_avoid_grace -= 1

    def calcular_error_linea(self, m_lin, cx_img):
        """Calcula el error horizontal de la linea en la franja inferior."""
        franja = m_lin[-self.FRANJA_ERROR:, :].astype(np.uint8)
        M = cv2.moments(franja, binaryImage=True)
        if M['m00'] < 1:
            return None
        return float((M['m10'] / M['m00'] - cx_img) / cx_img)

    def procesar_rojo(self, bgr, m_rojo):
        """Distingue entre marca conocida y flecha en la zona roja."""
        marca_actual = None
        flecha_visual = None
        if m_rojo.any():
            marca_actual = predecir_marca(bgr, self.clf_lda,
                                          self.rangos_marcas,
                                          self.MARCA_AREA_MIN)
            if marca_actual is None:
                flecha_visual = self.detectar_flecha(m_rojo)
            elif self.cooldown_marca == 0 and marca_actual[0] != self.last_marca:
                clase, conf, bbox = marca_actual
                cxm, cym = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
                print('>>> MARCA: %s (%.2f) en img=(%d,%d)'
                      % (clase, conf, cxm, cym))
                self.marcas_vistas.append((clase, conf, cxm, cym))
                self.last_marca = clase
                self.cooldown_marca = self.MARCA_COOLDOWN
        return marca_actual, flecha_visual

    def actualizar_memoria_flecha(self, flecha_visual):
        if flecha_visual:
            self.arrow_cache = flecha_visual
            self.arrow_ttl_left = self.ARROW_TTL
        elif self.arrow_ttl_left > 0:
            self.arrow_ttl_left -= 1
        return self.arrow_cache if self.arrow_ttl_left > 0 else None

    def hay_cruce_inminente(self, m_lin, salidas, h_img):
        """Detecta si hay una bifurcacion cercana."""
        if self.post_avoid_grace != 0 or len(salidas) < 2:
            return False

        jy = self.junction_y(m_lin)
        if jy is not None and jy >= h_img * self.JUNCTION_Y_FRAC:
            return True

        lim_y = h_img * self.SIDE_EXIT_Y_FRAC
        for salida in salidas:
            if (salida['lado'] in ('izquierda', 'derecha')
                    and salida['punto'][1] >= lim_y):
                return True
        return False

    def overlay(self, bgr, m_lin, m_rojo, ext, sel, flecha, marca,
                err, v, w, estado):
        """Dibuja informacion de depuracion sobre la imagen de camara."""
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
            cv2.putText(out, etq, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, etq, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, col, 1, cv2.LINE_AA)

        if flecha:
            cv2.drawContours(out, [flecha['contorno']], -1, (255, 255, 0), 1)
            cf = tuple(map(int, flecha['centro']))
            pf = tuple(map(int, flecha['punta']))
            cv2.arrowedLine(out, cf, pf, (0, 255, 255), 2, tipLength=0.30)

        if marca:
            x0, y0, bw, bh = marca[2]
            cv2.rectangle(out, (x0, y0), (x0 + bw, y0 + bh), (0, 200, 255), 2)
            cv2.putText(out, marca[0], (x0, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 200, 255), 1, cv2.LINE_AA)

        ang_str = '%+5.0f deg' % flecha['angulo'] if flecha else '--'
        err_str = '%+.2f' % err if err is not None else '--'
        for i, txt in enumerate((
                'Estado: ' + estado,
                'Eleg: ' + (sel['lado'] if sel else '--'),
                'Flecha (mem): ' + ang_str,
                'Error: ' + err_str,
                'v=%+.2f w=%+.2f' % (v, w))):
            yy = 14 + i * 13
            cv2.putText(out, txt, (4, yy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, txt, (4, yy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def step(self):
        ok, bgr = self.capture.read()

        if not ok or bgr is None:
            print("No se pudo leer frame de la camara")
            self.move(0.0, 0.0)
            return

        bgr = cv2.resize(bgr, (640, 360))

        h_img, w_img = bgr.shape[:2]
        cx_img = w_img / 2.0

        m_lin, m_rojo = segmentar_qda(self.clf_qda, bgr)


        min_front, min_left = self.leer_distancias()
        self.actualizar_temporizadores()

        extremos = self.detectar_extremos(m_lin)
        salidas  = [e for e in extremos if not e['es_entrada']]

        error = self.calcular_error_linea(m_lin, cx_img)
        marca_actual, flecha_visual = self.procesar_rojo(bgr, m_rojo)
        flecha_logica = self.actualizar_memoria_flecha(flecha_visual)
        cruce_inminente = self.hay_cruce_inminente(m_lin, salidas, h_img)

        salida_elegida = (self.elegir_salida(extremos, flecha_logica)
                          if len(salidas) >= 2 else None)

        v_cmd, w_cmd, estado = 0.0, 0.0, ''

        # Prioridad 1: evasion de obstaculos.
        if min_front < self.DIST_FRONTAL_OBST and not self.avoiding:
            self.avoiding   = True
            self.avoid_ticks = 0
            self.prev_error  = None

        if self.avoiding:
            self.avoid_ticks += 1

            # Sale de la evasion cuando la linea vuelve centrada o vence el tope.
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
                out = self.overlay(bgr, m_lin, m_rojo, extremos,
                                   salida_elegida, flecha_visual,
                                   marca_actual, error, 0.0, 0.0,
                                   'POST-AVOID (grace=%d)' % self.POST_AVOID_GRACE)
                cv2.imshow('Stage Camera Image', out)
                cv2.waitKey(1)
                return

            if min_front < self.DIST_FRONTAL_OBST:
                v_cmd, w_cmd, estado = 0.0, -1.0, 'AVOID-FRONT'
            elif min_left > 0.5:
                v_cmd, w_cmd, estado = 0.0, 1.0, 'AVOID-CORNER'
            else:
                w_cmd = _clamp(-2.5 * (self.DIST_OBJETIVO_PARED - min_left))
                v_cmd, estado = 0.15, 'AVOID-WALL'

        # Prioridad 2: elegir ramal en bifurcaciones.
        elif cruce_inminente and salida_elegida is not None:
            err_x = (salida_elegida['punto'][0] - cx_img) / cx_img
            w_cmd = _clamp(-math.tanh(self.CRUCE_KP * err_x))
            v_cmd = max(0.08, self.FULL_FORWARD - abs(err_x))
            self.last_error = err_x
            self.prev_error = None
            estado = 'CRUCE -> ' + salida_elegida['lado']

        # Prioridad 3: recuperar la linea girando hacia el ultimo error conocido.
        elif error is None:
            self.prev_error = None
            w_cmd = -0.8 if self.last_error > 0 else 0.8
            v_cmd, estado = 0.0, 'BUSCAR (spin)'

        # Prioridad 4: seguimiento normal de linea.
        else:
            if abs(error) > 0.15:
                self.last_error = error
            w_cmd = _clamp(-self.KP * error)
            v_cmd = max(self.SLOW_FORWARD, self.FULL_FORWARD - abs(error))
            estado = ('FOLLOW (grace=%d)' % self.post_avoid_grace
                      if self.post_avoid_grace else 'FOLLOW')

        self.move(v_cmd, w_cmd)
        out = self.overlay(bgr, m_lin, m_rojo, extremos, salida_elegida,
                           flecha_visual, marca_actual,
                           error, v_cmd, w_cmd, estado)
        cv2.imshow('Stage Camera Image', out)
        cv2.waitKey(1)


def INIT(engine):
    assert (engine.robot.requires('range-sensor') and
            engine.robot.requires('continuous-movement'))
    return BrainFollowLine('BrainFollowLine', engine)
