"""BrainFollowLine.py - Integración Práctica 1 (Control) + Práctica 2 (Percepción).

Examen final - Robótica y Percepción Computacional - UPM 2026.
"""

from pyrobot.brain import Brain

import os
import re
import math
import glob

import cv2
import numpy as np


CLASES_MARCAS = ('man', 'stairs', 'telephone', 'woman')

HSV_AZUL_LO = np.array([90,  80,  60], dtype=np.uint8)
HSV_AZUL_HI = np.array([130, 255, 255], dtype=np.uint8)

HSV_ROJO_LO1 = np.array([0,   80,  60], dtype=np.uint8)
HSV_ROJO_HI1 = np.array([12, 255, 255], dtype=np.uint8)
HSV_ROJO_LO2 = np.array([165, 80,  60], dtype=np.uint8)
HSV_ROJO_HI2 = np.array([179, 255, 255], dtype=np.uint8)


def _kernel(size=3):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _silueta_roja(bgr, area_min=80):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, HSV_ROJO_LO1, HSV_ROJO_HI1) | \
        cv2.inRange(hsv, HSV_ROJO_LO2, HSV_ROJO_HI2)
    k = _kernel(3)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    if not m.any():
        return None
    n, labs, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    if stats[idx, cv2.CC_STAT_AREA] < area_min:
        return None
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
    sil = (labs[y:y+h, x:x+w] == idx).astype(np.uint8) * 255
    return sil, (x, y, w, h)


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
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area else 0.0
    circ = 4 * math.pi * area / (perim ** 2)
    return np.array([aspect, extent, solidity, circ], dtype=np.float32)


def _descriptor(sil):
    return np.concatenate([_log_hu(sil), _ratios_forma(sil)]).astype(np.float32)


class BrainFollowLine(Brain):

    V_FULL = 0.55
    V_MED  = 0.35
    V_SLOW = 0.15
    V_ZERO = 0.0

    KP = 1.4
    KD = 0.6
    BANDA_ERROR_PX = 40
    V_MAX = 0.45
    V_MIN = 0.08
    ALPHA_V = 0.7

    AREA_MIN_LINEA = 120
    AREA_MIN_MARCA = 200
    EXTREMO_BANDA_PX = 4
    EXTREMO_MIN_SEG  = 5
    EXTREMO_GAP      = 8
    EXTREMO_AREA_MIN = 200
    FRAC_MIN_LINEA   = 0.012
    CIRCULARIDAD_FLECHA_MAX = 0.30
    UMBRAL_CLASIF_MARCAS = 6.0
    DEBOUNCE_MARCA_TICKS = 40

    DIST_FRONT_BLOQ = 0.35
    DIST_FRONT_LIBRE = 0.40
    DIST_PARED_OBJ  = 0.30
    AVOID_TICKS_MIN = 40

    def setup(self):
        self._prev_err = None
        self._last_err_sign = 0

        self.avoiding = False
        self.avoid_ticks = 0

        self.cross_lock_ticks = 0

        self.markers_seen = {}
        self.last_marker_class = None
        self.last_marker_debounce = 0

        self.marca_X = None
        self.marca_y = None
        self.marca_norm = None
        self._cargar_dataset_marcas()

        print('---------------------------------------------')
        print('BrainFollowLine - Integracion Practica 1 + 2')
        if self.marca_X is not None:
            print('  Dataset marcas:', self.marca_X.shape[0], 'ejemplos')
        else:
            print('  Dataset marcas NO encontrado')
        print('---------------------------------------------')

    def destroy(self):
        cv2.destroyAllWindows()
        if self.markers_seen:
            print('=============================================')
            print('Marcas detectadas durante el recorrido:')
            for c, n in sorted(self.markers_seen.items()):
                print('   - {}: {} detecciones'.format(c, n))
            print('=============================================')

    def _cargar_dataset_marcas(self):
        candidatos = [
            './marcas-capturasStage',
            '../marcas-capturasStage',
            os.path.expanduser('~/marcas-capturasStage'),
            ('../Practica 2 - Percepcion Computacional/'
             'Percepcion_Computacional/datos/marcas-capturasStage'),
        ]
        try:
            candidatos.insert(2, os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'marcas-capturasStage'))
        except NameError:
            pass
        for c in candidatos:
            if os.path.isdir(c):
                self._cargar_de_carpeta(c)
                if self.marca_X is not None:
                    return

    def _cargar_de_carpeta(self, carpeta):
        rx = re.compile(r'^([a-z]+)[-_]\d+\.png$', re.IGNORECASE)
        feats, labels = [], []
        for f in sorted(glob.glob(os.path.join(carpeta, '*.png'))):
            m = rx.match(os.path.basename(f))
            if m is None:
                continue
            clase = m.group(1).lower()
            if clase not in CLASES_MARCAS:
                continue
            bgr = cv2.imread(f)
            if bgr is None:
                continue
            out = _silueta_roja(bgr, area_min=80)
            if out is None:
                continue
            sil, _ = out
            feats.append(_descriptor(sil))
            labels.append(CLASES_MARCAS.index(clase))
        if not feats:
            return
        X = np.stack(feats).astype(np.float32)
        y = np.array(labels, dtype=np.int32)
        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-6
        self.marca_X = X
        self.marca_y = y
        self.marca_norm = (mu, sd)

    def _mascara_linea(self, hsv):
        m = cv2.inRange(hsv, HSV_AZUL_LO, HSV_AZUL_HI)
        k = _kernel(3)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
        return self._filtrar_componentes(m, self.AREA_MIN_LINEA)

    def _mascara_marca(self, hsv):
        m = cv2.inRange(hsv, HSV_ROJO_LO1, HSV_ROJO_HI1) | \
            cv2.inRange(hsv, HSV_ROJO_LO2, HSV_ROJO_HI2)
        k = _kernel(3)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
        return self._filtrar_componentes(m, self.AREA_MIN_MARCA)

    def _filtrar_componentes(self, mask, area_min):
        if area_min <= 0 or not mask.any():
            return mask
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        keep = np.zeros_like(mask)
        for lbl in range(1, n):
            if stats[lbl, cv2.CC_STAT_AREA] >= area_min:
                keep[labels == lbl] = 255
        return keep

    def _error_seguimiento(self, m_lin, w, h):
        banda = m_lin[-self.BANDA_ERROR_PX:, :]
        if not banda.any():
            return None
        _, xs = np.where(banda)
        cx = xs.mean()
        return float((cx - w / 2.0) / (w / 2.0))

    def _segmentos(self, perfil, min_len):
        p = perfil.astype(np.int8)
        d = np.diff(np.concatenate([[0], p, [0]]))
        s = np.where(d == 1)[0]
        e = np.where(d == -1)[0]
        out = []
        for a, b in zip(s, e):
            if (b - a) >= min_len:
                out.append((int(a), int(b)))
        return out

    def _fusionar(self, segs, gap):
        if not segs:
            return segs
        out = [segs[0]]
        for s, e in segs[1:]:
            ps, pe = out[-1]
            if s - pe <= gap:
                out[-1] = (ps, e)
            else:
                out.append((s, e))
        return out

    def _detectar_extremos(self, m_lin, w, h):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            m_lin, connectivity=8)
        extremos = []
        cx, cy = w / 2.0, h / 2.0
        for lbl in range(1, n):
            if stats[lbl, cv2.CC_STAT_AREA] < self.EXTREMO_AREA_MIN:
                continue
            comp = (labels == lbl)
            bandas = {
                'abajo':     (comp[-self.EXTREMO_BANDA_PX:, :].any(axis=0), 'x'),
                'arriba':    (comp[:self.EXTREMO_BANDA_PX,  :].any(axis=0), 'x'),
                'izquierda': (comp[:, :self.EXTREMO_BANDA_PX].any(axis=1), 'y'),
                'derecha':   (comp[:, -self.EXTREMO_BANDA_PX:].any(axis=1), 'y'),
            }
            for lado, (perfil, eje) in bandas.items():
                segs = self._fusionar(
                    self._segmentos(perfil, self.EXTREMO_MIN_SEG),
                    self.EXTREMO_GAP)
                for s, e in segs:
                    pos = (s + e) // 2
                    if eje == 'x':
                        px = pos
                        py = (h - 1) if lado == 'abajo' else 0
                    else:
                        px = 0 if lado == 'izquierda' else (w - 1)
                        py = pos
                    ang = math.degrees(math.atan2(cy - py, px - cx))
                    extremos.append({
                        'lado': lado, 'pos': int(pos), 'len': int(e - s),
                        'punto': (int(px), int(py)),
                        'es_entrada': (lado == 'abajo'),
                        'angulo': float(ang),
                    })
        return extremos

    def _orientar_flecha(self, m_mar):
        if not m_mar.any():
            return None
        cnts, _ = cv2.findContours(m_mar, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < self.AREA_MIN_MARCA or len(cnt) < 5:
            return None
        perim = cv2.arcLength(cnt, closed=True)
        circ = 4 * math.pi * area / max(perim ** 2, 1e-6)
        if circ > self.CIRCULARIDAD_FLECHA_MAX:
            return None

        ys, xs = np.where(m_mar)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        cov = np.cov(pts.T)
        eigval, eigvec = np.linalg.eigh(cov)
        eje  = eigvec[:, int(np.argmax(eigval))]
        perp = np.array([-eje[1], eje[0]])
        proy_a = (pts[:, 0] - cx) * eje[0]  + (pts[:, 1] - cy) * eje[1]
        proy_p = (pts[:, 0] - cx) * perp[0] + (pts[:, 1] - cy) * perp[1]
        pmin, pmax = float(proy_a.min()), float(proy_a.max())
        longitud = pmax - pmin

        def ancho(banda_proy, banda_perp):
            if len(banda_proy) < 5:
                return 0.0
            return float(banda_perp.max() - banda_perp.min())

        franja = 0.25 * longitud
        sel_pos = proy_a > (pmax - franja)
        sel_neg = proy_a < (pmin + franja)
        sel_cent = np.abs(proy_a) < (0.20 * longitud)
        w_pos = ancho(proy_a[sel_pos], proy_p[sel_pos])
        w_neg = ancho(proy_a[sel_neg], proy_p[sel_neg])
        w_cen = ancho(proy_a[sel_cent], proy_p[sel_cent]) + 1e-6

        ratio_pos = w_pos / w_cen
        ratio_neg = w_neg / w_cen
        sentido = +1 if ratio_pos > ratio_neg else -1

        idx_cabeza = int(np.argmax(proy_a)) if sentido == +1 else int(np.argmin(proy_a))
        px, py = float(pts[idx_cabeza, 0]), float(pts[idx_cabeza, 1])

        ang_flecha = math.degrees(math.atan2(cy - py, px - cx))

        return {
            'centro': (float(cx), float(cy)),
            'angulo': float(ang_flecha),
            'punta': (px, py),
        }

    def _clasificar_marca(self, bgr):
        if self.marca_X is None:
            return None
        out = _silueta_roja(bgr, area_min=self.AREA_MIN_MARCA)
        if out is None:
            return None
        sil, bbox = out
        feat = _descriptor(sil)
        mu, sd = self.marca_norm
        feat_n = (feat - mu) / sd
        Xn = (self.marca_X - mu) / sd
        dists = np.linalg.norm(Xn - feat_n, axis=1)
        k = min(3, len(dists))
        idx = np.argpartition(dists, k - 1)[:k]
        if dists[idx].min() > self.UMBRAL_CLASIF_MARCAS:
            return None
        votos = np.bincount(self.marca_y[idx], minlength=len(CLASES_MARCAS))
        cls = CLASES_MARCAS[int(np.argmax(votos))]
        return (cls, bbox)

    def _decidir_cruce(self, extremos, flecha, w):
        salidas = [e for e in extremos if not e['es_entrada']]
        if not salidas:
            return None
        if len(salidas) == 1:
            elegida = salidas[0]
        elif flecha is None:
            arribas = [s for s in salidas if s['lado'] == 'arriba']
            if arribas:
                elegida = min(arribas, key=lambda s: abs(s['pos'] - w / 2))
            else:
                return None
        else:
            def diff(a, b):
                d = (a - b + 180.0) % 360.0 - 180.0
                return abs(d)
            elegida = min(salidas,
                          key=lambda s: diff(s['angulo'], flecha['angulo']))
        sx, _ = elegida['punto']
        return float((sx - w / 2.0) / (w / 2.0)), elegida

    def step(self):
        cv_image = self.robot.getImage()
        if cv_image is None:
            self.move(self.V_ZERO, 0)
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        m_lin = self._mascara_linea(hsv)
        m_mar = self._mascara_marca(hsv)
        h, w = m_lin.shape

        if m_lin.sum() < self.FRAC_MIN_LINEA * h * w * 255:
            error = None
            found_line = False
            extremos = []
        else:
            error = self._error_seguimiento(m_lin, w, h)
            found_line = error is not None
            extremos = self._detectar_extremos(m_lin, w, h)

        salidas_sup = [e for e in extremos
                       if e['lado'] in ('arriba', 'izquierda', 'derecha')]
        is_cruce = len(salidas_sup) >= 2

        flecha = self._orientar_flecha(m_mar)

        if (not is_cruce) and (flecha is None) and self.last_marker_debounce == 0:
            res_marca = self._clasificar_marca(cv_image)
            if res_marca is not None:
                cls, bbox = res_marca
                if cls != self.last_marker_class:
                    self.markers_seen[cls] = self.markers_seen.get(cls, 0) + 1
                    self.last_marker_class = cls
                    self.last_marker_debounce = self.DEBOUNCE_MARCA_TICKS
                    print('>>> MARCA detectada: {}  bbox={}'.format(cls, bbox))
        else:
            res_marca = None

        if self.last_marker_debounce > 0:
            self.last_marker_debounce -= 1
            if self.last_marker_debounce == 0:
                self.last_marker_class = None

        try:
            front_d = [self.robot.range[i].distance() for i in range(2, 6)]
            left_d  = [self.robot.range[i].distance() for i in range(0, 3)]
            min_front = min(front_d)
            min_left  = min(left_d)
        except Exception:
            min_front = 10.0
            min_left  = 10.0

        debug = cv_image.copy()
        overlay = debug.copy()
        overlay[m_lin > 0] = (255, 200, 0)
        overlay[m_mar > 0] = (0, 200, 255)
        debug = cv2.addWeighted(debug, 0.6, overlay, 0.4, 0)
        for e in extremos:
            color = (0, 255, 0) if e['es_entrada'] else (0, 0, 255)
            cv2.circle(debug, e['punto'], 6, color, 2)
        if flecha is not None:
            c = (int(flecha['centro'][0]), int(flecha['centro'][1]))
            p = (int(flecha['punta'][0]),  int(flecha['punta'][1]))
            cv2.arrowedLine(debug, c, p, (0, 255, 255), 2, tipLength=0.4)
        if res_marca is not None:
            cls, (x, y, ww, hh) = res_marca
            cv2.rectangle(debug, (x, y), (x + ww, y + hh), (0, 200, 255), 2)
            cv2.putText(debug, cls, (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        estado = 'AVOID' if self.avoiding else ('CROSS' if is_cruce else 'LINE')
        if not found_line and not self.avoiding:
            estado = 'LOST'
        cv2.putText(debug, estado, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(debug, estado, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow('Stage Camera Image', debug)
        cv2.waitKey(1)

        if min_front < self.DIST_FRONT_BLOQ and not self.avoiding:
            self.avoiding = True
            self.avoid_ticks = 0
            print('!!! Obstaculo detectado a {:.2f} m - iniciando evasion'
                  .format(min_front))

        if self.avoiding:
            self.avoid_ticks += 1
            if (found_line and min_front > self.DIST_FRONT_LIBRE
                    and self.avoid_ticks > self.AVOID_TICKS_MIN):
                self.avoiding = False
                self._last_err_sign = +1
                self._prev_err = None
                print('--- Evasion completada, reintegrando linea')
            else:
                if min_front < self.DIST_FRONT_BLOQ:
                    self.move(0.0, -1.0)
                elif min_left > 0.5:
                    self.move(0.0, 1.0)
                else:
                    err_pared = self.DIST_PARED_OBJ - min_left
                    tv = max(-1.0, min(1.0, -2.5 * err_pared))
                    self.move(0.15, tv)
                return

        if is_cruce and found_line:
            decision = self._decidir_cruce(extremos, flecha, w)
            if decision is not None:
                err_cruce, elegida = decision
                omega = -self.KP * err_cruce
                v = max(self.V_MIN,
                        self.V_MED * (1.0 - self.ALPHA_V * abs(err_cruce)))
                self.move(v, omega)
                self.cross_lock_ticks = 6
                self._last_err_sign = (+1 if err_cruce > 0 else
                                       (-1 if err_cruce < 0 else
                                        self._last_err_sign))
                if flecha is not None:
                    print('CRUCE: flecha ang={:+.0f}, salida lado={} pos={}'
                          .format(flecha['angulo'], elegida['lado'],
                                  elegida['pos']))
                return

        if not found_line:
            if self._last_err_sign > 0:
                self.move(0.0, -0.6)
            elif self._last_err_sign < 0:
                self.move(0.0, +0.6)
            else:
                self.move(0.0, +0.6)
            return

        derror = 0.0 if self._prev_err is None else (error - self._prev_err)
        self._prev_err = error
        omega = -(self.KP * error + self.KD * derror)
        if omega >  1.0: omega =  1.0
        if omega < -1.0: omega = -1.0
        v = max(self.V_MIN, self.V_MAX * (1.0 - self.ALPHA_V * abs(error)))
        if abs(error) > 0.15:
            self._last_err_sign = +1 if error > 0 else -1
        self.move(v, omega)


def INIT(engine):
    assert (engine.robot.requires('range-sensor') and
            engine.robot.requires('continuous-movement'))
    return BrainFollowLine('BrainFollowLine', engine)
