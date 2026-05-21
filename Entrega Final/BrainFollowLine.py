from pyrobot.brain import Brain
import os
import re
import glob
import math
import cv2
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

# ======================================================================
# Rutas y Constantes
# ======================================================================
_AQUI = os.path.dirname(os.path.abspath(__file__))
RUTA_IMG_ORIGINAL   = os.path.join(_AQUI, 'imagen_original.png')
RUTA_IMG_MARCADA    = os.path.join(_AQUI, 'imagen_marcada.png')
RUTA_DATASET_MARCAS = os.path.join(_AQUI, 'marcas-capturasStage')
CLASES_MARCAS = ('man', 'stairs', 'telephone', 'woman')

# ======================================================================
# Visión: QDA (Línea) y LDA (Marcas)
# ======================================================================
def _features_pixel(rgb):
    if rgb.ndim == 3: rgb = rgb.reshape(-1, 3)
    img = rgb.reshape(1, -1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    hsv[:, 0] /= 179.0; hsv[:, 1] /= 255.0; hsv[:, 2] /= 255.0
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    L = lab[:, 0:1] / 255.0
    a = (lab[:, 1:2] - 128.0) / 128.0
    b = (lab[:, 2:3] - 128.0) / 128.0
    rgb_f = rgb.astype(np.float32)
    s = rgb_f.sum(axis=1, keepdims=True).clip(min=1.0)
    return np.hstack([hsv, a, b, rgb_f / s, L]).astype(np.float32)

def entrenar_qda_linea():
    orig = cv2.cvtColor(cv2.imread(RUTA_IMG_ORIGINAL), cv2.COLOR_BGR2RGB)
    marc = cv2.cvtColor(cv2.imread(RUTA_IMG_MARCADA), cv2.COLOR_BGR2RGB)
    m_marca = (marc[:, :, 0] == 255) & (marc[:, :, 1] == 0) & (marc[:, :, 2] == 0)
    m_fondo = (marc[:, :, 0] == 0) & (marc[:, :, 1] == 255) & (marc[:, :, 2] == 0)
    m_linea = (marc[:, :, 0] == 0) & (marc[:, :, 1] == 0) & (marc[:, :, 2] == 255)
    X = np.vstack([_features_pixel(orig[m_fondo]), _features_pixel(orig[m_marca]), _features_pixel(orig[m_linea])])
    y = np.hstack([np.zeros(int(m_fondo.sum()), dtype=int), np.ones(int(m_marca.sum()), dtype=int), np.full(int(m_linea.sum()), 2, dtype=int)])
    return QuadraticDiscriminantAnalysis(reg_param=0.01).fit(X, y)

def segmentar_qda(clf, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pred = clf.predict(_features_pixel(rgb)).reshape(rgb.shape[:2])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m_lin = cv2.morphologyEx((pred == 2).astype(np.uint8) * 255, cv2.MORPH_OPEN, k, iterations=1)
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_CLOSE, k, iterations=3)
    m_mar = cv2.morphologyEx((pred == 1).astype(np.uint8) * 255, cv2.MORPH_OPEN, k, iterations=1)
    return m_lin > 0, cv2.morphologyEx(m_mar, cv2.MORPH_CLOSE, k, iterations=2) > 0

def _descriptor_marca(sil):
    M = cv2.moments(sil, binaryImage=True)
    hu = cv2.HuMoments(M).flatten()
    log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    cnts, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return np.concatenate([log_hu, np.zeros(4, dtype=np.float32)])
    cnt = max(cnts, key=cv2.contourArea)
    area, perim = cv2.contourArea(cnt), cv2.arcLength(cnt, True)
    if perim <= 1 or area <= 1: return np.concatenate([log_hu, np.zeros(4, dtype=np.float32)])
    x, y, w, h = cv2.boundingRect(cnt)
    return np.concatenate([log_hu, np.array([h/w if w else 0, area/(w*h) if w and h else 0, area/(cv2.contourArea(cv2.convexHull(cnt)) or 1), 4*math.pi*area/(perim**2)], dtype=np.float32)]).astype(np.float32)

def _silueta_de_bgr(bgr, area_min=80):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([0, 100, 70]), np.array([12, 255, 255])) | cv2.inRange(hsv, np.array([165, 100, 70]), np.array([179, 255, 255]))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1), cv2.MORPH_CLOSE, k, iterations=2)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1: return None
    idx = int(np.argmax(st[1:, cv2.CC_STAT_AREA])) + 1
    if st[idx, cv2.CC_STAT_AREA] < area_min: return None
    x, y, w, h = st[idx, cv2.CC_STAT_LEFT], st[idx, cv2.CC_STAT_TOP], st[idx, cv2.CC_STAT_WIDTH], st[idx, cv2.CC_STAT_HEIGHT]
    return (lab[y:y+h, x:x+w] == idx).astype(np.uint8) * 255, (x, y, w, h)

def entrenar_lda_marcas():
    X, y, rangos_raw = [], [], {c: [] for c in range(len(CLASES_MARCAS))}
    for f in sorted(glob.glob(os.path.join(RUTA_DATASET_MARCAS, '*.png'))):
        m = re.match(r'^([a-z]+)[-_]\d+\.png$', os.path.basename(f), re.IGNORECASE)
        if m and m.group(1).lower() in CLASES_MARCAS:
            out = _silueta_de_bgr(cv2.imread(f))
            if out:
                d = _descriptor_marca(out[0])
                X.append(d); y.append(CLASES_MARCAS.index(m.group(1).lower()))
                rangos_raw[y[-1]].append(d[7:11])
    clf = LinearDiscriminantAnalysis(solver='svd').fit(np.stack(X), np.array(y))
    return clf, {k: np.column_stack([np.stack(l).min(0) - 0.05, np.stack(l).max(0) + 0.05]) for k, l in rangos_raw.items() if l}

def predecir_marca(bgr, clf, rangos, area_min=300, umbral_conf=0.55):
    out = _silueta_de_bgr(bgr, area_min)
    if not out: return None
    feat = _descriptor_marca(out[0]).reshape(1, -1)
    probs = clf.predict_proba(feat)[0]
    pred = int(clf.predict(feat)[0])
    conf = float(probs[pred])
    if conf < umbral_conf or (rangos and pred in rangos and (np.any(feat[0, 7:11] < rangos[pred][:, 0]) or np.any(feat[0, 7:11] > rangos[pred][:, 1]))): return None
    return CLASES_MARCAS[pred], conf, out[1]


# ======================================================================
# Cerebro Principal
# ======================================================================
class BrainFollowLine(Brain):
    
    # Velocidades y Control
    NO_FORWARD, SLOW_FORWARD, FULL_FORWARD = 0.0, 0.05, 0.40
    NO_TURN = 0.0
    KP, KD, ALPHA_V_W, ALPHA_V_E = 1.5, 0.6, 0.97, 0.90
    CRUCE_KP = 1.6

    # Umbrales
    FRANJA_ERROR, BANDA_BORDE, MIN_SEGMENTO, FUSION_GAP = 40, 4, 5, 8
    AREA_MIN_FLECHA, FLECHA_CIRC_MAX, FLECHA_ELONG_MIN = 120, 0.35, 3.0
    MARCA_AREA_MIN, MARCA_COOLDOWN, ARROW_TTL = 300, 25, 150
    DIST_FRONTAL_OBST, DIST_FRENTE_LIBRE, DIST_OBJETIVO_PARED, AVOID_TICKS_MIN = 0.40, 0.40, 0.30, 40
    
    JUNCTION_Y_FRAC = 0.75   
    JUNCTION_MIN_RUN = 3

    def setup(self):
        print('Cargando modelos QDA y LDA...')
        self.clf_qda = entrenar_qda_linea()
        self.clf_lda, self.rangos_marcas = entrenar_lda_marcas()
        print('Modelos listos. ¡Todo preparado para la prueba final!')
        
        self.prev_error = None
        self.last_error = 0.0
        self.avoiding = False
        self.avoid_ticks = 0
        self.arrow_cache = None
        self.arrow_ttl_left = 0
        self.last_marca = None
        self.cooldown_marca = 0

    def destroy(self):
        cv2.destroyAllWindows()

    def error_seguimiento(self, m_lin):
        h, w = m_lin.shape
        franja = m_lin[-self.FRANJA_ERROR:, :]
        if not franja.any(): return None
        return float((np.where(franja)[1].mean() - w / 2.0) / (w / 2.0))

    def junction_y(self, m_lin):
        h, w = m_lin.shape
        for y in range(h - 1, -1, -1):
            d = np.diff(np.concatenate([[0], m_lin[y].astype(np.uint8), [0]]))
            ini, fin = np.where(d == 1)[0], np.where(d == -1)[0]
            if sum(1 for s, e in zip(ini, fin) if (e - s) >= self.JUNCTION_MIN_RUN) >= 2:
                return int(y)
        return None

    def detectar_extremos(self, m_lin):
        h, w = m_lin.shape
        b, cx, cy = self.BANDA_BORDE, w/2.0, h/2.0
        bordes = {'abajo': (m_lin[-b:, :].any(0), 'x'), 'arriba': (m_lin[:b, :].any(0), 'x'), 'izquierda': (m_lin[:, :b].any(1), 'y'), 'derecha': (m_lin[:, -b:].any(1), 'y')}
        out = []
        for lado, (perfil, eje) in bordes.items():
            d = np.diff(np.concatenate([[0], perfil.astype(np.int8), [0]]))
            segs = [(int(a), int(b)) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]) if (b - a) >= self.MIN_SEGMENTO]
            if not segs: continue
            fusion = [segs[0]]
            for s, e in segs[1:]:
                if s - fusion[-1][1] <= self.FUSION_GAP: fusion[-1] = (fusion[-1][0], e)
                else: fusion.append((s, e))
            for s, e in fusion:
                pos = (s + e) // 2
                px, py = (pos, h-1 if lado == 'abajo' else 0) if eje == 'x' else (0 if lado == 'izquierda' else w-1, pos)
                out.append({'lado': lado, 'punto': (int(px), int(py)), 'es_entrada': lado == 'abajo', 'angulo': math.degrees(math.atan2(cy - py, px - cx))})
        return out

    def detectar_flecha(self, m_rojo):
        if not m_rojo.any(): return None
        cnts, _ = cv2.findContours(m_rojo.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: return None
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < self.AREA_MIN_FLECHA or len(cnt) < 5: return None
        if (4.0 * math.pi * area / max(cv2.arcLength(cnt, True)**2, 1e-6)) > self.FLECHA_CIRC_MAX: return None

        comp_mask = np.zeros_like(m_rojo.astype(np.uint8))
        cv2.drawContours(comp_mask, [cnt], -1, 255, thickness=-1)
        ys, xs = np.where(comp_mask > 0)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        
        eigval, eigvec = np.linalg.eigh(np.cov(pts.T))
        if math.sqrt(float(max(eigval)) / float(max(min(eigval), 1e-6))) < self.FLECHA_ELONG_MIN: return None
        
        eje = eigvec[:, int(np.argmax(eigval))]
        perp = np.array([-eje[1], eje[0]])
        proy_a = (pts[:, 0] - cx) * eje[0] + (pts[:, 1] - cy) * eje[1]
        proy_p = (pts[:, 0] - cx) * perp[0] + (pts[:, 1] - cy) * perp[1]
        
        longitud = float(proy_a.max() - proy_a.min())
        if longitud < 5: return None
        
        franja = 0.25 * longitud
        sel_pos, sel_neg = proy_a > (proy_a.max() - franja), proy_a < (proy_a.min() + franja)
        sel_cen = np.abs(proy_a) < (0.20 * longitud)
        
        def _span(p): return float(p.max() - p.min()) if len(p) >= 5 else 0.0
        w_pos, w_neg, w_cen = _span(proy_p[sel_pos]), _span(proy_p[sel_neg]), _span(proy_p[sel_cen]) + 1e-6
        
        ratio_pos, ratio_neg = w_pos / w_cen, w_neg / w_cen
        sentido = 1 if ratio_pos > ratio_neg else -1
        
        s_pos = float(proy_p[sel_pos].max() - proy_p[sel_pos].min()) if len(proy_p[sel_pos]) >= 5 else 0.0
        s_neg = float(proy_p[sel_neg].max() - proy_p[sel_neg].min()) if len(proy_p[sel_neg]) >= 5 else 0.0
        max_s = max(s_pos, s_neg)
        if max_s > 1e-3 and abs(s_pos - s_neg) / max_s < 0.40: return None
        
        idx = int(np.argmax(proy_a)) if sentido == 1 else int(np.argmin(proy_a))
        px, py = float(pts[idx, 0]), float(pts[idx, 1])
        return {'angulo': math.degrees(math.atan2(cy - py, px - cx)), 'centro': (cx, cy), 'contorno': cnt, 'punta': (px, py)}

    def elegir_salida(self, extremos, flecha_logica):
        salidas = [e for e in extremos if not e['es_entrada']]
        if not salidas: return None
        if not flecha_logica: return min(salidas, key=lambda e: abs((e['angulo'] - 90 + 180) % 360 - 180))
        return min(salidas, key=lambda e: abs((e['angulo'] - flecha_logica['angulo'] + 180) % 360 - 180))

    def control_pd(self, error):
        d = 0.0 if self.prev_error is None else (error - self.prev_error)
        self.prev_error = error
        omega = max(-1.0, min(1.0, -(self.KP * error + self.KD * d)))
        factor = min(max(0.0, 1.0 - self.ALPHA_V_W * abs(omega)), max(0.0, 1.0 - self.ALPHA_V_E * abs(error)))
        return max(self.SLOW_FORWARD, self.FULL_FORWARD * factor), omega

    def dibujar_overlay(self, bgr, m_lin, m_rojo, ext, sal_elegida, flecha_visual, flecha_logica, marca, err, v, w, estado, junc_y):
        h, w_img = bgr.shape[:2]
        
        overlay = bgr.copy()
        overlay[m_lin] = (255, 0, 0)
        overlay[m_rojo] = (0, 0, 255)
        out = cv2.addWeighted(overlay, 0.3, bgr, 0.7, 0)
        
        if junc_y: cv2.line(out, (0, junc_y), (w_img, junc_y), (0, 200, 255) if junc_y >= h*self.JUNCTION_Y_FRAC else (120, 120, 120), 1)
        cv2.line(out, (0, int(h*self.JUNCTION_Y_FRAC)), (w_img, int(h*self.JUNCTION_Y_FRAC)), (60, 60, 60), 1)
        cv2.line(out, (0, h - self.FRANJA_ERROR), (w_img, h - self.FRANJA_ERROR), (180, 180, 180), 1)
        
        for e in ext:
            es_eleg = sal_elegida and e['punto'] == sal_elegida['punto']
            col = (0, 255, 0) if e['es_entrada'] else (255, 255, 0) if es_eleg else (0, 255, 255)
            cv2.circle(out, e['punto'], 8, col, 3 if es_eleg else 2)
            
            txt_lbl = 'ENT' if e['es_entrada'] else 'ELEG' if es_eleg else 'S'
            tx, ty = e['punto']
            ty += 16 if e['lado'] == 'arriba' else -6 if e['lado'] == 'abajo' else 0
            tx += 12 if e['lado'] == 'izquierda' else -36 if e['lado'] == 'derecha' else 0
            cv2.putText(out, txt_lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(out, txt_lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
            
        if flecha_visual:
            cv2.drawContours(out, [flecha_visual['contorno']], -1, (255, 255, 0), 1)
            cxf, cyf, pxf, pyf = *map(int, flecha_visual['centro']), *map(int, flecha_visual['punta'])
            cv2.arrowedLine(out, (cxf, cyf), (pxf, pyf), (0, 255, 255), 2, tipLength=0.30)
            
        if marca:
            x0, y0, ww, hh = marca[2]
            cv2.rectangle(out, (x0, y0), (x0+ww, y0+hh), (0, 200, 255), 2)
            cv2.putText(out, f"{marca[0]}", (x0, max(12, y0-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        flecha_str = f"{flecha_logica['angulo']:+5.0f} deg" if flecha_logica else "--"
        lineas = [
            f"Estado: {estado}", 
            f"Eleg: {sal_elegida['lado'] if sal_elegida else '--'}", 
            f"Flecha (mem): {flecha_str}",
            f"Error: {err:+.2f}" if err is not None else "Error: --", 
            f"v={v:+.2f} w={w:+.2f}"
        ]
        for i, txt in enumerate(lineas):
            yy = 14 + i*13
            cv2.putText(out, txt, (4, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, txt, (4, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def step(self):
        bgr = self.robot.getImage()
        if bgr is None: return self.move(self.NO_FORWARD, self.NO_TURN)
        
        m_lin, m_rojo = segmentar_qda(self.clf_qda, bgr)
        try:
            min_front = min([self.robot.range[i].distance() for i in range(2, 6)])
            min_left  = min([self.robot.range[i].distance() for i in range(0, 3)])
        except:
            min_front, min_left = 99.0, 99.0

        self.cooldown_marca = max(0, self.cooldown_marca - 1)
        extremos = self.detectar_extremos(m_lin)
        salidas = [e for e in extremos if not e['es_entrada']]
        
        marca_actual = predecir_marca(bgr, self.clf_lda, self.rangos_marcas, self.MARCA_AREA_MIN) if m_rojo.any() else None
        if marca_actual and self.cooldown_marca == 0 and marca_actual[0] != self.last_marca:
            self.last_marca = marca_actual[0]
            self.cooldown_marca = self.MARCA_COOLDOWN

        flecha_visual = self.detectar_flecha(m_rojo) if not marca_actual else None
        if flecha_visual:
            self.arrow_cache, self.arrow_ttl_left = flecha_visual, self.ARROW_TTL
        elif self.arrow_ttl_left > 0:
            self.arrow_ttl_left -= 1
            
        flecha_logica = self.arrow_cache if self.arrow_ttl_left > 0 else None

        error = self.error_seguimiento(m_lin)
        junc_y = self.junction_y(m_lin) if len(salidas) >= 2 else None
        
        salida_elegida = self.elegir_salida(extremos, flecha_logica) if len(salidas) >= 2 else None
        v_cmd, w_cmd, estado = self.NO_FORWARD, self.NO_TURN, ""

        # =================================================================
        # 1. MÁQUINA DE ESTADOS: EVASIÓN CON REINTEGRACIÓN EN CURVA
        # =================================================================
        if min_front < self.DIST_FRONTAL_OBST and not self.avoiding:
            self.avoiding, self.avoid_ticks, self.prev_error = True, 0, None

        if self.avoiding:
            self.avoid_ticks += 1
            
            # Sale de la evasión cuando detecta la línea, pero SIN forzar self.last_error
            # Dejamos que el controlador PD lea el error real de la cámara.
            if error is not None and min_front > self.DIST_FRENTE_LIBRE and self.avoid_ticks > self.AVOID_TICKS_MIN:
                self.avoiding, estado = False, 'POST-AVOID'
            elif min_front < self.DIST_FRONTAL_OBST:
                v_cmd, w_cmd, estado = 0.0, -1.0, 'AVOID-FRONT' 
            elif min_left > 0.5:
                # AQUÍ ESTÁ LA MAGIA: Arco suave en lugar de pivotar en el sitio
                # Esto obliga al robot a morder la línea en diagonal.
                v_cmd, w_cmd, estado = 0.12, 0.5, 'AVOID-CORNER' 
            else:
                w_cmd = max(-1.0, min(1.0, -2.5 * (self.DIST_OBJETIVO_PARED - min_left)))
                v_cmd, estado = 0.15, 'AVOID-WALL'              
                
        # =================================================================
        # 2. CRUCE INMINENTE 
        # =================================================================
        elif len(salidas) >= 2 and junc_y is not None and junc_y >= bgr.shape[0] * self.JUNCTION_Y_FRAC:
            if salida_elegida:
                err_x = (salida_elegida['punto'][0] - bgr.shape[1] / 2.0) / (bgr.shape[1] / 2.0)
                w_cmd = max(-1.0, min(1.0, -self.CRUCE_KP * err_x))
                v_cmd = max(self.SLOW_FORWARD, self.FULL_FORWARD * max(0, 1.0 - self.ALPHA_V_W * abs(w_cmd)))
                self.last_error, self.prev_error, estado = err_x, None, 'CRUCE'
            elif error is not None:
                v_cmd, w_cmd = self.control_pd(error)
                estado = 'CRUCE -> PD'
                
        # =================================================================
        # 3. SEGUIMIENTO NORMAL
        # =================================================================
        elif error is not None:
            if abs(error) > 0.10: self.last_error = error
            v_cmd, w_cmd = self.control_pd(error)
            estado = 'FOLLOW'
            
        # =================================================================
        # 4. BÚSQUEDA 
        # =================================================================
        else:
            self.prev_error = None
            v_cmd, w_cmd, estado = self.SLOW_FORWARD, (-0.6 if self.last_error > 0 else 0.6), 'BUSCAR'

        self.move(v_cmd, w_cmd)
        
        out = self.dibujar_overlay(bgr, m_lin, m_rojo, extremos, salida_elegida, flecha_visual, flecha_logica, marca_actual, error, v_cmd, w_cmd, estado, junc_y)
        cv2.imshow('Stage Camera Image', out)
        cv2.waitKey(1)

def INIT(engine):
    assert (engine.robot.requires('range-sensor') and engine.robot.requires('continuous-movement'))
    return BrainFollowLine('BrainFollowLine', engine)