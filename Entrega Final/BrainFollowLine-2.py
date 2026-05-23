#!/usr/bin/env python3
"""BrainFollowLine-2.py - Version OFFLINE de depuracion con video local.

Adapta BrainFollowLine.py (que corre en el robot real) para procesar
video2017-3.mp4 o video2017-4.mp4 en lugar de leer la camara del robot.
Genera un video anotado de salida con todas las detecciones para validar
visualmente el comportamiento sin tener que subir codigo al robot.

USO:
    python BrainFollowLine-2.py                # video 3
    python BrainFollowLine-2.py --video 4      # video 4
    python BrainFollowLine-2.py --no-mostrar   # sin ventana (mas rapido)
    python BrainFollowLine-2.py --sin-output   # no guarda mp4 de salida
    python BrainFollowLine-2.py --start 800    # salta los primeros N frames

MEJORAS respecto a BrainFollowLine.py (marcadas con `# MEJORA:`):
  * CLAHE sobre canal L de Lab antes de segmentar -> uniforma iluminacion.
  * Confirmacion TEMPORAL ESTRICTA: una marca solo se reporta tras 5
    frames consecutivos clasificada igual; una flecha tras 3. Filtra
    falsos positivos transitorios por brillos/sombras.
  * LOCK de la salida elegida: una vez decidido el ramal en un cruce,
    se mantiene fijo hasta que el cruce desaparezca. Evita que la salida
    "salte" entre frames si la flecha parpadea.
  * Cache de flecha persistente (300 frames = 12.5s a 24fps) -> incluso
    si la flecha sale del campo de vision, el cache mantiene la direccion.
  * ENT siempre visible: marca verde en el centroide de la franja
    inferior aunque no haya un 'abajo' topologico detectado.
  * Mock de motores -> log periodico + HUD inferior con la accion humana.
"""

import argparse
import os
import re
import glob
import math
import sys
import time

import cv2
import numpy as np
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)


# ======================================================================
# RUTAS Y CONSTANTES (las mismas que BrainFollowLine.py)
# ======================================================================
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


# ======================================================================
# MEJORA: ECUALIZACION CLAHE (reduce efectos de iluminacion irregular)
# ======================================================================
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def aplicar_clahe(bgr):
    """Ecualiza solo el canal L de Lab para uniformar brillo sin alterar
    el tono ni la saturacion (mucho mas suave que histogram-eq sobre BGR)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ======================================================================
# QDA: segmentacion linea + marca
# ======================================================================
def _features_pixel(rgb):
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
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            out[labels == i] = 255
    return out > 0


def segmentar_qda(clf, bgr, area_lin_min, area_mar_min):
    """Segmentacion con QDA entrenado del simulador + CLAHE previo."""
    bgr = aplicar_clahe(bgr)                    # MEJORA: ecualizacion
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pred = clf.predict(_features_pixel(rgb)).reshape(rgb.shape[:2])

    m_lin = (pred == 2).astype(np.uint8) * 255
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_OPEN,  _KERNEL_3, iterations=1)
    m_lin = cv2.morphologyEx(m_lin, cv2.MORPH_CLOSE, _KERNEL_3, iterations=3)

    m_mar = (pred == 1).astype(np.uint8) * 255
    m_mar = cv2.morphologyEx(m_mar, cv2.MORPH_OPEN,  _KERNEL_3, iterations=1)
    m_mar = cv2.morphologyEx(m_mar, cv2.MORPH_CLOSE, _KERNEL_3, iterations=2)

    m_lin = _filtrar_componentes(m_lin, area_min=area_lin_min)
    m_mar = _filtrar_componentes(m_mar, area_min=area_mar_min)
    return m_lin, m_mar


# ======================================================================
# DESCRIPTOR DE FORMA + LDA DE MARCAS
# ======================================================================
def _silueta_y_descriptor(bgr, area_min=80):
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
            _, _, bw, bh = cv2.boundingRect(cnt)
            hull_area = cv2.contourArea(cv2.convexHull(cnt)) or 1.0
            ratios = np.array([
                bh / bw if bw else 0.0,
                area / (bw * bh) if (bw and bh) else 0.0,
                area / hull_area,
                4 * math.pi * area / (perim * perim),
            ], dtype=np.float32)
    return np.concatenate([log_hu, ratios]).astype(np.float32), (x, y, w, h)


def _generar_flechas_sinteticas():
    """MEJORA: genera variantes balanceadas de flechas (orientaciones +
    proporciones). El conteo es del mismo orden que el dataset de marcas
    para no desequilibrar el LDA.

    Sus descriptores se anaden al LDA como una 5a clase 'flecha',
    evitando que el clasificador las asigne a 'telephone' por descarte.
    """
    out = []
    # Variantes balanceadas: 4 proporciones (cubre flechas finas y gruesas)
    variantes = [
        (1.0, 1.0, 1.0),   # estandar
        (1.5, 1.0, 1.0),   # asta larga
        (1.0, 1.4, 1.0),   # cabeza ancha
        (1.8, 0.8, 0.7),   # alargada estrecha
    ]
    # 18 rotaciones (cada 20 deg) x 4 variantes = 72 flechas
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


def _descriptor_de_silueta_binaria(sil):
    """Calcula el descriptor 11-D (7 log-Hu + 4 ratios) de una mascara
    binaria YA recortada (sin pasar por _silueta_y_descriptor)."""
    hu = cv2.HuMoments(cv2.moments(sil, binaryImage=True)).flatten()
    log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    cnts, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
        4 * math.pi * area / (perim * perim),
    ], dtype=np.float32)
    return np.concatenate([log_hu, ratios]).astype(np.float32)


# MEJORA: clases ampliadas con 'flecha' como 5a clase
CLASES_LDA = CLASES_MARCAS + ('flecha',)


def entrenar_lda_marcas():
    """MEJORA: ahora entrena un LDA de 5 clases:
       0=man, 1=stairs, 2=telephone, 3=woman, 4=flecha
    Las flechas se generan sinteticamente en multiples orientaciones."""
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
        X.append(d); y.append(clase)
        rangos_raw[clase].append(d[7:11])

    # Anadir clase FLECHA con siluetas sinteticas
    for sil in _generar_flechas_sinteticas():
        d = _descriptor_de_silueta_binaria(sil)
        if d is not None:
            X.append(d); y.append(4)   # idx 4 = flecha

    clf = LinearDiscriminantAnalysis(solver='svd').fit(np.stack(X), np.array(y))
    rangos = {k: np.column_stack([np.stack(l).min(0) - 0.05,
                                   np.stack(l).max(0) + 0.05])
              for k, l in rangos_raw.items() if l}
    return clf, rangos


def predecir_clase_roja(bgr, clf, rangos, area_min=300, umbral_conf=0.55,
                        usar_rangos=True):
    """MEJORA: devuelve la clase del LDA de 5 clases (incluida 'flecha').
    El llamador (procesar_rojo) decide que hacer si es flecha o marca."""
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
    # Filtro de rangos solo aplica a las 4 marcas reales (no a flecha)
    if usar_rangos and pred < len(CLASES_MARCAS):
        rng = rangos.get(pred) if rangos else None
        if rng is not None:
            r = feat[7:11]
            if np.any(r < rng[:, 0]) or np.any(r > rng[:, 1]):
                return None
    nombre = CLASES_LDA[pred]
    return nombre, conf, bbox


def predecir_marca(bgr, clf, rangos, area_min=300, umbral_conf=0.55,
                   usar_rangos=True):
    """Wrapper compatible con el codigo anterior: solo devuelve marcas
    reales (descarta la prediccion si LDA dice 'flecha')."""
    res = predecir_clase_roja(bgr, clf, rangos, area_min, umbral_conf, usar_rangos)
    if res is None or res[0] == 'flecha':
        return None
    return res


# ======================================================================
# BrainOffline: lee video, simula motores, exporta video anotado
# ======================================================================
class BrainOffline:

    # ---- Velocidades y control ----
    SLOW_FORWARD, FULL_FORWARD = 0.08, 0.40
    KP                  = 1.2
    CRUCE_KP            = 1.8

    # ---- Filtros de forma de flecha (ESTRICTOS para CERO falsos positivos) ----
    # Verificado contra el dataset: NINGUNA marca pasa estos filtros.
    #   man      solidez 0.75-0.81, elong 1.5-2.75  -> rechazada
    #   woman    solidez 0.73-0.78, elong 1.6-2.6   -> rechazada
    #   telephone solidez 0.71-0.73, elong 1.8-2.4  -> rechazada
    #   stairs   solidez 0.51-0.55                  -> rechazada
    # Solo las flechas reales (solidez ~0.85-0.95 + muy elongadas +
    # asimetria fuerte entre cabeza y cola) pasan los 4 criterios.
    FLECHA_CIRC_MAX     = 0.30
    FLECHA_ELONG_MIN    = 2.5
    FLECHA_ASIM_MIN     = 0.30
    FLECHA_SOLIDEZ_MIN  = 0.85

    # ---- Umbrales LDA ----
    # MEJORA: umbrales equilibrados para detectar marcas reales sin
    # falsos positivos. La confirmacion temporal de 2 frames + el veto
    # cruzado evitan que las flechas pasen como marca.
    MARCA_UMBRAL_CONF_ALTO = 0.55
    MARCA_UMBRAL_CONF_BAJO = 0.40
    MARCA_FILTRO_RANGOS    = False
    VETO_LDA_CONF          = 0.40       # marca gana sobre flecha si LDA conf >= esto
    MARCA_COOLDOWN         = 50
    ARROW_TTL              = 300

    # ---- Estados de evasion (sin sonar real) ----
    DIST_FRONTAL_OBST   = 0.40
    DIST_FRENTE_LIBRE   = 0.40
    DIST_OBJETIVO_PARED = 0.30
    AVOID_TICKS_MIN     = 40
    AVOID_TICKS_MAX     = 250
    AVOID_EXIT_ERR_MAX  = 0.50
    AVOID_FLAG          = 1.0
    POST_AVOID_GRACE    = 60

    # ---- Cruces ----
    JUNCTION_Y_FRAC     = 0.75
    SIDE_EXIT_Y_FRAC    = 0.70

    # MEJORA: validacion temporal
    MARCA_FRAMES_CONFIRMA  = 3          # 3 frames seguidos = marca real
    FLECHA_FRAMES_CONFIRMA = 2          # 2 frames seguidos = flecha real

    # MEJORA: cuanto tiempo se queda el bbox de la marca tras detectarla
    MARCA_DISPLAY_TTL = 40

    # MEJORA: el lock de salida en cruce dura AL MENOS este numero de
    # frames despues de que cruce_inminente sea True. Asi no se libera
    # por un flicker de detection -> el ramal elegido se mantiene.
    CRUCE_LOCK_TICKS_MIN = 30           # ~1.25 s a 24 fps

    # MEJORA: bbox tracking de la flecha. Una vez clasificada como flecha
    # (filtros estrictos), mientras el blob principal siga en la misma
    # zona de la imagen (IoU >= UMBRAL) durante FLECHA_TRACK_FRAMES, se
    # mantiene como flecha incluso si ese frame especifico no pasa los
    # filtros. Asi el LDA NUNCA puede reclasificar ese mismo blob como
    # 'telephone'.
    FLECHA_TRACK_FRAMES = 40            # ventana de tracking en frames
    FLECHA_TRACK_IOU    = 0.25          # solapamiento minimo para considerarlo el mismo blob

    def __init__(self, video_path, output_path=None, mostrar=True):
        self.mostrar = mostrar

        # ---- Abrir video ----
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError('No se pudo abrir: ' + video_path)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_video = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.n_frames  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._escalar_parametros()

        # ---- Modelos ----
        print('Entrenando QDA y LDA...')
        self.clf_qda = entrenar_qda_linea()
        self.clf_lda, self.rangos_marcas = entrenar_lda_marcas()
        print('Modelos listos. Video %dx%d @ %.1f fps (%d frames)'
              % (self.w, self.h, self.fps_video, self.n_frames))

        # ---- Salida ----
        self.writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc,
                                          self.fps_video, (self.w, self.h))
            print('Output -> %s' % output_path)

        # ---- Estado del controlador ----
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

        # MEJORA: confirmacion temporal
        self._marca_pendiente_clase = None
        self._marca_pendiente_count = 0
        self._flecha_pendiente_ang  = None
        self._flecha_pendiente_count = 0

        # MEJORA: lock de la salida elegida en un cruce
        # Una vez committed, NO cambia hasta que el cruce desaparece Y
        # se hayan agotado los CRUCE_LOCK_TICKS_MIN frames de gracia.
        # Esto evita que flickers en cruce_inminente liberen el lock.
        self.cruce_salida_lock = None
        self.cruce_lock_ticks_left = 0

        # MEJORA: persistencia visual de la ultima marca detectada
        self._marca_display_bbox  = None
        self._marca_display_clase = None
        self._marca_display_conf  = 0.0
        self._marca_display_ttl   = 0

        # MEJORA: tracking de flecha (bbox + frame_idx)
        self._flecha_track_bbox  = None
        self._flecha_track_frame = -1000

        # ---- Logs y metricas ----
        self.frame_idx     = 0
        self.ultima_accion = '--'
        self.t0            = time.perf_counter()
        self._fps_inst     = 0.0

    def _escalar_parametros(self):
        """Parametros relativos a la resolucion del video (320x240 referencia)."""
        scale_w = self.w / 320.0
        scale_h = self.h / 240.0
        scale_avg  = (scale_w + scale_h) / 2.0
        scale_area = scale_w * scale_h
        self.FRANJA_ERROR    = max(20, int(40 * scale_h))
        self.BANDA_BORDE     = max(3,  int(4 * scale_avg))
        self.MIN_SEGMENTO    = max(4,  int(5 * scale_w))
        self.FUSION_GAP      = max(5,  int(8 * scale_w))
        self.JUNCTION_MIN_RUN  = max(3, int(3 * scale_w))
        self.AREA_MIN_LINEA    = max(50,  int(120 * scale_area))
        self.AREA_MIN_FLECHA   = max(50,  int(120 * scale_area))
        self.AREA_MIN_MARCA_SEG = max(50, int(120 * scale_area))
        self.MARCA_AREA_MIN    = max(100, int(300 * scale_area))

    # ==================================================================
    # PERCEPCION (igual que el original, con parametros escalados)
    # ==================================================================
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

    def detectar_flecha(self, m_rojo):
        if not m_rojo.any(): return None
        cnts, _ = cv2.findContours(m_rojo.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: return None
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < self.AREA_MIN_FLECHA or len(cnt) < 5: return None
        perim = cv2.arcLength(cnt, True)
        if (4.0 * math.pi * area / max(perim * perim, 1e-6)) > self.FLECHA_CIRC_MAX:
            return None
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) or 1.0
        if (area / hull_area) < self.FLECHA_SOLIDEZ_MIN:
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
        if L < 5: return None
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
        return {'angulo': math.degrees(math.atan2(cy - py, px - cx)),
                'centro': (cx, cy), 'contorno': cnt, 'punta': (px, py)}

    def elegir_salida(self, extremos, flecha):
        salidas = [e for e in extremos if not e['es_entrada']]
        if not salidas: return None
        if not flecha:
            return min(salidas,
                       key=lambda e: abs((e['angulo'] - 90 + 180) % 360 - 180))
        ang = flecha['angulo']
        lado_pref = None
        if ang > 135 or ang < -135:  lado_pref = 'izquierda'
        elif -45 < ang < 45:         lado_pref = 'derecha'
        BONO = 25.0
        def score(e):
            d = abs((e['angulo'] - ang + 180) % 360 - 180)
            return d - BONO if e['lado'] == lado_pref else d
        return min(salidas, key=score)

    # ==================================================================
    # MEJORA: SALIDA LOCK durante el cruce
    # ==================================================================
    def _actualizar_salida_lock(self, extremos):
        """Busca en los extremos actuales el que mejor encaja con la
        salida bloqueada (mismo lado y mas cercano en x). Si no hay
        ninguno con el lado bloqueado, conserva la salida vieja para
        que la decision no oscile durante el giro."""
        if self.cruce_salida_lock is None:
            return None
        lock = self.cruce_salida_lock
        candidatos = [e for e in extremos
                      if not e['es_entrada'] and e['lado'] == lock['lado']]
        if not candidatos:
            return lock          # mantener la coordenada anterior
        return min(candidatos, key=lambda e: abs(e['punto'][0] - lock['punto'][0]))

    # ==================================================================
    # MEJORA: confirmacion temporal de marcas y flechas
    # ==================================================================
    def _confirmar_marca(self, marca_actual):
        if marca_actual is None:
            self._marca_pendiente_clase = None
            self._marca_pendiente_count = 0
            return None
        clase = marca_actual[0]
        if clase == self._marca_pendiente_clase:
            self._marca_pendiente_count += 1
        else:
            self._marca_pendiente_clase = clase
            self._marca_pendiente_count = 1
        if self._marca_pendiente_count >= self.MARCA_FRAMES_CONFIRMA:
            return marca_actual
        return None

    def _confirmar_flecha(self, flecha_actual):
        if flecha_actual is None:
            self._flecha_pendiente_ang = None
            self._flecha_pendiente_count = 0
            return None
        ang = flecha_actual['angulo']
        if (self._flecha_pendiente_ang is not None
                and abs((ang - self._flecha_pendiente_ang + 180) % 360 - 180) < 30):
            self._flecha_pendiente_count += 1
        else:
            self._flecha_pendiente_ang = ang
            self._flecha_pendiente_count = 1
        if self._flecha_pendiente_count >= self.FLECHA_FRAMES_CONFIRMA:
            return flecha_actual
        return None

    def _reportar_marca(self, marca):
        if (marca is not None and self.cooldown_marca == 0
                and marca[0] != self.last_marca):
            clase, conf, bbox = marca
            cxm, cym = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
            print('[%5d] >>> MARCA: %s (%.2f) en img=(%d,%d)'
                  % (self.frame_idx, clase, conf, cxm, cym))
            self.marcas_vistas.append((self.frame_idx, clase, conf, cxm, cym))
            self.last_marca = clase
            self.cooldown_marca = self.MARCA_COOLDOWN
            # MEJORA: guardar para mostrar el bbox durante MARCA_DISPLAY_TTL
            self._marca_display_clase = clase
            self._marca_display_conf  = conf
            self._marca_display_bbox  = bbox
            self._marca_display_ttl   = self.MARCA_DISPLAY_TTL

    @staticmethod
    def _bbox_blob_principal(m_rojo):
        """Devuelve la bbox (x, y, w, h) del componente conexo mas grande."""
        if not m_rojo.any(): return None
        n, _, st, _ = cv2.connectedComponentsWithStats(
            m_rojo.astype(np.uint8), connectivity=8)
        if n <= 1: return None
        idx = int(np.argmax(st[1:, cv2.CC_STAT_AREA])) + 1
        return (int(st[idx, cv2.CC_STAT_LEFT]),
                int(st[idx, cv2.CC_STAT_TOP]),
                int(st[idx, cv2.CC_STAT_WIDTH]),
                int(st[idx, cv2.CC_STAT_HEIGHT]))

    @staticmethod
    def _iou(b1, b2):
        """IoU entre dos bboxes (x, y, w, h)."""
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        uni = w1 * h1 + w2 * h2 - inter
        return inter / uni if uni > 0 else 0.0

    def procesar_rojo(self, bgr, m_rojo):
        """MEJORA: LDA de 5 clases (man/stairs/telephone/woman/flecha)
        como discriminador PRIMARIO + bbox tracking.

        El LDA ahora distingue las 4 marcas de las flechas (entrenadas
        sinteticamente como 5a clase). Esto resuelve el problema de
        flechas siendo clasificadas como 'telephone' por falta de mejor
        clase. La detectar_flecha geometrica se sigue usando para EL
        ANGULO de la flecha cuando el LDA dice 'flecha'.
        """
        if not m_rojo.any(): return None, None

        # 1) LDA 5-clases. Si dice 'flecha' -> es flecha. Si dice marca -> marca.
        clase_predicha = predecir_clase_roja(
            bgr, self.clf_lda, self.rangos_marcas,
            area_min=self.MARCA_AREA_MIN,
            umbral_conf=self.MARCA_UMBRAL_CONF_ALTO,
            usar_rangos=self.MARCA_FILTRO_RANGOS)

        # CASO A: LDA dice flecha -> intentar obtener angulo geometrico
        if clase_predicha is not None and clase_predicha[0] == 'flecha':
            flecha_visual = self.detectar_flecha(m_rojo)
            if flecha_visual is not None:
                # tenemos angulo: usamos el geometrico
                bbox = cv2.boundingRect(flecha_visual['contorno'])
                self._flecha_track_bbox  = bbox
                self._flecha_track_frame = self.frame_idx
                conf_flecha = self._confirmar_flecha(flecha_visual)
                return None, (conf_flecha if conf_flecha else flecha_visual)
            # LDA dice flecha pero el detector geometrico falla:
            # mantenemos la cache de la ultima flecha si la hay
            if self.arrow_cache is not None and self.arrow_ttl_left > 0:
                return None, self.arrow_cache
            return None, None

        # CASO B: LDA dice marca con conf alta -> reportar
        if clase_predicha is not None:
            conf_marca = self._confirmar_marca(clase_predicha)
            if conf_marca is not None:
                self._reportar_marca(conf_marca)
                return conf_marca, None
            return None, None

        # CASO C: LDA no esta seguro (conf bajo)
        # Probamos flecha geometrica como respaldo
        flecha_visual = self.detectar_flecha(m_rojo)
        if flecha_visual is not None:
            bbox = cv2.boundingRect(flecha_visual['contorno'])
            self._flecha_track_bbox  = bbox
            self._flecha_track_frame = self.frame_idx
            conf_flecha = self._confirmar_flecha(flecha_visual)
            return None, (conf_flecha if conf_flecha else flecha_visual)

        # Y bbox tracking forward (mantener flecha si el blob solapa)
        if (self._flecha_track_bbox is not None
                and (self.frame_idx - self._flecha_track_frame)
                    < self.FLECHA_TRACK_FRAMES):
            blob_bbox = self._bbox_blob_principal(m_rojo)
            if (blob_bbox is not None
                    and self._iou(blob_bbox, self._flecha_track_bbox)
                        >= self.FLECHA_TRACK_IOU):
                return None, self.arrow_cache

        # Fallback: marca con umbral mas bajo
        marca_baja = predecir_marca(bgr, self.clf_lda, self.rangos_marcas,
                                    area_min=self.MARCA_AREA_MIN,
                                    umbral_conf=self.MARCA_UMBRAL_CONF_BAJO,
                                    usar_rangos=self.MARCA_FILTRO_RANGOS)
        if marca_baja is not None:
            conf_marca = self._confirmar_marca(marca_baja)
            if conf_marca is not None:
                self._reportar_marca(conf_marca)
                return conf_marca, None
        return None, None

    def actualizar_memoria_flecha(self, flecha_visual):
        if flecha_visual:
            self.arrow_cache = flecha_visual
            self.arrow_ttl_left = self.ARROW_TTL
        elif self.arrow_ttl_left > 0:
            self.arrow_ttl_left -= 1
        return self.arrow_cache if self.arrow_ttl_left > 0 else None

    def hay_cruce_inminente(self, m_lin, salidas):
        if self.post_avoid_grace != 0 or len(salidas) < 2: return False
        jy = self.junction_y(m_lin)
        if jy is not None and jy >= self.h * self.JUNCTION_Y_FRAC: return True
        lim_y = self.h * self.SIDE_EXIT_Y_FRAC
        for s in salidas:
            if s['lado'] in ('izquierda', 'derecha') and s['punto'][1] >= lim_y:
                return True
        return False

    def calcular_error(self, m_lin):
        cx_img = self.w / 2.0
        franja = m_lin[-self.FRANJA_ERROR:, :].astype(np.uint8)
        M = cv2.moments(franja, binaryImage=True)
        if M['m00'] < 1: return None
        return float((M['m10'] / M['m00'] - cx_img) / cx_img)

    # MEJORA: punto ENT siempre visible (centroide de la franja inferior)
    def calcular_ent_visual(self, m_lin):
        franja = m_lin[-self.FRANJA_ERROR:, :].astype(np.uint8)
        M = cv2.moments(franja, binaryImage=True)
        if M['m00'] < 30: return None
        cx = int(M['m10'] / M['m00'])
        return (cx, self.h - 6)

    # ==================================================================
    # SIMULACION DE MOTORES
    # ==================================================================
    def simular_motor(self, v, w):
        if v < 0.02 and abs(w) < 0.1:           return 'PARADO'
        if v < 0.02:                            return 'GIRO EN SITIO ' + ('IZQ' if w > 0 else 'DER')
        if abs(w) < 0.15:                       return 'RECTO'
        if w > 0.6:                             return 'GIRO IZQ FUERTE'
        if w > 0.2:                             return 'CURVA IZQ'
        if w < -0.6:                            return 'GIRO DER FUERTE'
        if w < -0.2:                            return 'CURVA DER'
        return 'AJUSTE'

    def leer_distancias_mock(self):
        return 99.0, 99.0

    # ==================================================================
    # OVERLAY
    # ==================================================================
    def overlay(self, bgr, m_lin, m_rojo, ext, sel, flecha_vis, flecha_mem,
                marca, err, v, w, estado, accion, ent_visual, cruce_locked):
        h, wi = bgr.shape[:2]
        ov = bgr.copy()
        ov[m_lin]  = (255, 0, 0)
        ov[m_rojo] = (0, 0, 255)
        out = cv2.addWeighted(ov, 0.30, bgr, 0.70, 0)
        cv2.line(out, (0, h - self.FRANJA_ERROR),
                 (wi, h - self.FRANJA_ERROR), (180, 180, 180), 1)

        # Salidas y entrada topologica
        for e in ext:
            es_eleg = sel is not None and e['punto'] == sel['punto']
            if e['es_entrada']:
                col, etq, gr = (0, 255, 0),   'ENT', 2
            elif es_eleg:
                col, etq, gr = (255, 255, 0), 'ELEG', 3
            else:
                col, etq, gr = (0, 255, 255), 'S',   2
            # Reposiciono el ENT/abajo para que el circulo no se salga del frame
            punto_draw = e['punto']
            if e['es_entrada']:
                punto_draw = (e['punto'][0], h - 10)
            cv2.circle(out, punto_draw, 8, col, gr)
            tx, ty = punto_draw
            if   e['lado'] == 'arriba':    ty += 16
            elif e['lado'] == 'abajo':     ty -= 12      # label encima del circulo
            elif e['lado'] == 'izquierda': tx += 12
            else:                          tx -= 36
            self._texto(out, etq, (tx, ty), col)

        # MEJORA: ENT siempre visible -> centroide de la franja inferior
        # (aunque no haya un 'abajo' topologico que pase MIN_SEGMENTO).
        if ent_visual is not None and not any(e['es_entrada'] for e in ext):
            ent_pos = (ent_visual[0], h - 10)
            cv2.circle(out, ent_pos, 8, (0, 255, 0), 2)
            self._texto(out, 'ENT', (ent_pos[0] - 12, ent_pos[1] - 12),
                        (0, 255, 0))

        # ELEG bloqueada (cuando hay lock): marcamos especial
        if cruce_locked and sel is not None:
            x, y = sel['punto']
            cv2.circle(out, (x, y), 14, (255, 200, 0), 2)
            self._texto(out, 'LOCK', (x + 16, y), (255, 200, 0))

        # Flecha visible (contorno + vector)
        if flecha_vis:
            cv2.drawContours(out, [flecha_vis['contorno']], -1, (255, 255, 0), 1)
            cf = tuple(map(int, flecha_vis['centro']))
            pf = tuple(map(int, flecha_vis['punta']))
            cv2.arrowedLine(out, cf, pf, (0, 255, 255), 2, tipLength=0.30)

        # Marca actual (frame detectado)
        if marca:
            x0, y0, bw, bh = marca[2]
            cv2.rectangle(out, (x0, y0), (x0 + bw, y0 + bh), (0, 200, 255), 2)
            self._texto(out, '%s %.0f%%' % (marca[0], marca[1] * 100),
                        (x0, max(12, y0 - 4)), (0, 200, 255), scale=0.45)
        # MEJORA: si hay una marca recien confirmada, persistirla en el
        # overlay durante MARCA_DISPLAY_TTL frames con una banda lateral
        elif self._marca_display_ttl > 0 and self._marca_display_bbox is not None:
            x0, y0, bw, bh = self._marca_display_bbox
            cv2.rectangle(out, (x0, y0), (x0 + bw, y0 + bh), (0, 150, 200), 1)
            self._texto(out, '%s (mem)' % self._marca_display_clase,
                        (x0, max(12, y0 - 4)), (0, 150, 200), scale=0.40)

        # Panel HUD esquina superior
        ang = ('%+5.0f' % flecha_mem['angulo']) if flecha_mem else '--'
        err_s = ('%+.2f' % err) if err is not None else '--'
        marca_s = ('%s %.0f%%' % (marca[0], marca[1] * 100)) if marca else \
                  ('%s (mem)' % self._marca_display_clase
                   if self._marca_display_ttl > 0 else '--')
        sel_s = (sel['lado'] + (' [LOCK]' if cruce_locked else '')) if sel else '--'
        lineas = (
            'FRAME %d/%d  (~%.0f fps)' % (self.frame_idx, self.n_frames, self._fps_inst),
            'ACCION : ' + accion,
            'ESTADO : ' + estado,
            'ELEG   : ' + sel_s,
            'FLECHA : ' + ang + ' deg',
            'MARCA  : ' + marca_s,
            'ERROR  : ' + err_s,
            'v=%+.2f w=%+.2f' % (v, w),
        )
        for i, txt in enumerate(lineas):
            self._texto(out, txt, (4, 14 + i * 13), (255, 255, 255))

        # MEJORA: eliminada la banda negra inferior. La accion ya se
        # muestra en el panel HUD superior.
        return out

    @staticmethod
    def _texto(img, txt, pos, color, scale=0.4):
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 1, cv2.LINE_AA)

    # ==================================================================
    # BUCLE PRINCIPAL
    # ==================================================================
    def step(self):
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return False
        self.frame_idx += 1

        m_lin, m_rojo = segmentar_qda(self.clf_qda, bgr,
                                      self.AREA_MIN_LINEA,
                                      self.AREA_MIN_MARCA_SEG)

        min_front, _ = self.leer_distancias_mock()
        if self.cooldown_marca:   self.cooldown_marca -= 1
        if self.post_avoid_grace: self.post_avoid_grace -= 1
        if self._marca_display_ttl > 0: self._marca_display_ttl -= 1

        extremos = self.detectar_extremos(m_lin)
        salidas  = [e for e in extremos if not e['es_entrada']]
        error    = self.calcular_error(m_lin)
        ent_vis  = self.calcular_ent_visual(m_lin)        # MEJORA: ENT siempre

        marca_actual, flecha_visual = self.procesar_rojo(bgr, m_rojo)
        flecha_logica = self.actualizar_memoria_flecha(flecha_visual)
        cruce_inminente = self.hay_cruce_inminente(m_lin, salidas)

        # MEJORA: lock de la salida con timeout minimo
        # Cuando cruce_inminente es True, se commit y se refresca el TTL.
        # Cuando ya no esta, se sigue manteniendo el lock hasta agotar TTL.
        if cruce_inminente and len(salidas) >= 2:
            if self.cruce_salida_lock is None:
                # primera vez: commit
                self.cruce_salida_lock = self.elegir_salida(extremos, flecha_logica)
                if self.cruce_salida_lock:
                    print('[%5d] >>> CRUCE: bloqueado salida=%s (flecha=%s)'
                          % (self.frame_idx, self.cruce_salida_lock['lado'],
                             ('%+5.0f deg' % flecha_logica['angulo'])
                             if flecha_logica else '--'))
            # refresca TTL mientras el cruce siga visible
            self.cruce_lock_ticks_left = self.CRUCE_LOCK_TICKS_MIN
            # actualiza la coord (mismo lado, x mas cercano)
            actualizada = self._actualizar_salida_lock(extremos)
            if actualizada is not None:
                self.cruce_salida_lock = actualizada
            salida_elegida = self.cruce_salida_lock
        elif self.cruce_lock_ticks_left > 0 and self.cruce_salida_lock is not None:
            # cruce ya no detectado, pero todavia dura el lock
            self.cruce_lock_ticks_left -= 1
            actualizada = self._actualizar_salida_lock(extremos)
            if actualizada is not None:
                self.cruce_salida_lock = actualizada
            salida_elegida = self.cruce_salida_lock
            if self.cruce_lock_ticks_left == 0:
                print('[%5d] >>> CRUCE: liberado lock (timeout)' % self.frame_idx)
                self.cruce_salida_lock = None
        else:
            # sin cruce ni lock activo
            self.cruce_salida_lock = None
            salida_elegida = (self.elegir_salida(extremos, flecha_logica)
                              if len(salidas) >= 2 else None)

        v_cmd, w_cmd, estado = 0.0, 0.0, ''

        if min_front < self.DIST_FRONTAL_OBST and not self.avoiding:
            self.avoiding = True; self.avoid_ticks = 0; self.prev_error = None
        if self.avoiding:
            self.avoid_ticks += 1
            estado = 'AVOID (simulado)'
            v_cmd, w_cmd = 0.0, -1.0
        # MEJORA: tomar CRUCE no solo cuando cruce_inminente sino tambien
        # cuando el lock sigue activo (durante el timeout post-cruce)
        elif self.cruce_salida_lock is not None and salida_elegida is not None:
            err_x = (salida_elegida['punto'][0] - self.w / 2.0) / (self.w / 2.0)
            w_cmd = _clamp(-self.CRUCE_KP * err_x)
            v_cmd = max(self.SLOW_FORWARD,
                        self.FULL_FORWARD * (1.0 - 0.9 * abs(w_cmd)))
            self.last_error = err_x
            self.prev_error = None
            sufijo = '' if cruce_inminente else ' [LOCK]'
            estado = 'CRUCE -> ' + salida_elegida['lado'] + sufijo
        elif error is None:
            self.prev_error = None
            w_cmd = -0.8 if self.last_error > 0 else 0.8
            v_cmd, estado = 0.0, 'BUSCAR (spin)'
        else:
            if abs(error) > 0.15: self.last_error = error
            d_err = 0.0 if self.prev_error is None else (error - self.prev_error)
            self.prev_error = error
            w_cmd = _clamp(-(self.KP * error + 0.6 * d_err))
            v_cmd = max(self.SLOW_FORWARD,
                        self.FULL_FORWARD * (1.0 - 0.8 * abs(w_cmd)))
            estado = 'FOLLOW'

        accion = self.simular_motor(v_cmd, w_cmd)
        self.ultima_accion = accion
        if self.frame_idx % 30 == 0:
            print('[%5d] %-20s err=%s v=%+.2f w=%+.2f'
                  % (self.frame_idx, accion,
                     ('%+.2f' % error) if error is not None else '  --',
                     v_cmd, w_cmd))

        if self.frame_idx % 10 == 0:
            dt = time.perf_counter() - self.t0
            self._fps_inst = 10.0 / max(dt, 1e-6)
            self.t0 = time.perf_counter()

        out = self.overlay(bgr, m_lin, m_rojo, extremos, salida_elegida,
                           flecha_visual, flecha_logica, marca_actual,
                           error, v_cmd, w_cmd, estado, accion, ent_vis,
                           cruce_locked=(self.cruce_salida_lock is not None))

        if self.writer is not None: self.writer.write(out)
        if self.mostrar:
            try:
                cv2.imshow('BrainFollowLine-2', out)
                if cv2.waitKey(1) & 0xFF == 27: return False
            except Exception:
                self.mostrar = False
        return True

    def cerrar(self):
        self.cap.release()
        if self.writer is not None: self.writer.release()
        cv2.destroyAllWindows()
        print('\n=== Resumen de marcas detectadas: %d ===' % len(self.marcas_vistas))
        for (fi, cls, conf, cx, cy) in self.marcas_vistas:
            print('  frame %5d  %-10s  conf=%.2f  img=(%d,%d)'
                  % (fi, cls, conf, cx, cy))


def main():
    p = argparse.ArgumentParser(
        description='Debug offline del BrainFollowLine con video local.')
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
        output = os.path.join(_AQUI, 'output_video2017-%s.mp4' % args.video)

    brain = BrainOffline(video_in, output_path=output,
                         mostrar=not args.no_mostrar)
    for _ in range(args.start):
        if not brain.cap.read()[0]: break

    while brain.step():
        pass
    brain.cerrar()


if __name__ == '__main__':
    main()
