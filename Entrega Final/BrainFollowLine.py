"""BrainFollowLine - Entrega Final Robotica y Percepcion Computacional.

Cerebro pyrobot que integra control (Practica 1) y percepcion (Practica 2)
para seguir una linea azul con cruces, marcas y obstaculos.

Modulos del pipeline (en cada step):
    1. Segmentacion QDA (linea azul / marca roja / fondo)
    2. Deteccion de extremos en bordes de la mascara + punto de bifurcacion
    3. Clasificacion LDA de marcas (man / stairs / telephone / woman)
    4. Deteccion de flecha (PCA + asimetria del extremo mas ancho)
    5. Maquina de estados de control:
         AVOID  -> esquiva el obstaculo
         CRUCE  -> usa la flecha cacheada para elegir ramal
         FOLLOW -> seguimiento PD sobre la franja inferior
         BUSCAR -> giro en sitio hacia el ultimo sentido conocido

Archivos necesarios en el mismo directorio:
    imagen_original.png      -> imagen de referencia para entrenar QDA
    imagen_marcada.png       -> mascara pintada (R=marca, G=fondo, B=linea)
    marcas-capturasStage/    -> 28 PNG de las 4 clases (man/stairs/telephone/woman)
"""

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
# RUTAS Y CONSTANTES PRECOMPUTADAS (modulo)
# ======================================================================
_AQUI = os.path.dirname(os.path.abspath(__file__))
RUTA_IMG_ORIGINAL   = os.path.join(_AQUI, 'imagen_original.png')
RUTA_IMG_MARCADA    = os.path.join(_AQUI, 'imagen_marcada.png')
RUTA_DATASET_MARCAS = os.path.join(_AQUI, 'marcas-capturasStage')

CLASES_MARCAS = ('man', 'stairs', 'telephone', 'woman')

# Rangos HSV del rojo "marca/flecha" (se envuelve en H=0 -> dos tramos).
# Se crean una sola vez aqui para no reservar arrays en cada frame.
_HSV_ROJO_LO1 = np.array([0,   100,  70], dtype=np.uint8)
_HSV_ROJO_HI1 = np.array([12,  255, 255], dtype=np.uint8)
_HSV_ROJO_LO2 = np.array([165, 100,  70], dtype=np.uint8)
_HSV_ROJO_HI2 = np.array([179, 255, 255], dtype=np.uint8)

# Kernel morfologico unico (elipse 3x3) para apertura/cierre.
_KERNEL_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Regex para parsear nombres del dataset: 'man-3.png', 'telephone_5.png', etc.
_RX_MARCA = re.compile(r'^([a-z]+)[-_]\d+\.png$', re.IGNORECASE)


# ======================================================================
# VISION: SEGMENTACION QDA Y CLASIFICADOR LDA
# ======================================================================
def _features_pixel(rgb):
    """Vector de 9 features por pixel para el QDA de segmentacion.

    Features: HSV normalizado (3) + a/b de Lab (2) + RGB normalizado (3)
    + L de Lab (1). Lab a/b es muy discriminante para azul/rojo.

    Args:
        rgb: array (N,3) o (H,W,3) en uint8 RGB.
    Returns:
        array (N, 9) en float32.
    """
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
    """Entrena el QDA de 3 clases (fondo / marca / linea).

    Lee imagen_original.png + imagen_marcada.png y construye el dataset
    usando los pixeles con colores puros como ground truth:
        R (255,0,0)  -> marca
        G (0,255,0)  -> fondo
        B (0,0,255)  -> linea
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


def segmentar_qda(clf, bgr):
    """Aplica el QDA a un frame BGR y devuelve mascaras (linea, marca).

    Postprocesa con apertura/cierre morfologico para limpiar ruido.
    """
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
    """Aisla la silueta roja principal y calcula su descriptor de forma.

    Descriptor 11-D = 7 log-Hu moments (invariantes a t/r/s) + 4 ratios
    geometricos (aspect, extent, solidity, circularity).

    Returns:
        (descriptor, bbox) o None si no hay blob rojo >= area_min.
    """
    # 1) Mascara roja con cierre/apertura.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (cv2.inRange(hsv, _HSV_ROJO_LO1, _HSV_ROJO_HI1) |
         cv2.inRange(hsv, _HSV_ROJO_LO2, _HSV_ROJO_HI2))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  _KERNEL_3, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _KERNEL_3, iterations=2)

    # 2) Componente conexa mas grande (descarta ruido).
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

    # 3) Descriptor 11-D = log-Hu (7) + ratios (4).
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
                bh / bw if bw else 0.0,                # aspect
                area / (bw * bh) if (bw and bh) else 0.0,  # extent
                area / hull_area,                      # solidity
                4 * math.pi * area / (perim * perim),  # circularity
            ], dtype=np.float32)

    return np.concatenate([log_hu, ratios]).astype(np.float32), (x, y, w, h)


def entrenar_lda_marcas():
    """Entrena un LDA con las 28 imagenes de marcas-capturasStage/.

    Returns:
        (clf_lda, rangos_por_clase). rangos_por_clase es un dict
        {idx_clase: array(4,2)} con (min, max) +- 0.05 de los 4 ratios
        por clase. Sirve para rechazar siluetas que no se parezcan a las
        entrenadas (filtro de outliers anti-flecha).
    """
    X, y, rangos_raw = [], [], {c: [] for c in range(len(CLASES_MARCAS))}
    for f in sorted(glob.glob(os.path.join(RUTA_DATASET_MARCAS, '*.png'))):
        m = _RX_MARCA.match(os.path.basename(f))
        if not (m and m.group(1).lower() in CLASES_MARCAS):
            continue
        out = _silueta_y_descriptor(cv2.imread(f))
        if not out:
            continue
        desc, _ = out
        clase = CLASES_MARCAS.index(m.group(1).lower())
        X.append(desc)
        y.append(clase)
        rangos_raw[clase].append(desc[7:11])

    clf = LinearDiscriminantAnalysis(solver='svd').fit(np.stack(X), np.array(y))
    rangos = {k: np.column_stack([np.stack(l).min(0) - 0.05,
                                   np.stack(l).max(0) + 0.05])
              for k, l in rangos_raw.items() if l}
    return clf, rangos


def predecir_marca(bgr, clf, rangos, area_min=300, umbral_conf=0.55):
    """Clasifica la marca roja del frame; None si no es ninguna conocida.

    Filtros: umbral de area_min, umbral de confianza LDA, escudo de borde
    inferior (los blobs que tocan el suelo son flechas, no marcas) y
    filtro de rangos por clase (rechaza outliers como las propias flechas).

    Returns:
        (nombre_clase, confianza, bbox) o None.
    """
    out = _silueta_y_descriptor(bgr, area_min)
    if not out:
        return None
    feat, bbox = out

    # Escudo de borde: si el blob toca el borde inferior es probablemente
    # la flecha del cruce, no una marca lateral.
    if (bbox[1] + bbox[3]) >= (bgr.shape[0] - 5):
        return None

    probs = clf.predict_proba(feat.reshape(1, -1))[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    if conf < umbral_conf:
        return None

    # Filtro estricto: los 4 ratios deben caer dentro del rango entrenado
    # de la clase predicha.
    rng = rangos.get(pred) if rangos else None
    if rng is not None:
        r = feat[7:11]
        if np.any(r < rng[:, 0]) or np.any(r > rng[:, 1]):
            return None

    return CLASES_MARCAS[pred], conf, bbox


# ======================================================================
# CEREBRO PRINCIPAL
# ======================================================================
class BrainFollowLine(Brain):
    """Brain pyrobot que sigue linea azul + esquiva + interpreta marcas/flechas.

    Maquina de estados (orden de prioridad):

        AVOID  - Frente bloqueado: rodea el obstaculo por la izquierda.
                 Sale solo cuando la linea es visible y CENTRADA (|err|<0.5)
                 -> reintegracion tangencial. Setea last_error=+1 y activa
                 una ventana de "gracia" (POST_AVOID_GRACE) en la que el
                 estado CRUCE queda desactivado para no confundirse con la
                 geometria caotica del post-evasion.

        CRUCE  - Hay >=2 salidas y la bifurcacion esta cerca del robot
                 (junction_y en el cuarto inferior, o salida lateral muy
                 abajo). El ramal se elige por similitud angular con la
                 flecha cacheada (o por defecto la mas centrada).

        FOLLOW - Hay linea en la franja inferior. PD estilo Practica 1:
                 w = -KP * error;  v = max(SLOW, FULL - |error|).
                 No actualiza last_error si |error|<0.15 (anti-retroceso).

        BUSCAR - No hay linea. Gira EN SITIO (v=0) hacia el ultimo lado
                 conocido. v=0 evita que el robot avance y cruce la linea
                 perpendicularmente.
    """

    # ------------------------------------------------------------------
    # CONSTANTES DE CONTROL
    # ------------------------------------------------------------------
    # Velocidades (formula PD de Practica 1: v = max(SLOW, FULL-|err|))
    SLOW_FORWARD, FULL_FORWARD = 0.05, 0.40
    KP        = 1.2     # ganancia P del seguimiento de linea
    CRUCE_KP  = 1.6     # ganancia P del giro en cruce (tanh)

    # ------------------------------------------------------------------
    # CONSTANTES DE PERCEPCION
    # ------------------------------------------------------------------
    # Franja inferior usada para calcular el error de seguimiento (px)
    FRANJA_ERROR     = 40
    # Banda perimetral para detectar extremos en los bordes (px)
    BANDA_BORDE      = 4
    # Longitud minima y gap de fusion entre segmentos del mismo borde
    MIN_SEGMENTO     = 5
    FUSION_GAP       = 8
    # Cruce inminente: bifurcacion >= H * frac (cuarto inferior por defecto)
    JUNCTION_Y_FRAC  = 0.75
    JUNCTION_MIN_RUN = 3
    # Salida lateral 'izq'/'der' dispara cruce solo si esta MUY abajo
    SIDE_EXIT_Y_FRAC = 0.80

    # Filtros de la flecha (PCA + asimetria del extremo ancho)
    AREA_MIN_FLECHA  = 120
    FLECHA_CIRC_MAX  = 0.35
    FLECHA_ELONG_MIN = 3.0
    FLECHA_ASIM_MIN  = 0.40

    # Marca: minima area + cooldown entre reportes + TTL de cache flecha
    MARCA_AREA_MIN = 300
    MARCA_COOLDOWN = 25
    ARROW_TTL      = 300   # ~30 s @ 10 fps -> sobrevive la aproximacion al cruce

    # ------------------------------------------------------------------
    # CONSTANTES DE EVASION (igualadas a Practica 1)
    # ------------------------------------------------------------------
    DIST_FRONTAL_OBST   = 0.40   # distancia frontal que dispara la evasion
    DIST_FRENTE_LIBRE   = 0.40   # frente despejado para poder salir
    DIST_OBJETIVO_PARED = 0.30   # objetivo del control proporcional invertido
    AVOID_TICKS_MIN     = 40     # ticks minimos de evasion (anti-rebote)
    AVOID_TICKS_MAX     = 250    # tope absoluto -> salida forzada
    AVOID_EXIT_ERR_MAX  = 0.50   # solo sale si la linea esta centrada
    AVOID_FLAG          = 1.0    # centinela puesto en last_error tras AVOID
    POST_AVOID_GRACE    = 60     # frames sin CRUCE tras AVOID (~6 s)

    # ==================================================================
    # CICLO DE VIDA
    # ==================================================================
    def setup(self):
        """Carga y entrena los modelos una vez al iniciar el simulador."""
        print('Cargando modelos QDA y LDA...')
        self.clf_qda = entrenar_qda_linea()
        self.clf_lda, self.rangos_marcas = entrenar_lda_marcas()
        print('Listo. Todo preparado para la entrega final.')

        # --- Estado del controlador ---
        self.prev_error       = None   # error del frame anterior (para D si se usa)
        self.last_error       = 0.0    # ultimo error significativo (sesgo de busqueda)

        # --- Maquina de estados de evasion ---
        self.avoiding         = False
        self.avoid_ticks      = 0
        self.post_avoid_grace = 0      # cuenta atras de la ventana sin CRUCE

        # --- Cache de la flecha (sobrevive ARROW_TTL frames) ---
        self.arrow_cache      = None
        self.arrow_ttl_left   = 0

        # --- Reporte de marcas con cooldown ---
        self.last_marca       = None
        self.cooldown_marca   = 0
        self.marcas_vistas    = []

    def destroy(self):
        """Imprime resumen de marcas y cierra las ventanas de OpenCV."""
        if self.marcas_vistas:
            print('=== Marcas detectadas durante la ejecucion ===')
            for i, (clase, conf, x, y) in enumerate(self.marcas_vistas, 1):
                print('  %d. %s (conf=%.2f) en img=(%d,%d)'
                      % (i, clase, conf, x, y))
        cv2.destroyAllWindows()

    # ==================================================================
    # PERCEPCION DE LA LINEA
    # ==================================================================
    def detectar_extremos(self, m_lin):
        """Devuelve los puntos donde la linea toca cada borde del frame.

        Para cada borde (abajo/arriba/izq/der) busca segmentos de longitud
        >= MIN_SEGMENTO, los fusiona si estan a menos de FUSION_GAP, y
        crea un dict con su posicion, lado y angulo desde el centro.

        El extremo 'abajo' se marca como entrada (es_entrada=True).
        """
        h, w = m_lin.shape
        b, cx, cy = self.BANDA_BORDE, w / 2.0, h / 2.0

        # (nombre, perfil 1D del borde, eje variable, coord fija)
        bordes = (
            ('abajo',     m_lin[-b:, :].any(0), 'x', h - 1),
            ('arriba',    m_lin[:b,  :].any(0), 'x', 0),
            ('izquierda', m_lin[:, :b].any(1),  'y', 0),
            ('derecha',   m_lin[:, -b:].any(1), 'y', w - 1),
        )
        out = []
        for lado, perfil, eje, fija in bordes:
            # Busca los "runs" de True en el perfil binario.
            d = np.diff(np.concatenate([[0], perfil.astype(np.int8), [0]]))
            ini = np.where(d == 1)[0]
            fin = np.where(d == -1)[0]
            segs = [(int(a), int(z)) for a, z in zip(ini, fin)
                    if (z - a) >= self.MIN_SEGMENTO]
            if not segs:
                continue

            # Fusiona segmentos cercanos para no contar la misma rama dos veces.
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
                    'lado'      : lado,
                    'punto'     : (int(px), int(py)),
                    'es_entrada': lado == 'abajo',
                    'angulo'    : math.degrees(math.atan2(cy - py, px - cx)),
                })
        return out

    def junction_y(self, m_lin):
        """Fila mas baja (en pixels) donde la linea tiene >=2 segmentos.

        Sirve para saber si la bifurcacion del cruce esta cerca del robot
        (y alto -> cerca, y bajo -> lejos). Devuelve None si no se bifurca.
        """
        min_run = self.JUNCTION_MIN_RUN
        for y in range(m_lin.shape[0] - 1, -1, -1):
            d = np.diff(np.concatenate([[0], m_lin[y].astype(np.uint8), [0]]))
            ini = np.where(d == 1)[0]
            fin = np.where(d == -1)[0]
            n_runs = 0
            for s, e in zip(ini, fin):
                if (e - s) >= min_run:
                    n_runs += 1
                    if n_runs >= 2:
                        return int(y)
        return None

    # ==================================================================
    # PERCEPCION DE LA FLECHA (PCA del blob rojo)
    # ==================================================================
    def detectar_flecha(self, m_rojo):
        """Detecta una flecha en la mascara roja y devuelve su orientacion.

        Algoritmo:
          1. Contorno principal y filtro por area + circularidad.
          2. PCA sobre los pixeles del contorno: si la elongacion es baja
             no es una flecha.
          3. La cabeza es el extremo (positivo o negativo del eje principal)
             cuyo ancho perpendicular es MAYOR -> el triangulo de la flecha.
          4. Rechaza siluetas simetricas (ambos extremos igual de anchos).

        Returns:
            dict con angulo (en grados, convencion imagen: 0=der, 90=arr,
            +/-180=izq, -90=abj), centro, contorno y punta. None si no
            cumple los filtros.
        """
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

        # Circularidad: flecha alargada => circ baja
        perim = cv2.arcLength(cnt, True)
        if (4.0 * math.pi * area / max(perim * perim, 1e-6)) > self.FLECHA_CIRC_MAX:
            return None

        # PCA sobre los pixeles rellenos del contorno
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
        long_ax = pmax - pmin
        if long_ax < 5:
            return None

        # Ancho perpendicular en la franja extrema (25% mas alejado del centro)
        franja = 0.25 * long_ax
        pp_pos = proy_p[proy_a > pmax - franja]
        pp_neg = proy_p[proy_a < pmin + franja]
        s_pos = float(pp_pos.max() - pp_pos.min()) if len(pp_pos) >= 5 else 0.0
        s_neg = float(pp_neg.max() - pp_neg.min()) if len(pp_neg) >= 5 else 0.0
        max_s = max(s_pos, s_neg)
        # Rechaza siluetas simetricas (no son flechas reales)
        if max_s < 1e-3 or abs(s_pos - s_neg) / max_s < self.FLECHA_ASIM_MIN:
            return None

        # Cabeza = extremo MAS ancho (la base del triangulo)
        idx = int(np.argmax(proy_a)) if s_pos > s_neg else int(np.argmin(proy_a))
        px, py = float(pts[idx, 0]), float(pts[idx, 1])
        return {
            'angulo'  : math.degrees(math.atan2(cy - py, px - cx)),
            'centro'  : (cx, cy),
            'contorno': cnt,
            'punta'   : (px, py),
        }

    def elegir_salida(self, extremos, flecha):
        """Elige la salida (extremo no-entrada) cuya direccion mejor
        coincide con el angulo de la flecha (o con 'arriba'=90 si no hay).

        Sin filtros adicionales: respeta a rajatabla la direccion indicada
        por la flecha, incluso si apunta a un lateral inferior.
        """
        salidas = [e for e in extremos if not e['es_entrada']]
        if not salidas:
            return None
        ref = flecha['angulo'] if flecha else 90.0
        # Distancia angular minima (envolvente +/-180)
        return min(salidas,
                   key=lambda e: abs((e['angulo'] - ref + 180) % 360 - 180))

    # ==================================================================
    # VISUALIZACION (overlay sobre la ventana de camara)
    # ==================================================================
    def overlay(self, bgr, m_lin, m_rojo, extremos, sel, flecha, marca,
                err, v, w, estado):
        """Dibuja sobre el frame las mascaras, extremos, flecha, marca y
        un panel de texto con el estado actual. Devuelve el BGR anotado.
        """
        h, wi = bgr.shape[:2]

        # 1) Mascaras translucidas (azul = linea, rojo = marca)
        ov = bgr.copy()
        ov[m_lin]  = (255, 0, 0)
        ov[m_rojo] = (0, 0, 255)
        out = cv2.addWeighted(ov, 0.3, bgr, 0.7, 0)

        # 2) Linea separadora de la franja inferior (donde se mide el error)
        cv2.line(out, (0, h - self.FRANJA_ERROR),
                 (wi, h - self.FRANJA_ERROR), (180, 180, 180), 1)

        # 3) Extremos: ENT verde, ELEG cyan, S amarilla
        for e in extremos:
            es_eleg = sel is not None and e['punto'] == sel['punto']
            if e['es_entrada']:
                col, etq, gr = (0, 255, 0),  'ENT',  2
            elif es_eleg:
                col, etq, gr = (255, 255, 0), 'ELEG', 3
            else:
                col, etq, gr = (0, 255, 255), 'S',    2
            cv2.circle(out, e['punto'], 8, col, gr)
            # Desplaza el texto fuera del borde
            tx, ty = e['punto']
            if   e['lado'] == 'arriba':    ty += 16
            elif e['lado'] == 'abajo':     ty -= 6
            elif e['lado'] == 'izquierda': tx += 12
            else:                          tx -= 36
            self._texto(out, etq, (tx, ty), col)

        # 4) Flecha: contorno + vector centro -> punta
        if flecha:
            cv2.drawContours(out, [flecha['contorno']], -1, (255, 255, 0), 1)
            cf = tuple(map(int, flecha['centro']))
            pf = tuple(map(int, flecha['punta']))
            cv2.arrowedLine(out, cf, pf, (0, 255, 255), 2, tipLength=0.30)

        # 5) Marca: bbox + nombre de la clase
        if marca:
            x0, y0, bw, bh = marca[2]
            cv2.rectangle(out, (x0, y0), (x0 + bw, y0 + bh), (0, 200, 255), 2)
            self._texto(out, marca[0], (x0, max(12, y0 - 4)),
                        (0, 200, 255), scale=0.45)

        # 6) Panel de texto en la esquina superior izquierda
        ang_str = '%+5.0f deg' % flecha['angulo'] if flecha else '--'
        err_str = '%+.2f' % err if err is not None else '--'
        lineas = (
            'Estado: ' + estado,
            'Eleg: ' + (sel['lado'] if sel else '--'),
            'Flecha (mem): ' + ang_str,
            'Error: ' + err_str,
            'v=%+.2f w=%+.2f' % (v, w),
        )
        for i, txt in enumerate(lineas):
            self._texto(out, txt, (4, 14 + i * 13), (255, 255, 255))
        return out

    @staticmethod
    def _texto(img, txt, pos, color, scale=0.4):
        """Texto con halo negro para legibilidad sobre cualquier fondo."""
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color,     1, cv2.LINE_AA)

    # ==================================================================
    # BUCLE PRINCIPAL
    # ==================================================================
    def step(self):
        """Un ciclo del controlador: percepcion + decision + accion."""
        bgr = self.robot.getImage()
        if bgr is None:
            self.move(0.0, 0.0)
            return

        h_img, w_img = bgr.shape[:2]
        cx_img = w_img / 2.0

        # ---- 1) SEGMENTACION + SONAR ----
        m_lin, m_rojo = segmentar_qda(self.clf_qda, bgr)
        try:
            min_front = min(self.robot.range[i].distance() for i in range(2, 6))
            min_left  = min(self.robot.range[i].distance() for i in range(0, 3))
        except Exception:
            min_front, min_left = 99.0, 99.0

        # Decremento de cooldowns
        if self.cooldown_marca:
            self.cooldown_marca -= 1
        if self.post_avoid_grace:
            self.post_avoid_grace -= 1

        # ---- 2) EXTREMOS + ERROR ----
        extremos = self.detectar_extremos(m_lin)
        salidas  = [e for e in extremos if not e['es_entrada']]

        # Error = centroide horizontal de la linea en la franja inferior.
        # Se usan momentos por eficiencia (mas rapido que np.where + mean).
        franja = m_lin[-self.FRANJA_ERROR:, :].astype(np.uint8)
        M = cv2.moments(franja, binaryImage=True)
        error = (float((M['m10'] / M['m00'] - cx_img) / cx_img)
                 if M['m00'] >= 1 else None)

        # ---- 3) PERCEPCION ROJA: primero MARCA, luego FLECHA si no es marca ----
        marca_actual  = None
        flecha_visual = None
        if m_rojo.any():
            marca_actual = predecir_marca(bgr, self.clf_lda,
                                          self.rangos_marcas,
                                          self.MARCA_AREA_MIN)
            if marca_actual is None:
                # La silueta roja NO es ninguna marca conocida -> puede ser flecha.
                flecha_visual = self.detectar_flecha(m_rojo)
            elif (self.cooldown_marca == 0
                  and marca_actual[0] != self.last_marca):
                # Marca nueva: reportar + activar cooldown.
                clase, conf, bbox = marca_actual
                cxm = bbox[0] + bbox[2] // 2
                cym = bbox[1] + bbox[3] // 2
                print('>>> MARCA: %s (%.2f) en img=(%d,%d)'
                      % (clase, conf, cxm, cym))
                self.marcas_vistas.append((clase, conf, cxm, cym))
                self.last_marca     = clase
                self.cooldown_marca = self.MARCA_COOLDOWN

        # Cache de flecha (sobrevive ARROW_TTL frames -> persiste durante la
        # aproximacion aunque la flecha salga del campo de vision).
        if flecha_visual:
            self.arrow_cache    = flecha_visual
            self.arrow_ttl_left = self.ARROW_TTL
        elif self.arrow_ttl_left > 0:
            self.arrow_ttl_left -= 1
        flecha_logica = self.arrow_cache if self.arrow_ttl_left > 0 else None

        # ---- 4) DETECCION DE CRUCE INMINENTE ----
        # Solo evaluamos CRUCE si NO estamos en la ventana de gracia
        # post-evasion (para evitar elegir el extremo "hacia atras").
        cruce_inminente = False
        if self.post_avoid_grace == 0 and len(salidas) >= 2:
            jy = self.junction_y(m_lin)
            if jy is not None and jy >= h_img * self.JUNCTION_Y_FRAC:
                cruce_inminente = True
            else:
                # Tambien dispara si una salida lateral esta MUY abajo
                # en la imagen (T-cruce cerca del robot).
                lim_y = h_img * self.SIDE_EXIT_Y_FRAC
                for s in salidas:
                    if s['lado'] in ('izquierda', 'derecha') \
                            and s['punto'][1] >= lim_y:
                        cruce_inminente = True
                        break

        salida_elegida = (self.elegir_salida(extremos, flecha_logica)
                          if len(salidas) >= 2 else None)

        # ============================================================
        # 5) MAQUINA DE ESTADOS DE CONTROL
        # ============================================================
        v_cmd, w_cmd, estado = 0.0, 0.0, ''

        # ---- a) EVASION DE OBSTACULO ----
        if min_front < self.DIST_FRONTAL_OBST and not self.avoiding:
            self.avoiding    = True
            self.avoid_ticks = 0
            self.prev_error  = None

        if self.avoiding:
            self.avoid_ticks += 1
            # Salida tangencial: la linea debe estar visible Y CENTRADA
            # (|err|<0.5) o haber pasado el tope absoluto de ticks.
            linea_centrada = (error is not None
                              and abs(error) < self.AVOID_EXIT_ERR_MAX)
            timeout = self.avoid_ticks > self.AVOID_TICKS_MAX
            if (self.avoid_ticks > self.AVOID_TICKS_MIN
                    and min_front > self.DIST_FRENTE_LIBRE
                    and (linea_centrada or timeout)):
                # Cierre limpio: sin movimiento este frame, bandera de
                # anti-retroceso + ventana sin CRUCE
                self.avoiding         = False
                self.last_error       = self.AVOID_FLAG
                self.prev_error       = None
                self.post_avoid_grace = self.POST_AVOID_GRACE
                self.move(0.0, 0.0)
                out = self.overlay(bgr, m_lin, m_rojo, extremos,
                                   salida_elegida, flecha_visual,
                                   marca_actual, error, 0.0, 0.0,
                                   'POST-AVOID (grace=%d)'
                                   % self.POST_AVOID_GRACE)
                cv2.imshow('Stage Camera Image', out)
                cv2.waitKey(1)
                return

            # Subestados (iguales a Practica 1: rodear por la izquierda)
            if min_front < self.DIST_FRONTAL_OBST:
                # Frente bloqueado -> giro brusco a la derecha
                v_cmd, w_cmd, estado = 0.0, -1.0, 'AVOID-FRONT'
            elif min_left > 0.5:
                # Esquina detectada -> envolver el obstaculo (gira en sitio)
                v_cmd, w_cmd, estado = 0.0, 1.0, 'AVOID-CORNER'
            else:
                # Bordear la pared izquierda: P invertido sobre dist_objetivo
                w_cmd = max(-1.0, min(1.0,
                          -2.5 * (self.DIST_OBJETIVO_PARED - min_left)))
                v_cmd, estado = 0.15, 'AVOID-WALL'

        # ---- b) CRUCE INMINENTE: tomar el ramal indicado por la flecha ----
        elif cruce_inminente and salida_elegida is not None:
            # omega proporcional (tanh suaviza la respuesta cerca del centro)
            err_x = (salida_elegida['punto'][0] - cx_img) / cx_img
            w_cmd = max(-1.0, min(1.0, -math.tanh(self.CRUCE_KP * err_x)))
            v_cmd = max(0.08, self.FULL_FORWARD - abs(err_x))
            self.last_error = err_x
            self.prev_error = None
            estado = 'CRUCE -> ' + salida_elegida['lado']

        # ---- c) SIN LINEA: spin EN SITIO hacia el ultimo lado conocido ----
        elif error is None:
            self.prev_error = None
            w_cmd = -0.8 if self.last_error > 0 else 0.8
            v_cmd, estado = 0.0, 'BUSCAR (spin)'

        # ---- d) SEGUIMIENTO PD ----
        else:
            # Anti-retroceso: si la linea esta cerca del centro no borramos
            # el sesgo de busqueda (importante para reintegrar tras AVOID).
            if abs(error) > 0.15:
                self.last_error = error
            # Formula PD estilo Practica 1: w = -KP*err, v = max(SLOW, FULL-|err|)
            w_cmd = max(-1.0, min(1.0, -self.KP * error))
            v_cmd = max(self.SLOW_FORWARD, self.FULL_FORWARD - abs(error))
            estado = ('FOLLOW (grace=%d)' % self.post_avoid_grace
                      if self.post_avoid_grace else 'FOLLOW')

        # ---- 6) ACCION + OVERLAY ----
        self.move(v_cmd, w_cmd)
        out = self.overlay(bgr, m_lin, m_rojo, extremos, salida_elegida,
                           flecha_visual, marca_actual,
                           error, v_cmd, w_cmd, estado)
        cv2.imshow('Stage Camera Image', out)
        cv2.waitKey(1)


# ======================================================================
# PUNTO DE ENTRADA REQUERIDO POR PYROBOT
# ======================================================================
def INIT(engine):
    """Constructor del brain que pyrobot llama al cargar."""
    assert (engine.robot.requires('range-sensor') and
            engine.robot.requires('continuous-movement'))
    return BrainFollowLine('BrainFollowLine', engine)
