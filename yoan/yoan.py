from pyrobot.brain import Brain
import cv2
import numpy as np
import os
import joblib
import math
import collections
import time 

# ==============================================================================
# FUNCIONES DE VISIÓN
# ==============================================================================

def segmentar_escena(roi_imagen):
    desenfocado = cv2.GaussianBlur(roi_imagen, (5, 5), 0)
    hsv = cv2.cvtColor(desenfocado, cv2.COLOR_BGR2HSV)
    
    # --- CALIBRACIÓN LÍNEA AZUL (MUNDO REAL) ---
    rango_bajo = np.array([95, 120, 80])
    rango_alto = np.array([135, 255, 255])
    
    binaria = cv2.inRange(hsv, rango_bajo, rango_alto)
    kernel_cierre = np.ones((7, 7), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel_cierre)
    kernel_apertura = np.ones((5, 5), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_apertura)

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos

def segmentar_marca(roi_imagen):
    desenfocado = cv2.GaussianBlur(roi_imagen, (7, 7), 0)
    hsv = cv2.cvtColor(desenfocado, cv2.COLOR_BGR2HSV)
    
    # --- CALIBRACIÓN FLECHA ROJA (MUNDO REAL) ---
    S_MIN = 100
    V_MIN = 80
    
    mascara1 = cv2.inRange(hsv, np.array([0, S_MIN, V_MIN]), np.array([15, 255, 255]))
    mascara2 = cv2.inRange(hsv, np.array([165, S_MIN, V_MIN]), np.array([180, 255, 255]))
    mascara = cv2.bitwise_or(mascara1, mascara2)
    kernel = np.ones((9, 9), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos

def identificar_entradas_salidas(contorno, roi_shape):
    puntos_borde = []
    alto, ancho = roi_shape[:2]
    margen = 5
    
    # Distancia a 150 px para unificar la cinta en el mundo real
    distancia_minima = 150 
    
    for punto in contorno:
        x, y = punto[0]
        if x <= margen or x >= ancho - margen or y <= margen or y >= alto - margen:
            if not puntos_borde or all(np.linalg.norm(np.array([x, y]) - np.array(p)) > distancia_minima for p in puntos_borde):
                puntos_borde.append((x, y))

    puntos_borde.sort(key=lambda p: p[1], reverse=True)
    return puntos_borde

def clasificar_escena(puntos_borde, roi_shape):
    alto, ancho = roi_shape[:2]
    num_salidas = len(puntos_borde) - 1
    if num_salidas == 1:
        salida = puntos_borde[1]
        if salida[0] < ancho * 0.35:
            return "C.Izda"
        elif salida[0] > ancho * 0.65:
            return "C.Dcha"
        return "Linea"
    elif num_salidas == 2:
        return "Cruce(2)"
    elif num_salidas > 2:
        return "Cruce(3+)"
    return "Linea"

def procesar_marca(contornos_marca, modelo_knn, scaler, tipo_escena):
    if not contornos_marca:
        return None, None, None
    contorno_marca = max(contornos_marca, key=cv2.contourArea)
    if cv2.contourArea(contorno_marca) < 250 or len(contorno_marca) < 5:
        return None, None, None

    es_cruce = "Cruce" in tipo_escena

    if es_cruce:
        x, y, w, h = cv2.boundingRect(contorno_marca)
        gx, gy = x + w/2.0, y + h/2.0
        M = cv2.moments(contorno_marca)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            dx_masa = cx - gx
            dy_masa = cy - gy
            angulo_masa = math.degrees(math.atan2(dy_masa, dx_masa))
            if angulo_masa < 0:
                angulo_masa += 360

            [vx, vy, x0, y0] = cv2.fitLine(contorno_marca, cv2.DIST_L2, 0, 0.01, 0.01)
            angulo_linea = math.degrees(math.atan2(vy[0], vx[0]))
            if angulo_linea < 0:
                angulo_linea += 360
            diferencia = abs(angulo_linea - angulo_masa)
            if diferencia > 180:
                diferencia = 360 - diferencia
            if diferencia > 90:
                angulo_linea = (angulo_linea + 180) % 360
            return contorno_marca, angulo_linea, "Flecha"
        _, _, angulo = cv2.fitEllipse(contorno_marca)
        return contorno_marca, angulo, "Flecha"

    if modelo_knn is not None and scaler is not None:
        momentos = cv2.moments(contorno_marca)
        momentos_hu = cv2.HuMoments(momentos).flatten()
        for i in range(7):
            if momentos_hu[i] != 0:
                momentos_hu[i] = -1 * math.copysign(1.0, momentos_hu[i]) * math.log10(abs(momentos_hu[i]))
        try:
            momentos_escalados = scaler.transform([momentos_hu])
            clase = modelo_knn.predict(momentos_escalados)[0]
            return contorno_marca, None, clase
        except Exception:
            pass

    return contorno_marca, None, "Desconocida"

def categorizar_flecha(angulo):
    if angulo is None:
        return None
    if 225 <= angulo <= 315:
        return "Delante"
    elif 135 < angulo < 225:
        return "Izquierda"
    elif 45 <= angulo <= 135:
        return "Atras" 
    else:
        return "Derecha"

def seleccionar_salida_logica(puntos_borde, direccion, roi_shape):
    alto, ancho = roi_shape[:2]
    if len(puntos_borde) < 2: return None
    puntos_ordenados = sorted(puntos_borde, key=lambda p: p[1], reverse=True)
    entrada = puntos_ordenados[0]
    salidas = [p for p in puntos_borde if p != entrada]
    if not salidas: return None

    if direccion == "Delante":
        return min(salidas, key=lambda p: abs(p[0] - ancho//2)) 
    elif direccion == "Izquierda":
        return min(salidas, key=lambda p: p[0]) 
    elif direccion == "Derecha":
        return max(salidas, key=lambda p: p[0]) 
    else:
        return min(salidas, key=lambda p: p[1]) 

# ==============================================================================
# CLASE PRINCIPAL: BRAIN FOLLOW LINE + VISION
# ==============================================================================

class BrainFollowLine(Brain):
    Kp = -4.0
    Kd = -1.0
    Ki = 0.0

    def setup(self):
        # =====================================================================
        # MODO CÁMARA FÍSICA (Paso 1 de 3)
        # Descomenta las siguientes 3 líneas para encender la webcam
        # =====================================================================
        # self.capture = cv2.VideoCapture(0)
        # if not self.capture.isOpened():
        #     print("Error: No se pudo acceder a la cámara física.")

        self.esquivando = False
        self.primerGiro = False
        self.ultimaDire = 0
        self.historial_giros = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.intentos = 0
        self.error_previo = 0
        self.suma_errores = 0
        self.error_suavizado = 0.0

        self.memoria_cruce_error = None
        self.frames_memoria = 0
        self.angulo_recordado = None
        self.frames_angulo = 0
        self.cooldown_cruce = 0
        self.memoria_direcciones = collections.deque(maxlen=25)
        self.memoria_formas = collections.deque(maxlen=25)
        self.tiempo_anterior = time.time()

        directorio_script = os.path.dirname(os.path.abspath(__file__))
        ruta_modelo = os.path.join(directorio_script, 'modelo_marcas.pkl')
        ruta_scaler = os.path.join(directorio_script, 'scaler_marcas.pkl')
        self.modelo_knn = None
        self.scaler = None

        if os.path.exists(ruta_modelo) and os.path.exists(ruta_scaler):
            try:
                self.modelo_knn = joblib.load(ruta_modelo)
                self.scaler = joblib.load(ruta_scaler)
                print("Modelos de visión cargados correctamente.")
            except Exception as e:
                print("Error al cargar modelos:", e)

    def destroy(self):
        # =====================================================================
        # MODO CÁMARA FÍSICA (Paso 2 de 3)
        # Descomenta las siguientes 2 líneas para liberar la webcam al salir
        # =====================================================================
        # if hasattr(self, 'capture') and self.capture.isOpened():
        #     self.capture.release()
            
        cv2.destroyAllWindows()

    def step(self):
        # =====================================================================
        # SELECCIÓN DE ENTORNO (Paso 3 de 3)
        # =====================================================================
        
        # OPCIÓN A) Simulador STAGE (Comenta la línea de abajo para usar webcam)
        cv_image = self.robot.getImage()
        
        # OPCIÓN B) Cámara Física (Descomenta las 3 líneas de abajo para usar webcam)
        # ret, cv_image = self.capture.read()
        # if not ret:
        #     return
        
        # Comprobación de seguridad
        if cv_image is None or cv_image.size == 0:
            return
        # =====================================================================

        tiempo_actual = time.time()
        dt = tiempo_actual - self.tiempo_anterior
        fps = 1.0 / dt if dt > 0 else 0.0
        self.tiempo_anterior = tiempo_actual

        alto, ancho = cv_image.shape[:2]

        contornos_vision = segmentar_escena(cv_image)
        contornos_marca = segmentar_marca(cv_image)
        tipo_escena = "Buscando"
        error_cruces = None
        forma_estable = ""

        if contornos_vision:
            contorno_vision = max(contornos_vision, key=cv2.contourArea)
            
            # Dibujo de la vía en azul oscuro
            cv2.drawContours(cv_image, [contorno_vision], -1, (255, 0, 0), 3)

            puntos_borde_vision = identificar_entradas_salidas(contorno_vision, cv_image.shape)
            tipo_escena = clasificar_escena(puntos_borde_vision, cv_image.shape)
            
            # Puntos en rojo
            for p in puntos_borde_vision:
                cv2.circle(cv_image, p, 7, (0, 0, 255), -1) 

            contorno_marca, angulo_actual, clase_marca = procesar_marca(
                contornos_marca, self.modelo_knn, self.scaler, tipo_escena
            )

            if angulo_actual is not None:
                self.angulo_recordado = angulo_actual
                self.frames_angulo = 60
            elif self.frames_angulo > 0:
                self.frames_angulo -= 1
                if self.frames_angulo == 0:
                    self.angulo_recordado = None

            if contorno_marca is not None:
                cv2.drawContours(cv_image, [contorno_marca], -1, (0, 255, 255), 2)
                if clase_marca and "Cruce" not in tipo_escena:
                    self.memoria_formas.append(clase_marca)
                    forma_estable = max(set(self.memoria_formas), key=self.memoria_formas.count)
                    cv2.putText(cv_image, f"Forma: {forma_estable}", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            if "Cruce" in tipo_escena and self.angulo_recordado is not None and self.cooldown_cruce == 0:
                direccion_logica = categorizar_flecha(self.angulo_recordado)
                cv2.putText(cv_image, f"Intencion: {direccion_logica}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)
                salida_elegida = seleccionar_salida_logica(puntos_borde_vision, direccion_logica, cv_image.shape)
                
                if salida_elegida:
                    entrada_punto = sorted(puntos_borde_vision, key=lambda p: p[1], reverse=True)[0]
                    cv2.arrowedLine(cv_image, entrada_punto, salida_elegida, (255, 0, 255), 5, tipLength=0.15)

                    salidas = [p for p in puntos_borde_vision if p != entrada_punto]
                    if salidas:
                        trigger_y = max(p[1] for p in salidas) 
                    else:
                        trigger_y = entrada_punto[1]

                    umbral_y = int(alto * 0.60)
                    cv2.line(cv_image, (0, umbral_y), (ancho, umbral_y), (0, 165, 255), 2)
                    cv2.line(cv_image, (0, trigger_y), (ancho, trigger_y), (255, 0, 0), 1)

                    if trigger_y > umbral_y:
                        if direccion_logica == "Izquierda":
                            error_cruces = -0.6
                        elif direccion_logica == "Derecha":
                            error_cruces = 0.6
                        else:
                            cx = salida_elegida[0]
                            error_cruces = (cx - (ancho / 2.0)) / (ancho / 2.0)
                        
                        self.memoria_cruce_error = error_cruces
                        self.frames_memoria = 30
                        self.cooldown_cruce = 25
                        cv2.putText(cv_image, "EJECUTANDO SALIDA", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cx_entrada = entrada_punto[0]
                        error_cruces = (cx_entrada - (ancho / 2.0)) / (ancho / 2.0)
                        cv2.line(cv_image, (int(ancho/2), alto), entrada_punto, (0, 255, 0), 2)
                        cv2.putText(cv_image, "Aproximando...", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # -------------------------------------------------------------
        # SEGUIMIENTO BASE 100% VISIÓN HSV
        # -------------------------------------------------------------
        foundLine = False
        error_base = 0.0

        if contornos_vision:
            foundLine = True
            M = cv2.moments(contorno_vision)
            if M['m00'] > 0:
                cx_base = M['m10'] / M['m00']
                error_base = (cx_base - (ancho / 2.0)) / (ancho / 2.0)
                
        # --- RECUPERACIÓN DE MEMORIA ---
        if error_cruces is None and self.frames_memoria > 0:
            if foundLine and abs(error_base) < 0.50 and self.frames_memoria < 20:
                self.frames_memoria = 0
                cv2.putText(cv_image, "Memoria Anulada (Linea Cazada)", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                error_cruces = self.memoria_cruce_error
                self.frames_memoria -= 1
                cv2.putText(cv_image, f"Memoria Activa ({self.frames_memoria})", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        if self.cooldown_cruce > 0:
            self.cooldown_cruce -= 1
            cv2.putText(cv_image, f"Ignorando vision: {self.cooldown_cruce}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        if error_cruces is not None:
            error_objetivo = max(-0.4, min(0.4, error_cruces))
            kp_actual = -2.0
            foundLine = True
        else:
            error_objetivo = error_base
            kp_actual = self.Kp
            
        self.error_suavizado = (0.7 * self.error_suavizado) + (0.3 * error_objetivo)

        # -------------------------------------------------------------
        # MÁQUINA DE ESTADOS MOTRIZ
        # -------------------------------------------------------------
        front = min([s.distance() for s in self.robot.range["front"]])
        left = min([s.distance() for s in self.robot.range["left-front"]])
        right = min([s.distance() for s in self.robot.range["right-front"]])

        if front < 0.40 and not self.esquivando:
            self.esquivando = True

        if self.esquivando:
            if right < 0.5:
                self.robot.move(0.5, -0.2)
                self.primerGiro = True
            elif foundLine and self.primerGiro and right > 0.5:
                self.esquivando = False
                self.primerGiro = False
                self.historial_giros = [1.0, 1.0, 1.0, 1.0, 1.0]
                self.ultimaDire = 1.0
                self.error_previo = self.error_suavizado
            elif right > 0.5 and self.primerGiro:
                self.robot.move(0.0, -0.5)
            elif not self.primerGiro:
                self.robot.move(0.0, 0.5)
            else:
                self.robot.move(0.0, 0.0)

        elif foundLine:
            self.intentos = 0
            derivada = self.error_suavizado - self.error_previo
            self.suma_errores += self.error_suavizado
            giro = (kp_actual * self.error_suavizado) + (self.Kd * derivada) + (self.Ki * self.suma_errores)
            self.error_previo = self.error_suavizado
            aceleracion = max(0.1, 1.0 - abs(giro * 2.0))
            self.robot.move(aceleracion, giro)
            if abs(self.error_suavizado) > 0.4:
                self.historial_giros.pop(0)
                self.historial_giros.append(giro)
                self.ultimaDire = self.historial_giros[0]

        elif self.ultimaDire != 0 and self.intentos < 5:
            self.intentos += 1
            self.robot.move(0.0, self.ultimaDire)
        else:
            self.robot.move(0.0, 0.0)

        # Información visual (HUD)
        cv2.putText(cv_image, f"Via: {tipo_escena}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(cv_image, f"Err: {self.error_suavizado:.3f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(cv_image, f"Evadiendo: {self.esquivando}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        cv2.putText(cv_image, f"FPS: {fps:.1f}", (ancho - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Integracion Robot", cv_image)
        cv2.waitKey(1)

def INIT(engine):
    assert (engine.robot.requires("range-sensor") and engine.robot.requires("continuous-movement"))
    return BrainFollowLine('BrainFollowLine', engine)
