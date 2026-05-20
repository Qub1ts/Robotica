from pyrobot.brain import Brain
import cv2
from pyrobot.tools.followLineTools import findLineDeviation

class BrainFollowLine(Brain):

    def setup(self):
        self.last_error = 0.0 
        self.avoiding = False 
        self.avoid_ticks = 0 

    def destroy(self):
        cv2.destroyAllWindows()

    def step(self):
        cv_image = self.robot.getImage()
        cv2.imshow("Stage Camera Image", cv_image)
        cv2.waitKey(1)

        imageGray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        foundLine, error = findLineDeviation(imageGray)

        # Sensores frontales (2, 3, 4, 5) para choque
        front_distances = [self.robot.range[i].distance() for i in range(2, 6)]
        min_front_dist = min(front_distances)

        # Sensores izquierdos (0, 1, 2) para seguir la pared 
        # (Como esquivamos por la derecha, la caja SIEMPRE estará a la izquierda)
        left_distances = [self.robot.range[i].distance() for i in range(0, 3)]
        min_left_dist = min(left_distances)

        # 1. INICIAR EVASIÓN (Siempre a la derecha)
        if min_front_dist < 0.35 and not self.avoiding:
            self.avoiding = True
            self.avoid_ticks = 0 
            
        # 2. MÁQUINA DE ESTADOS: ESQUIVAR POR LA DERECHA
        elif self.avoiding:
            self.avoid_ticks += 1 

            # Condición de salida (Ceguera temporal superada)
            if foundLine and min_front_dist > 0.4 and self.avoid_ticks > 40:
                self.avoiding = False
                
                # CLAVE: Al terminar de rodear la caja, cruzará la línea.
                # Inyectamos un error positivo (1.0) para OBLIGARLO a girar a la 
                # derecha cuando pierda la línea, alineándose con la pista.
                self.last_error = 1.0 
                return 

            # CASO A: Frente bloqueado -> Girar a la DERECHA para esquivar
            if min_front_dist < 0.35:
                self.move(0.0, -1.0)
            
            # CASO B: Esquina detectada -> Girar a la IZQUIERDA para envolver la caja
            elif min_left_dist > 0.5:
                self.move(0.0, 1.0)
            
            # CASO C: Bordeando la pared izquierda (Controlador Proporcional Invertido)
            else:
                dist_objetivo = 0.30 
                error_pared = dist_objetivo - min_left_dist
                
                # Si está muy cerca de la pared izquierda, debe girar a la derecha (negativo)
                tv_avoid = -2.5 * error_pared 
                tv_avoid = max(-1.0, min(1.0, tv_avoid)) 
                
                self.move(0.15, tv_avoid)
            
        # 3. RECUPERACIÓN DE LÍNEA ANTI-RETROCESO
        elif not foundLine:
            if self.last_error > 0:
                self.move(0.0, -0.8) # Gira a la derecha
            else:
                self.move(0.0, 0.8)  # Gira a la izquierda
                
        # 4. SEGUIMIENTO DE LÍNEA FLUIDO (Normal)
        else:
            self.avoiding = False 
            
            # Ignoramos lecturas de error = ~0.0 para no borrar la memoria
            # del giro a la derecha que inyectamos al salir de la evasión.
            if abs(error) > 0.15:
                self.last_error = error 
            
            tv = -1.2 * error 
            fv = max(0.05, 0.4 - abs(error)) 
            
            self.move(fv, tv)

def INIT(engine):
    assert (engine.robot.requires("range-sensor") and
            engine.robot.requires("continuous-movement"))

    return BrainFollowLine('BrainFollowLine', engine) 
