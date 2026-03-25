from pyrobot.brain import Brain
import cv2
from pyrobot.tools.followLineTools import findLineDeviation

class BrainFollowLine(Brain):
 
    SLOW_FORWARD = 0.1
    MED_FORWARD = 0.5

    def setup(self):
        self.last_error = 0 
        self.avoidance_steps = 0 

    def destroy(self):
        cv2.destroyAllWindows()

    def step(self):
        cv_image = self.robot.getImage()
        cv2.imshow("Stage Camera Image", cv_image)
        cv2.waitKey(1)

        imageGray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        foundLine, error = findLineDeviation(imageGray)

        front_distances = [self.robot.range[i].distance() for i in range(2, 6)]
        min_front_dist = min(front_distances)

        # 1. EVASIÓN DE OBSTÁCULOS (Transición Fluida)
        # Aumentamos un pelín la distancia de detección para tener espacio de hacer la curva
        if min_front_dist < 0.35:
            # FLUCTUACIÓN 1: En lugar de frenar a cero, avanzamos a 0.15 mientras giramos fuerte.
            # Esto dibuja una curva suave hacia afuera en lugar de un quiebre en "V".
            self.move(0.15, 1.0) 
            self.avoidance_steps = 15 
            self.last_error = 1.0 
            
        elif self.avoidance_steps > 0:
            # FLUCTUACIÓN 2: En lugar de ir recto, hacemos una curva muy abierta a la derecha.
            # Esto hace que el robot empiece a "abrazar" o envolver el obstáculo con fluidez.
            self.move(self.MED_FORWARD, -0.25) 
            self.avoidance_steps -= 1
            
        elif not foundLine:
            # 3. RECUPERACIÓN DE LÍNEA
            # Mantenemos un poco de avance (0.2) para que la reincorporación sea en arco.
            if self.last_error > 0:
                self.move(0.2, -0.8) # Arco suave a la derecha
            else:
                self.move(0.2, 0.8)  # Arco suave a la izquierda
                
        else:
            # 4. SEGUIMIENTO DE LÍNEA FLUIDO
            self.last_error = error 
            
            # FLUCTUACIÓN 3: Suavizamos el controlador proporcional.
            # Al multiplicar por 0.8 en vez de 1.0, el robot zigzaguea menos en las rectas.
            tv = -0.8 * error 
            
            # El frenado en curvas ahora es más gradual.
            fv = max(0.15, 0.8 - abs(error)) 
            
            self.move(fv, tv)

def INIT(engine):
    assert (engine.robot.requires("range-sensor") and
            engine.robot.requires("continuous-movement"))

    return BrainFollowLine('BrainFollowLine', engine)