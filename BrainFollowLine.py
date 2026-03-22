from pyrobot.brain import Brain
import cv2
from pyrobot.tools.followLineTools import findLineDeviation

class BrainFollowLine(Brain):
 
  NO_FORWARD = 0
  SLOW_FORWARD = 0.1
  MED_FORWARD = 0.5
  FULL_FORWARD = 1.0

  NO_TURN = 0
  MED_LEFT = 0.5
  HARD_LEFT = 1.0
  MED_RIGHT = -0.5
  HARD_RIGHT = -1.0

  NO_ERROR = 0

  def setup(self):
    pass

    def destroy(self):
        cv2.destroyAllWindows()

  def step(self):
    cv_image = self.robot.getImage()

    # Mostrar la imagen de la cámara (ideal para tus capturas de pantalla del reporte)
    cv2.imshow("Stage Camera Image", cv_image)
    cv2.waitKey(1)

    # Convertir a escala de grises y buscar la línea
    imageGray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    foundLine, error = findLineDeviation(imageGray)

    # 1. EVASIÓN DE OBSTÁCULOS
    # Leemos los sonares frontales (índices 2, 3, 4 y 5 del anillo de 8 sensores)
    front_distances = [self.robot.range[i].distance() for i in range(2, 6)]
    min_front_dist = min(front_distances)

    if min_front_dist < 0.4:
      # Si hay un obstáculo muy cerca (a menos de 0.4m), giramos a la izquierda para rodearlo
      self.move(self.MED_FORWARD, self.HARD_LEFT)
      
    elif not foundLine:
      # Si perdemos la línea (por ejemplo, al esquivar el obstáculo), 
      # giramos suavemente hacia la derecha para intentar reencontrarla
      self.move(self.MED_FORWARD, self.MED_RIGHT)
      
    else:
      # 2. SEGUIMIENTO DE LÍNEA (Controlador Proporcional)
      # Fórmulas extraídas de las diapositivas de clase
      tv = -1.0 * error
      fv = max(0.2, 1.0 - abs(tv * 1.5)) # Si el giro es brusco, el robot frena un poco
      
      self.move(fv, tv)

    else:
      # Se perdió la línea: buscar hacia el último lado conocido
      self.frames_without_line += 1

      if self.frames_without_line >= self.LINE_END_THRESHOLD:
        self.line_ended = True
        self.move(self.NO_FORWARD, self.NO_TURN)
      else:
        # Girar hacia el lado donde estaba la línea por última vez
        if self.last_error < 0:
          self.move(self.SLOW_FORWARD, self.MED_LEFT)
        elif self.last_error > 0:
          self.move(self.SLOW_FORWARD, self.MED_RIGHT)
        else:
          self.move(self.SLOW_FORWARD, self.NO_TURN)

def INIT(engine):
    assert (engine.robot.requires("range-sensor") and
            engine.robot.requires("continuous-movement"))

    return BrainFollowLine('BrainFollowLine', engine)