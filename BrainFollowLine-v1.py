from pyrobot.brain import Brain

import cv2
from pyrobot.tools.followLineTools import findLineDeviation

class BrainFollowLine(Brain):

  # Constantes de control
  Kp = 1.0          # Ganancia proporcional para el giro
  FORWARD_VEL = 0.5 # Velocidad base hacia adelante

  # Umbral de distancia para detectar obstáculo (en metros)
  OBSTACLE_THRESHOLD = 0.4

  # Cuántos pasos girar para esquivar el obstáculo
  AVOID_STEPS = 15

  def setup(self):
    self.avoidCounter = 0  # Contador para esquivar el obstáculo

  def destroy(self):
    cv2.destroyAllWindows()

  def step(self):
    cv_image = self.robot.getImage()

    # Mostrar imagen de la cámara
    cv2.imshow("Stage Camera Image", cv_image)
    cv2.waitKey(1)

    # --- DETECCIÓN DE OBSTÁCULO ---
    # Leer todos los sensores de rango y quedarse con el mínimo frontal
    numSensors = len(self.robot.range)
    minFront = float('inf')
    for i in range(numSensors):
      dist = self.robot.range[i].distance()
      if dist < minFront:
        minFront = dist

    # Si estamos en modo evasión, seguir girando unos pasos
    if self.avoidCounter > 0:
      self.avoidCounter -= 1
      self.move(self.FORWARD_VEL, 0.8)  # Gira a la izquierda mientras avanza
      return

    # Si hay un obstáculo cerca, activar la evasión
    if minFront < self.OBSTACLE_THRESHOLD:
      self.avoidCounter = self.AVOID_STEPS
      self.move(0.0, 0.8)  # Para y gira
      return

    # --- SEGUIMIENTO DE LÍNEA (Controlador Proporcional) ---
    imageGray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    foundLine, error = findLineDeviation(imageGray)

    if foundLine:
      # Turn velocity proporcional al error (negativo porque error>0 → girar derecha)
      turn_vel = -self.Kp * error

      # Forward velocity: reducir velocidad cuanto más hay que girar
      forward_vel = max(0.1, self.FORWARD_VEL - abs(turn_vel) * 0.5)

      self.move(forward_vel, turn_vel)
    else:
      # Si pierde la línea, gira lentamente para buscarla
      self.move(0.0, 0.3)


def INIT(engine):
  assert (engine.robot.requires("range-sensor") and
          engine.robot.requires("continuous-movement"))

  return BrainFollowLine('BrainFollowLine', engine)
