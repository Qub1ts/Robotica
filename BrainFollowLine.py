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

  LINE_END_THRESHOLD = 40

  def setup(self):
    self.frames_without_line = 0
    self.line_ended = False
    self.avoiding = False   # Obstáculo activo enfrente
    self.post_avoid = False # Esquivamos un obstáculo, buscando la línea
    self.last_error = 0.0   # último error conocido de la línea

  def destroy(self):
    cv2.destroyAllWindows()

  def step(self):
    if self.line_ended:
      self.move(self.NO_FORWARD, self.NO_TURN)
      return

    cv_image = self.robot.getImage()
    cv2.imshow("Stage Camera Image", cv_image)
    cv2.waitKey(1)

    imageGray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    foundLine, error = findLineDeviation(imageGray)

    front_distances = [self.robot.range[i].distance() for i in range(2, 6)]
    min_front_dist = min(front_distances)
    obstacle_detected = min_front_dist < 0.4

    # ── 1. OBSTÁCULO ACTIVO ───────────────────────────────────────────────────
    if obstacle_detected:
      self.avoiding = True
      self.post_avoid = False
      self.frames_without_line = 0
      self.move(self.MED_FORWARD, self.HARD_LEFT)
      return

    # ── 2. SE DESPEJÓ EL OBSTÁCULO → entrar en búsqueda ──────────────────────
    if self.avoiding:
      self.avoiding = False
      self.post_avoid = True

    # ── 3. MODO BÚSQUEDA POST-EVASIÓN ─────────────────────────────────────────
    # Suponemos que la línea SIEMPRE reaparece después de un obstáculo
    if self.post_avoid:
      self.frames_without_line = 0  # Nunca contar fin de línea en este modo

      if foundLine:
        self.post_avoid = False 
      else:
        # Girar a la derecha buscando la línea, sin rendirse nunca
        self.move(self.MED_FORWARD, self.MED_RIGHT)
        return

    # ── 4. SEGUIMIENTO NORMAL ─────────────────────────────────────────────────
    if foundLine:
      self.frames_without_line = 0
      self.last_error = error

      tv = -1.0 * error
      fv = max(0.2, 1.0 - abs(tv * 1.5))
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