# Despliegue en el robot — Guía de instalación y ejecución

> Este documento describe **cómo subir el código al controlador del
> robot y arrancarlo**. No forma parte de la memoria de la práctica;
> es una guía operativa.

---

## 1. Estructura mínima a copiar al robot

En la raíz del proyecto sólo hace falta llevar al robot lo siguiente:

```
/home/robot/practica2/
├── main.py                       ← punto de entrada
├── requirements.txt
├── imagen_original.png           ← calibración del QDA
├── imagen_marcada.png            ← calibración del QDA
├── marcas-capturasStage/         ← dataset del clasificador (28 imgs)
│   ├── man-1.png    ... man-7.png
│   ├── stairs-1.png ... stairs-7.png
│   ├── telephone-1.png ... telephone-7.png
│   └── woman-1.png  ... woman-7.png
└── robot_vision/                 ← paquete completo
    ├── __init__.py
    ├── config.py
    ├── segmentacion.py
    ├── escena.py
    ├── flecha.py
    ├── control.py
    ├── distancia.py
    ├── marcas.py
    ├── anotacion.py
    └── pipeline.py
```

**No** hace falta llevar al robot:

* La carpeta `notebooks/`
* La carpeta `memoria/`
* Los vídeos de prueba (`video1.mp4`, `video2017-3.avi`, …)
* Los wrappers de retrocompatibilidad (`analisis_escena.py`,
  `clasificador_marcas.py`, `segmentacion_hsv.py`)
* La carpeta `scripts/` (sólo si vas a reentrenar en el propio robot)

> ✅ Tamaño total a transferir: ~5 MB.

---

## 2. Cómo copiar los archivos al robot

### Opción A — vía SSH/SCP (recomendado para robots con Linux)

Desde tu portátil, en la carpeta `Practica 2 - Percepcion Computacional/`:

```bash
# 1) Crea la carpeta de destino en el robot
ssh robot@<IP_ROBOT> "mkdir -p /home/robot/practica2"

# 2) Copia el código y los datos imprescindibles
scp -r main.py requirements.txt \
       imagen_original.png imagen_marcada.png \
       marcas-capturasStage \
       robot_vision \
       robot@<IP_ROBOT>:/home/robot/practica2/
```

### Opción B — vía USB / pendrive

Copia la misma estructura mínima del paso 1 a un pendrive y, ya en
el robot:

```bash
mkdir -p /home/robot/practica2
cp -r /media/robot/USB/practica2/* /home/robot/practica2/
```

### Opción C — vía git

Si tienes el proyecto en un repositorio:

```bash
ssh robot@<IP_ROBOT>
cd /home/robot
git clone <URL_DEL_REPO> practica2
```

---

## 3. Instalación de dependencias en el robot

Una sola vez, conectado al robot:

```bash
cd /home/robot/practica2
python3 -m pip install --user -r requirements.txt
```

`requirements.txt`:

```
numpy>=1.23
opencv-python>=4.7
scikit-learn>=1.2
imageio>=2.25
matplotlib>=3.6        # opcional (sólo para demos / notebooks)
```

> En robots ROS basados en Ubuntu 20.04/22.04 ya viene `python3` y
> normalmente `python3-opencv`. Si OpenCV no está, también vale
> `sudo apt-get install python3-opencv`.

---

## 4. Comprobación rápida

```bash
cd /home/robot/practica2
python3 -c "import robot_vision as rv; print(rv.__version__)"
# → debe imprimir: 1.0.0
```

```bash
python3 main.py --help
# → debe listar: pipeline, segmentar, marcas, distancia, calibrar, live
```

---

## 5. Ejecución según la tarea

### 5.1 Modo en vivo (cámara del robot) — **el modo principal**

```bash
python3 main.py live --camara 0
```

Esto abre la cámara `/dev/video0`, aplica el pipeline completo
(segmentación + escena + flecha + marca + control PD) y muestra una
ventana con el frame anotado. La consigna `(v, ω)` se imprime por
consola cada segundo; en una integración real se publicaría en el
topic ROS correspondiente.

> En robots sin display, sustituye `cv2.imshow` por una salida
> alternativa o usa el modo headless del paso 5.5.

### 5.2 Procesar un vídeo grabado

```bash
python3 main.py pipeline \
    --video grabacion.mp4 \
    --salida grabacion_anotado.mp4
```

### 5.3 Sólo segmentación (útil para depurar)

```bash
python3 main.py segmentar \
    --video grabacion.mp4 \
    --salida-puro    seg_puro.mp4 \
    --salida-overlay seg_overlay.mp4
```

### 5.4 Calibración de la cámara (Parte 3)

```bash
# 1) Coloca la pelota a una distancia conocida (p.ej. 0.50 m) y haz
#    una foto:
python3 -c "import cv2; c=cv2.VideoCapture(0); _,f=c.read(); \
            cv2.imwrite('calib.png', f); c.release()"

# 2) Calibra la focal:
python3 main.py calibrar --imagen calib.png --distancia 0.50 --diametro 0.07

# Salida ejemplo:
#   Diametro detectado: 84.2 px
#   Distancia conocida: 0.500 m
#   Diametro real:       0.070 m
#   Distancia focal calibrada: f = 601.4 px
```

3) Edita `robot_vision/config.py` y pon `PELOTA_FOCAL_PX = 601.4`
   (o pásala con `--focal` cada vez).

### 5.5 Modo *headless* (sin pantalla) — integración ROS

Si vas a integrar con ROS sin que se abra ventana, en lugar del
modo `live` importa el paquete directamente desde tu nodo:

```python
# en tu nodo ROS:
import cv2, rospy, robot_vision as rv
import imageio.v2 as iio
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class VisionNode:
    def __init__(self):
        orig = iio.imread('/home/robot/practica2/imagen_original.png')
        marc = iio.imread('/home/robot/practica2/imagen_marcada.png')
        self.clf_qda = rv.segmentacion.entrenar_qda(orig, marc)
        ds = rv.marcas.cargar_dataset('/home/robot/practica2/marcas-capturasStage')
        self.clf_m  = rv.marcas.entrenar(ds)
        self.rng    = rv.marcas.rangos_por_clase(ds)
        self.ctrl   = rv.control.ControlPD()

        self.bridge = CvBridge()
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/camera/image_raw', Image, self.cb)

    def cb(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = rv.pipeline.analizar_frame(
            rgb, self.clf_qda, self.ctrl,
            clf_marcas=self.clf_m, rangos_marcas=self.rng)
        v, omega = res.consigna
        cmd = Twist()
        cmd.linear.x  = v
        cmd.angular.z = omega
        self.pub_cmd.publish(cmd)

if __name__ == '__main__':
    rospy.init_node('robot_vision')
    VisionNode()
    rospy.spin()
```

Coloca este nodo dentro de un paquete catkin `robot_vision_node/`
y lánzalo con:

```bash
roslaunch robot_vision_node robot_vision.launch
```

---

## 6. Ajustes en \texttt{config.py} antes del despliegue

Repasa los valores de `robot_vision/config.py` adaptándolos a tu robot:

| Parámetro                    | Por qué cambiarlo                                  |
|------------------------------|----------------------------------------------------|
| `CONTROL_PD_KP`, `KD`        | Depende de la cinemática real del robot.           |
| `CONTROL_PD_V_MAX`, `V_MIN`  | Velocidad máxima/mínima en m/s del robot.          |
| `BANDA_INFERIOR_ERROR_PX`    | Si la cámara va más alta o más baja, ajusta esto.  |
| `PELOTA_DIAMETRO_M`          | Diámetro real del balón que use el robot.          |
| `PELOTA_FOCAL_PX`            | Resultado de la calibración (paso 5.4).            |
| `PELOTA_HSV_LO/HI`           | Color de la pelota (verde, naranja, …).            |
| `AREA_MIN_LINEA / MARCA`     | Si la cámara tiene resolución muy diferente.       |

---

## 7. Arranque automático en el robot

Para que el sistema arranque solo al encender el robot, crea un
servicio systemd:

```bash
sudo tee /etc/systemd/system/robot-vision.service > /dev/null << 'EOF'
[Unit]
Description=Robot Vision (Practica 2)
After=network.target

[Service]
Type=simple
User=robot
WorkingDirectory=/home/robot/practica2
ExecStart=/usr/bin/python3 /home/robot/practica2/main.py live --camara 0
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable robot-vision.service
sudo systemctl start robot-vision.service

# Comprobaciones:
systemctl status robot-vision.service
journalctl -u robot-vision.service -f       # ver logs en directo
```

---

## 8. Solución de problemas

| Síntoma                                   | Causa probable / solución                              |
|-------------------------------------------|--------------------------------------------------------|
| `ModuleNotFoundError: robot_vision`        | Ejecuta desde `/home/robot/practica2`, no desde otra carpeta. |
| `cv2.error: ... imshow ... no display`     | Estás en headless. Usa la integración ROS o quita `cv2.imshow`. |
| `cannot open camera 0`                     | Comprueba `ls /dev/video*` y prueba `--camara 1` o ruta de stream. |
| Marca clasificada como otra clase           | Los píxeles del suelo confunden el descriptor; aumenta `MARCA_AREA_MIN` o ajusta los rangos HSV de la marca. |
| Línea perdida frecuentemente                | Aumenta el área de la franja inferior o baja `FRAC_MIN_LINEA`. |
| Distancia oscila mucho                      | Ruido en el contorno; prueba `--metodo hough` o suaviza con un filtro temporal. |

---

## 9. Resumen de comandos útiles

```bash
# ENTRENAMIENTO (sólo si quieres regenerar modelos en el robot)
python3 scripts/entrenar_qda.py    --salida modelos/qda.pkl
python3 scripts/entrenar_marcas.py --salida modelos/marcas.pkl

# DEMOSTRACIÓN OFFLINE
python3 main.py pipeline  --video <path>.mp4 --salida out.mp4
python3 main.py segmentar --video <path>.mp4 \
        --salida-puro p.mp4 --salida-overlay o.mp4
python3 main.py marcas    --video <path>.mp4 --salida m.mp4
python3 main.py distancia --video <path>.mp4 --salida d.mp4 \
        --diametro 0.07 --focal 601.4

# CALIBRACIÓN
python3 main.py calibrar  --imagen calib.png --distancia 0.5 --diametro 0.07

# TIEMPO REAL
python3 main.py live      --camara 0
```
