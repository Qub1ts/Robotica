---
name: practica2_percepcion_computacional
description: Estado y estructura de la Práctica 2 de Robótica - Percepción Computacional
type: project
---

Práctica 2 completada con todos los entregables generados.

**Why:** Asignatura de Robótica (Luis Baumela, USACH/UPM). Entrega completa requerida.

**How to apply:** Usar `.venv/bin/python` para ejecutar los scripts, no `python` ni `python3`.

## Entorno
- Python: `/Users/qub1ts/Documents/Robotica/.venv/bin/python`
- Video entrada: `video1.mp4` (320×240, 100fps, 1877 frames)

## Archivos clave
- `analisis_escena.py` — módulo principal de análisis (QDA, escena, flecha, control)
- `segmentacion_hsv.py` — segmentación HSV (escenarios 1 y 2)
- `segmentacion_QDA.ipynb` — notebook con entrenamiento QDA y visualizaciones
- `generar_video_resultado.py` — genera `video1_resultado.mp4`
- `generar_presentacion_pdf.py` — genera `presentacion_practica2.pdf`
- `imagen_original.png` + `imagen_marcada.png` — datos de entrenamiento manual

## Algoritmos implementados (según PDFs del curso)
- Features: HSV + RGB normalizado r/(r+g+b) para constancia de color
- Flecha: cv2.convexityDefects para detectar muescas → punta de flecha
- Escena: cv2.fitLine en franjas horizontales para detectar curvatura
- Control: tv = -(Kp·ẽ + Kd·Δẽ), fv = max(0.15, 0.8-|ẽ|)  [mismo que Práctica 1]
- Marcas: Momentos de Hu (log-normalizados) + ORB descriptor

## Entregables generados
- `video1_resultado.mp4` (4.6 MB, 103 fps de procesamiento)
- `presentacion_practica2.pdf` (1.0 MB, 8 slides)
