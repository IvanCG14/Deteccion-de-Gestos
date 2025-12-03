# 📸 Hand Gesture Dataset Creator

Sistema para crear datasets de gestos de mano (Rock, Paper, Scissors) usando MediaPipe y OpenCV, con extracción automática de landmarks en formato CSV.

## 📋 Tabla de Contenidos
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Uso](#uso)
- [Formato de Datos](#formato-de-datos)
- [Landmarks de MediaPipe](#landmarks-de-mediapipe)
- [Solución de Problemas](#solución-de-problemas)

---

## 🔧 Requisitos

### Software
- Python 3.7 o superior
- Webcam funcional
- Sistema operativo: Windows, macOS, o Linux

### Librerías Principales
```
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.21.0
```

---

## 📦 Instalación

### 1. Clonar/Descargar el proyecto
```bash
cd hand-dataset-creator
```

### 2. Crear entorno virtual (recomendado)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install mediapipe opencv-python numpy
```

O usando requirements.txt:
```bash
pip install -r requirements.txt
```

**Archivo `requirements.txt`:**
```
mediapipe==0.10.14
opencv-python==4.8.1.78
numpy==1.24.3
```

---

## 📁 Estructura del Proyecto

```
hand-dataset-creator/
│
├── dataset_creator.py          # Script principal
├── requirements.txt            # Dependencias
├── README.md                   # Este archivo
│
└── dataset/                    # Carpeta generada automáticamente
    ├── images/                 # Imágenes capturadas
    │   ├── rock/              # Imágenes de "piedra"
    │   ├── paper/             # Imágenes de "papel"
    │   ├── scissors/          # Imágenes de "tijeras"
    │   └── none/              # Otros gestos
    │
    └── landmarks/             # Coordenadas en CSV
        ├── rock_landmarks.csv
        ├── paper_landmarks.csv
        ├── scissors_landmarks.csv
        └── none_landmarks.csv
```

---

## 🎮 Uso

### Ejecutar el programa
```bash
python dataset_creator.py
```

### Controles del Teclado

| Tecla | Función |
|-------|---------|
| `1` | Modo ROCK (piedra) |
| `2` | Modo PAPER (papel) |
| `3` | Modo SCISSORS (tijeras) |
| `4` | Modo NONE (sin gesto) |
| `ESPACIO` | Iniciar/Pausar captura automática |
| `S` | Capturar imagen individual |
| `Q` | Salir del programa |

### Proceso Recomendado

1. **Inicia el programa**
   ```bash
   python dataset_creator.py
   ```

2. **Selecciona el gesto** (presiona 1, 2, 3, o 4)

3. **Inicia la captura automática** (presiona ESPACIO)

4. **Mueve tu mano** en diferentes:
   - Ángulos (horizontal, vertical, diagonal)
   - Posiciones (cerca, lejos, izquierda, derecha)
   - Rotaciones (palm up, palm down, lateral)
   - Distancias a la cámara

5. **Captura 100-200 imágenes** por gesto

6. **Repite** para cada gesto (rock, paper, scissors)

### Ejemplo de Sesión
```
Rock...
Paper...
Scissors...
Shoot!

>>> Modo cambiado a: ROCK
>>> Captura INICIADA
✓ Guardado: rock #1
✓ Guardado: rock #2
...
✓ Guardado: rock #150
>>> Captura PAUSADA

>>> Modo cambiado a: PAPER
>>> Captura INICIADA
✓ Guardado: paper #1
...
```

---

## 📊 Formato de Datos

### Imágenes
- **Formato:** JPG
- **Resolución:** 640x480 píxeles
- **Nomenclatura:** `{gesto}_{número:04d}.jpg`
- **Ejemplo:** `rock_0001.jpg`, `paper_0042.jpg`

### CSV de Landmarks

Cada archivo CSV contiene las coordenadas 3D de 21 puntos de la mano:

```csv
image_file,label,x0,y0,z0,x1,y1,z1,...,x20,y20,z20
dataset/images/rock/rock_0001.jpg,rock,320,240,0,350,220,5,...,280,180,10
```

**Columnas:**
- `image_file`: Ruta de la imagen
- `label`: Etiqueta del gesto (rock/paper/scissors/none)
- `x0-x20`: Coordenada X de cada landmark (píxeles)
- `y0-y20`: Coordenada Y de cada landmark (píxeles)
- `z0-z20`: Coordenada Z de cada landmark (profundidad relativa)

---

## 🖐️ Landmarks de MediaPipe

MediaPipe detecta **21 puntos** en la mano:

### Estructura de la Mano
```
       ( P1 ) ( I1 ) ( M1 ) ( A1 ) ( E1 )  <== 5 Marcadores en Puntas (Falange Distal)
         |      |      |      |      |
         |   .---.  .---.  .---.    |
         |  ( I2 )( M2 )( A2 )      |      <== 3 Marcadores en Falange Intermedia (PIP)
         |   `|'   `|'   `|'        |
         |      |      |      |      |
       .---.  .---.  .---.  .---.  .---.
      ( P2 )( I3 )( M3 )( A3 )( E2 )      <== 5 Marcadores en Nudillos (MCP)
       `-'    `-'    `-'    `-'    `-'
         \____/      \____/


            .---.  .---.  .---.
           ( R ) ( C ) ( U )             <== 3 Marcadores en Muñeca
            `-'   `-|'   `-'
```

### Índices de Landmarks

| Índice | Nombre | Descripción |
|--------|--------|-------------|
| 0 | WRIST | Muñeca |
| 1 | THUMB_CMC | Base del pulgar |
| 2 | THUMB_MCP | Nudillo del pulgar |
| 3 | THUMB_IP | Articulación del pulgar |
| 4 | THUMB_TIP | Punta del pulgar |
| 5 | INDEX_FINGER_MCP | Nudillo del índice |
| 6 | INDEX_FINGER_PIP | Articulación media del índice |
| 7 | INDEX_FINGER_DIP | Articulación distal del índice |
| 8 | INDEX_FINGER_TIP | Punta del índice |
| 9 | MIDDLE_FINGER_MCP | Nudillo del medio |
| 10 | MIDDLE_FINGER_PIP | Articulación media del medio |
| 11 | MIDDLE_FINGER_DIP | Articulación distal del medio |
| 12 | MIDDLE_FINGER_TIP | Punta del medio |
| 13 | RING_FINGER_MCP | Nudillo del anular |
| 14 | RING_FINGER_PIP | Articulación media del anular |
| 15 | RING_FINGER_DIP | Articulación distal del anular |
| 16 | RING_FINGER_TIP | Punta del anular |
| 17 | PINKY_MCP | Nudillo del meñique |
| 18 | PINKY_PIP | Articulación media del meñique |
| 19 | PINKY_DIP | Articulación distal del meñique |
| 20 | PINKY_TIP | Punta del meñique |

### Mapeo de Nomenclatura

```
P1, P2 = Pulgar (Thumb)
I1, I2, I3 = Índice (Index)
M1, M2, M3 = Medio (Middle)
A1, A2, A3 = Anular (Ring)
E1, E2 = Meñique (Pinky)
R, C, U = Muñeca (Wrist)
```

---

## 🎯 Recomendaciones para un Buen Dataset

### Cantidad de Datos
- **Mínimo:** 100 imágenes por clase
- **Recomendado:** 200-300 imágenes por clase
- **Óptimo:** 500+ imágenes por clase

### Variedad
✅ **SÍ hacer:**
- Diferentes ángulos de la mano
- Diferentes distancias a la cámara
- Diferentes rotaciones (palm up, down, lateral)
- Diferentes posiciones en el encuadre
- Diferentes iluminaciones (si es posible)
- Ambas manos (izquierda y derecha)

❌ **NO hacer:**
- Motion blur (movimientos muy rápidos)
- Mano parcialmente fuera del encuadre
- Dedos ocultos u ocluidos
- Iluminación muy baja (mano no visible)

### Balance del Dataset
Intenta tener un número similar de imágenes en cada clase:
```
Rock:     250 imágenes
Paper:    240 imágenes
Scissors: 260 imágenes
None:     100 imágenes (opcional)
```

---

## 🔍 Solución de Problemas

### Error: "No module named 'mediapipe'"
```bash
pip install mediapipe opencv-python
```

### Error: "Can't open camera"
- Verifica que tu webcam esté conectada
- Cierra otras aplicaciones que usen la cámara (Zoom, Teams, etc.)
- En Linux, verifica permisos: `sudo usermod -a -G video $USER`

### La mano no se detecta
- Mejora la iluminación
- Acerca más la mano a la cámara
- Asegúrate de que toda la mano esté visible
- Prueba con un fondo menos complejo

### Imágenes borrosas
- Reduce la velocidad de movimiento de la mano
- Mantén la mano más estable
- Mejora la iluminación

### El programa está lento
- Cierra otras aplicaciones
- Verifica que tienes buena CPU (MediaPipe es intensivo)
- Reduce la resolución en el código si es necesario

---

## 📈 Siguientes Pasos

Una vez tengas tu dataset:

1. **Entrenar un modelo de clasificación:**
   ```python
   from sklearn.ensemble import RandomForestClassifier
   import pandas as pd
   
   df = pd.read_csv('dataset/landmarks/rock_landmarks.csv')
   # ... entrenar modelo
   ```

2. **Usar el detector de gestos en tiempo real:**
   - Carga tu modelo entrenado
   - Aplica el mismo feature engineering
   - Clasifica gestos en tiempo real

3. **Mejorar el dataset:**
   - Añadir más variedad
   - Balancear las clases
   - Capturar con diferentes personas

---

## 🤝 Contribuciones

Si mejoras el código o encuentras bugs, ¡comparte tus cambios!

---

## 📝 Licencia

Este proyecto es de código abierto para uso educativo y de investigación.

---

## 📧 Contacto

Para preguntas o sugerencias sobre el dataset creator, consulta la documentación de:
- [MediaPipe Hand Landmark Detection](https://google.github.io/mediapipe/solutions/hands.html)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

---

**Creado con ❤️ para investigación en Computer Vision y Machine Learning**
