# 🕹️ Implementación y Control en Tiempo Real

Este módulo permite desplegar el modelo entrenado para realizar detección de gestos en vivo. El sistema procesa video de la cámara y señales del **Myo Armband** simultáneamente, enviando comandos de movimiento a un **Servomotor** controlado por Arduino.

---

## 🛠️ 1. Configuración del Hardware

#### Conexión del Arduino
El archivo `arduino_servo_control.txt` (debe cargarse como `.ino` en el IDE de Arduino) gestiona el movimiento físico.

**Diagrama de conexiones:**
* **Servo Signal:** Pin 9
* **Servo VCC:** 5V
* **Servo GND:** GND

#### Preparación del Myo Armband
1.  Asegúrate de tener el **Myo Connect** iniciado.
2.  Coloca el brazalete en el antebrazo y realiza el gesto de sincronización (Sync Gesture).
3.  Verifica que el brazalete esté conectado antes de lanzar el script de Python.

---

## 💻 2. Configuración del Software

Antes de ejecutar, abre `realtime_gesture_detection.py` y verifica las constantes en la sección `CONFIG`:

```python
CONFIG = {
    'model_path': 'best_model_synchronized.pth', # Tu modelo entrenado
    'arduino_port': 'COM7',                      # Ajusta según tu PC
    'confidence_threshold': 0.7,                 # Sensibilidad de detección
    'smoothing_window': 5                        # Suavizado de predicciones
}
```

---

## 🚀 3. Ejecución del Sistema

Sigue este orden estrictamente para asegurar que los puertos de comunicación no se bloqueen:

#### Paso 1: Cargar el código al Arduino
1. Abre el IDE de Arduino y carga el archivo `arduino_servo_control.txt` (renómbralo a `.ino`).
2. Conecta tu Arduino y selecciona el puerto correcto (ej. `COM7`).
3. Haz clic en **Subir (Upload)**. El servo realizará un movimiento inicial de prueba.

#### Paso 2: Iniciar el Script de Python
Desde tu terminal con el entorno activo (`cti_env_gpu`), ejecuta:

```bash
python realtime_gesture_detection.py
```

#### ⌨️ Controles de la Aplicación

Durante la ejecución del script `realtime_gesture_detection.py`, puedes interactuar con el sistema utilizando las siguientes teclas:

| Tecla | Acción | Descripción |
| :--- | :--- | :--- |
| `Q` | **Salir** | Finaliza la captura de video, detiene los hilos del Myo y cierra la conexión Serial de forma segura. |
| `R` | **Reset Servo** | Envía un comando inmediato para posicionar el servomotor en 90° (Posición neutral). |
| `S` | **Status** | Imprime en la terminal el estado de conexión del Myo y el llenado actual de los buffers de EMG e IMU. |

---

## 📊 4. Flujo de Trabajo y Lógica de Control

El sistema opera mediante un ciclo de retroalimentación de baja latencia que conecta la inteligencia artificial con el hardware físico:

#### Proceso de Inferencia Multimodal
1.  **Captura de Datos:** El sistema extrae simultáneamente el frame de la cámara y las últimas 400 muestras de los sensores del Myo Armband.
2.  **Preprocesamiento:** La imagen se redimensiona a `128x128` y las señales de los sensores se normalizan para entrar al modelo.
3.  **Predicción:** El modelo procesa las 4 ramas (RGB, Landmarks, EMG e IMU) y genera una probabilidad para cada gesto (`paper`, `rock`, `scissors`).

#### Control del Servomotor
Para evitar movimientos erráticos o vibraciones en el servo por falsos positivos, el script implementa dos capas de seguridad:

* **Smoothing Window:** Se promedian las últimas 5 predicciones. Solo si un gesto es constante se envía la orden de movimiento.
* **Confidence Threshold:** Solo se envían comandos al Arduino si la confianza del modelo es superior al **70%**.

#### Mapeo de Ángulos
Una vez validada la detección, se envía el ángulo correspondiente a través del puerto Serial:
* **Papel:** `0°` (Mano abierta)
* **Piedra:** `90°` (Puño cerrado)
* **Tijera:** `180°` (Gesto de victoria)

---

## ⚠️ Solución de Problemas (Troubleshooting)

Si encuentras dificultades al ejecutar la detección en tiempo real o al interactuar con el hardware, consulta esta guía de soluciones rápidas:

#### 1. Errores de Conexión Serial (Arduino)
* **Error:** `serial.serialutil.SerialException: could not open port 'COM7'`.
    * **Causa:** El puerto está siendo usado por otro programa (como el Monitor Serial del IDE de Arduino) o el nombre del puerto es incorrecto.
    * **Solución:** Cierra cualquier ventana de Arduino IDE que esté monitoreando el puerto y verifica el nombre del puerto en el Administrador de Dispositivos. Actualiza `CONFIG['arduino_port']` en el script si es necesario.

#### 2. Problemas con el Myo Armband
* **Detección pero no movimiento:** El Myo puede estar conectado pero bloqueado.
    * **Solución:** Realiza el gesto de sincronización (extender la mano y tocar el pulgar con el meñique). Verifica que el LED del brazalete esté fijo y no parpadeando.
* **Error de SDK:** `RuntimeError: Myo SDK not found`.
    * **Solución:** Asegúrate de que la carpeta del SDK de Myo esté en tu `PATH` o que el archivo `myo64.dll` esté en la misma carpeta que el script.

#### 3. Problemas de Detección e Inferencia
* **Lag o retraso en el video:** * **Causa:** El modelo está corriendo en CPU en lugar de GPU.
    * **Solución:** Verifica que el entorno de Conda tenga instalada la versión de PyTorch con soporte CUDA. El script imprimirá al inicio: `Using device: cuda`.
* **Falsos Positivos constantes:** * **Causa:** Mala iluminación o interferencia en los sensores EMG por sudor o mala colocación.
    * **Solución:** Limpia los sensores del brazalete con un poco de alcohol y asegúrate de que esté bien apretado en el antebrazo. Mejora la iluminación de la cámara para que MediaPipe no pierda los landmarks.

#### 4. Hardware (Servo)
* **El servo vibra o se mueve erráticamente:**
    * **Causa:** Falta de potencia eléctrica o ruido en la señal.
    * **Solución:** Asegúrate de que el GND del Arduino esté conectado al GND de la fuente de alimentación externa del servo (si usas una). Si alimentas el servo directamente desde el Arduino, intenta usar un puerto USB 3.0 para más corriente.

#### 🛠️ Tabla de Diagnóstico Rápido

| Síntoma | Verificar | Comando Útil |
| :--- | :--- | :--- |
| La cámara no abre | Índice de cámara en `cv2.VideoCapture()` | Cambiar `0` por `1` o `2` |
| El modelo no carga | Ruta del archivo `.pth` | `os.path.exists('best_model.pth')` |
| No llegan datos EMG | Myo Connect | Reiniciar el servicio Myo Connect |
| El servo no responde | Baud Rate | Debe ser `9600` en ambos códigos |

