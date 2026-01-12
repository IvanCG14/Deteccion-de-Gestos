"""
Sistema de Detección de Gestos en Tiempo Real
Usando: Cámara + MediaPipe + Myo Armband + Modelo Multimodal
Envía comandos al Arduino para mover servo
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import collections
import threading
import time
import myo
import serial
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

# Importar tu modelo
from multimodal_myo_model import MultimodalGestureModelWithMyo

# ============================================
# CONFIGURACIÓN
# ============================================
CONFIG = {
    'model_path': 'best_model_synchronized01.pth',
    'classes': ['paper', 'rock', 'scissors'],
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Secuencias temporales
    'emg_seq_len': 400,
    'imu_seq_len': 100,
    'buffer_size': 400,  # Tamaño del buffer circular
    
    # Servo angles (ajusta según tu configuración)
    'servo_angles': {
        'paper': 0,
        'rock': 90,
        'scissors': 180,
        'neutral': 90
    },
    
    # Arduino
    'arduino_port': 'COM7',  # Ajusta según tu puerto
    'baud_rate': 9600,
    
    # Detección
    'confidence_threshold': 0.7,
    'smoothing_window': 5,  # Ventana para suavizar predicciones
}

# ============================================
# LISTENER MYO ARMBAND
# ============================================
class MyoListener(myo.DeviceListener):
    """Listener para capturar datos del Myo Armband"""
    
    def __init__(self, buffer_size=400):
        super().__init__()
        self.buffer_size = buffer_size
        self.lock = threading.Lock()
        
        # Buffers circulares para datos temporales
        self.emg_buffer = collections.deque(maxlen=buffer_size)
        self.orientation_buffer = collections.deque(maxlen=buffer_size)
        self.acceleration_buffer = collections.deque(maxlen=buffer_size)
        self.gyroscope_buffer = collections.deque(maxlen=buffer_size)
        
        self.connected = False
        self.last_update = time.time()
    
    def on_connected(self, event):
        print(f"✓ Myo conectado: {event.device_name}")
        event.device.vibrate(myo.VibrationType.short)
        event.device.stream_emg(True)
        self.connected = True
    
    def on_disconnected(self, event):
        print("✗ Myo desconectado")
        self.connected = False
    
    def on_orientation(self, event):
        with self.lock:
            self.orientation_buffer.append(list(event.orientation))
            self.acceleration_buffer.append(list(event.acceleration))
            self.gyroscope_buffer.append(list(event.gyroscope))
            self.last_update = time.time()
    
    def on_emg(self, event):
        with self.lock:
            self.emg_buffer.append(list(event.emg))
            self.last_update = time.time()
    
    def get_emg_sequence(self, seq_len):
        """Obtiene secuencia EMG normalizada"""
        with self.lock:
            if len(self.emg_buffer) == 0:
                return np.zeros((seq_len, 8), dtype=np.float32)
            
            # Convertir a array
            emg_array = np.array(list(self.emg_buffer), dtype=np.float32)
            
            # Normalizar [-128, 127] -> [-1, 1]
            emg_array = emg_array / 128.0
            
            # Ajustar longitud
            if len(emg_array) < seq_len:
                # Pad con últimos valores
                padding = np.repeat(emg_array[-1:], seq_len - len(emg_array), axis=0)
                emg_array = np.vstack([emg_array, padding])
            else:
                # Tomar últimos seq_len samples
                emg_array = emg_array[-seq_len:]
            
            return emg_array
    
    def get_imu_sequence(self, seq_len):
        """Obtiene secuencia IMU normalizada"""
        with self.lock:
            if len(self.orientation_buffer) == 0:
                return np.zeros((seq_len, 10), dtype=np.float32)
            
            # Obtener datos
            ori = np.array(list(self.orientation_buffer), dtype=np.float32)
            acc = np.array(list(self.acceleration_buffer), dtype=np.float32)
            gyr = np.array(list(self.gyroscope_buffer), dtype=np.float32)
            
            # Normalizar
            acc = acc / 10.0
            gyr = gyr / 100.0
            
            # Concatenar [ori(4) + acc(3) + gyr(3)] = 10
            imu_array = np.concatenate([ori, acc, gyr], axis=1)
            
            # Ajustar longitud
            if len(imu_array) < seq_len:
                padding = np.repeat(imu_array[-1:], seq_len - len(imu_array), axis=0)
                imu_array = np.vstack([imu_array, padding])
            else:
                imu_array = imu_array[-seq_len:]
            
            return imu_array
    
    def is_receiving_data(self):
        """Verifica si estamos recibiendo datos recientes"""
        return self.connected and (time.time() - self.last_update) < 1.0

# ============================================
# SISTEMA DE DETECCIÓN EN TIEMPO REAL
# ============================================
class GestureDetector:
    """Detector de gestos en tiempo real"""
    
    def __init__(self):

        # 1. Obtener la ruta de la carpeta raíz del proyecto (subiendo un nivel desde 'getdata')
        # __file__ es la ubicación de dataset_creator_myo.py
        BASE_DIR = Path(__file__).resolve().parent.parent

        # 2. Construir la ruta al SDK de forma relativa
        sdk_path = os.path.join(BASE_DIR, "MYO_armband_SDK", "myo-sdk-win-0.9.0")

        print("="*60)
        print("INICIALIZANDO SISTEMA DE DETECCIÓN")
        print("="*60)
        
        # Cargar modelo
        print("\n1. Cargando modelo...")
        self.device = torch.device(CONFIG['device'])
        self.model = MultimodalGestureModelWithMyo(
            num_classes=len(CONFIG['classes'])
        ).to(self.device)
        
        checkpoint = torch.load(CONFIG['model_path'], map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"   ✓ Modelo cargado desde {CONFIG['model_path']}")
        
        # Stats de landmarks (desde checkpoint)
        if 'config' in checkpoint and 'landmark_stats' in checkpoint:
            stats = checkpoint['landmark_stats']
            self.lm_mean = stats['mean']
            self.lm_std = stats['std']
        else:
            # Valores por defecto si no están guardados
            self.lm_mean = np.zeros(63, dtype=np.float32)
            self.lm_std = np.ones(63, dtype=np.float32)

        # Estado
        self.running = True
        self.current_gesture = 'neutral'
         # Buffer para suavizar predicciones
        self.prediction_buffer = collections.deque(maxlen=CONFIG['smoothing_window'])
        
        # MediaPipe
        print("\n2. Inicializando MediaPipe...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        print("   ✓ MediaPipe inicializado")
        
        # Transform para imágenes
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Myo Armband
        print("\n3. Inicializando Myo Armband...")
        myo.init(sdk_path=sdk_path)
        self.hub = myo.Hub()
        self.myo_listener = MyoListener(buffer_size=CONFIG['buffer_size'])
        
        # Thread para Myo
        self.myo_thread = threading.Thread(target=self._run_myo_hub, daemon=True)
        self.myo_thread.start()
        print("   ✓ Myo Armband inicializado")
        
        # Arduino
        print("\n4. Conectando Arduino...")
        try:
            self.arduino = serial.Serial(CONFIG['arduino_port'], CONFIG['baud_rate'], timeout=1)
            time.sleep(2)  # Esperar conexión
            print(f"   ✓ Arduino conectado en {CONFIG['arduino_port']}")
        except Exception as e:
            print(f"   ✗ Error conectando Arduino: {e}")
            self.arduino = None
        
        
        print("\n" + "="*60)
        print("✓ SISTEMA LISTO")
        print("="*60)
        print("\nControles:")
        print("  'q' - Salir")
        print("  'r' - Reset servo a posición neutral")
        print("  's' - Mostrar estadísticas")
        print()
    
    def _run_myo_hub(self):
        """Thread para ejecutar el hub de Myo"""
        while self.running:
            self.hub.run(self.myo_listener.on_event, 500)
            time.sleep(0.01)
    
    def extract_landmarks(self, frame):
        """Extrae landmarks de MediaPipe"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extraer coordenadas [x, y, z] de 21 landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            landmarks = np.array(landmarks, dtype=np.float32)
            
            # Normalizar
            landmarks = (landmarks - self.lm_mean) / self.lm_std
            
            # Dibujar en frame
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )
            
            return landmarks, True
        
        return np.zeros(63, dtype=np.float32), False
    
    def predict_gesture(self, frame):
        """Predice gesto usando todas las modalidades"""
        
        # 1. Extraer landmarks
        landmarks, hand_detected = self.extract_landmarks(frame)
        
        if not hand_detected:
            return None, 0.0
        
        # 2. Preparar imagen
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = self.transform(frame_pil).unsqueeze(0).to(self.device)
        
        # 3. Preparar landmarks
        landmarks_tensor = torch.from_numpy(landmarks).unsqueeze(0).to(self.device)
        
        # 4. Obtener secuencias de Myo
        if self.myo_listener.is_receiving_data():
            emg_seq = self.myo_listener.get_emg_sequence(CONFIG['emg_seq_len'])
            imu_seq = self.myo_listener.get_imu_sequence(CONFIG['imu_seq_len'])
            
            emg_tensor = torch.from_numpy(emg_seq).unsqueeze(0).to(self.device)
            imu_tensor = torch.from_numpy(imu_seq).unsqueeze(0).to(self.device)
        else:
            # Si Myo no disponible, usar zeros
            emg_tensor = None
            imu_tensor = None
        
        # 5. Inferencia
        with torch.no_grad():
            output = self.model(
                image=image_tensor,
                landmarks=landmarks_tensor,
                emg=emg_tensor,
                imu=imu_tensor
            )
            
            probs = torch.softmax(output['logits'], dim=1)
            confidence, pred_idx = probs.max(1)
            
            gesture = CONFIG['classes'][pred_idx.item()]
            confidence = confidence.item()
        
        return gesture, confidence
    
    def smooth_prediction(self, gesture, confidence):
        """Suaviza predicciones usando ventana temporal"""
        
        if confidence < CONFIG['confidence_threshold']:
            return None
        
        self.prediction_buffer.append(gesture)
        
        if len(self.prediction_buffer) < CONFIG['smoothing_window']:
            return None
        
        # Retornar gesto más frecuente
        from collections import Counter
        counts = Counter(self.prediction_buffer)
        most_common = counts.most_common(1)[0]
        
        # Requiere al menos 60% de consistencia
        if most_common[1] >= CONFIG['smoothing_window'] * 0.6:
            return most_common[0]
        
        return None
    
    def send_to_arduino(self, gesture):
        """Envía comando al Arduino para mover servo"""
        if self.arduino is None:
            return
        
        angle = CONFIG['servo_angles'].get(gesture, 90)
        
        try:
            command = f"{angle}\n"
            self.arduino.write(command.encode())
            print(f"   → Servo: {gesture} ({angle}°)")
        except Exception as e:
            print(f"   ✗ Error enviando comando: {e}")
    
    def run(self):
        """Loop principal de detección"""
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        fps_counter = 0
        fps_start = time.time()
        
        print("\n▶ Iniciando detección...")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip para efecto espejo
            frame = cv2.flip(frame, 1)
            
            # Predecir gesto
            gesture, confidence = self.predict_gesture(frame)
            
            # Suavizar predicción
            if gesture:
                smoothed_gesture = self.smooth_prediction(gesture, confidence)
                
                if smoothed_gesture and smoothed_gesture != self.current_gesture:
                    self.current_gesture = smoothed_gesture
                    self.send_to_arduino(smoothed_gesture)
            
            # Mostrar información en frame
            h, w = frame.shape[:2]
            
            # Estado de Myo
            myo_status = "✓ Conectado" if self.myo_listener.is_receiving_data() else "✗ Desconectado"
            cv2.putText(frame, f"Myo: {myo_status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.myo_listener.connected else (0, 0, 255), 2)
            
            # Gesto actual
            if gesture:
                color = (0, 255, 0) if confidence > CONFIG['confidence_threshold'] else (0, 165, 255)
                cv2.putText(frame, f"Gesto: {gesture} ({confidence:.2f})", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Gesto confirmado
            cv2.putText(frame, f"Confirmado: {self.current_gesture}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # FPS
            fps_counter += 1
            if time.time() - fps_start > 1.0:
                fps = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start = time.time()
            else:
                fps = fps_counter / max(time.time() - fps_start, 0.001)
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Detección de Gestos', frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.current_gesture = 'neutral'
                self.send_to_arduino('neutral')
                print("   ⟳ Reset a posición neutral")
            elif key == ord('s'):
                print(f"\n--- Estadísticas ---")
                print(f"Gesto actual: {self.current_gesture}")
                print(f"Myo conectado: {self.myo_listener.connected}")
                print(f"EMG buffer: {len(self.myo_listener.emg_buffer)}")
                print(f"IMU buffer: {len(self.myo_listener.orientation_buffer)}")
                print("-------------------\n")
        
        # Cleanup
        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        if self.arduino:
            self.arduino.close()
        
        print("\n✓ Sistema cerrado correctamente")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    try:
        detector = GestureDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\n⚠ Interrupción del usuario")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
