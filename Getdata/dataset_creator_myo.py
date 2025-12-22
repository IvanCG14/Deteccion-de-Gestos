import mediapipe
import cv2
import math
import os
import csv
import time
import shutil
import myo
import threading
import collections
import numpy as np
import pandas as pd
import json
from datetime import datetime

# ==================== CONFIGURACIÓN MYO ARMBAND ====================
class MyoListener(myo.DeviceListener):
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        
        # Buffers para datos
        self.orientation = collections.deque(maxlen=500)
        self.accel = collections.deque(maxlen=500)
        self.gyro = collections.deque(maxlen=500)
        self.timestamps_imu = collections.deque(maxlen=500)
        
        self.emg = [collections.deque(maxlen=500) for _ in range(8)]
        self.timestamps_emg = collections.deque(maxlen=500)
        
        # Device info
        self.battery_level = None
        self.rssi = None
        self.connected = False

    def on_connected(self, event):
        try:
            event.device.stream_emg(True)
        except Exception:
            try:
                event.device.stream_emg = myo.StreamEmg.enabled
            except Exception:
                pass
        
        # Battery / RSSI
        try:
            self.battery_level = getattr(event.device, "battery_level", None)
            if self.battery_level is None:
                self.battery_level = getattr(event.device, "battery", None)
        except Exception:
            self.battery_level = None
        
        try:
            self.rssi = getattr(event.device, "rssi", None)
        except Exception:
            self.rssi = None
        
        # Vibrar como feedback
        try:
            event.device.vibrate(myo.VibrationType.short)
        except Exception:
            pass
        
        self.connected = True
        print("[INFO] Myo Armband conectado correctamente")
        print(f"       Battery: {self.battery_level}, RSSI: {self.rssi}")

    def on_disconnected(self, event):
        self.connected = False
        print("[WARNING] Myo Armband desconectado")

    def on_orientation(self, event):
        with self.lock:
            ts = int(time.time() * 1e6)  # microsegundos
            self.orientation.append([event.orientation.x, event.orientation.y,
                                     event.orientation.z, event.orientation.w])
            try:
                self.accel.append([event.acceleration.x, event.acceleration.y, 
                                  event.acceleration.z])
            except Exception:
                self.accel.append([0.0, 0.0, 0.0])
            try:
                self.gyro.append([event.gyroscope.x, event.gyroscope.y, 
                                 event.gyroscope.z])
            except Exception:
                self.gyro.append([0.0, 0.0, 0.0])
            self.timestamps_imu.append(ts)

    def on_emg(self, event):
        with self.lock:
            ts = int(time.time() * 1e6)
            self.timestamps_emg.append(ts)
            for i, val in enumerate(event.emg):
                self.emg[i].append(val)

    def get_snapshot(self):
        """Obtiene una copia de todos los datos actuales"""
        with self.lock:
            return {
                'emg_timestamps': list(self.timestamps_emg),
                'emg_channels': [list(ch) for ch in self.emg],
                'imu_timestamps': list(self.timestamps_imu),
                'orientation': list(self.orientation),
                'accel': list(self.accel),
                'gyro': list(self.gyro)
            }
    
    def reset_buffers(self):
        """Limpia todos los buffers"""
        with self.lock:
            self.orientation.clear()
            self.accel.clear()
            self.gyro.clear()
            self.timestamps_imu.clear()
            for ch in self.emg:
                ch.clear()
            self.timestamps_emg.clear()

# ==================== MEDIAPIPE SETUP ====================
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

# Configuración de la cámara
cap = cv2.VideoCapture(0)
h = 480
w = 640

# ==================== CREAR CARPETAS DEL DATASET ====================
def create_dataset_folders():
    """Crea la estructura de carpetas del dataset"""
    folders = [
        'dataset/images/rock',
        'dataset/images/paper',
        'dataset/images/scissors',
        'dataset/images/none',
        'dataset/landmarks',
        'dataset/emg',
        'dataset/imu'
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

create_dataset_folders()

# ==================== FUNCIONES AUXILIARES ====================
def get_current_counters():
    """Lee los contadores actuales del dataset existente"""
    counters = {'rock': 0, 'paper': 0, 'scissors': 0, 'none': 0}
    
    for gesture in counters.keys():
        folder = f'dataset/images/{gesture}'
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
            if files:
                numbers = []
                for f in files:
                    try:
                        num = int(f.split('_')[1].split('.')[0])
                        numbers.append(num)
                    except:
                        pass
                if numbers:
                    counters[gesture] = max(numbers)
    
    return counters

def calculate_distance(point1, point2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_landmarks_from_mediapipe(handLandmarks):
    """Extrae landmarks de MediaPipe"""
    landmarks = []
    for id, pt in enumerate(handLandmarks.landmark):
        x = int(pt.x * w)
        y = int(pt.y * h)
        landmarks.append([id, x, y])
    return landmarks

# ==================== GUARDAR MUESTRA SINCRONIZADA ====================
def save_synchronized_sample(frame_clean, landmarks, myo_data, gesture_label, count):
    """
    Guarda una muestra completa sincronizada:
    - Imagen (sin landmarks)
    - Landmarks (CSV)
    - EMG (CSV)
    - IMU (CSV)
    - Metadata (JSON)
    """
    timestamp = int(time.time() * 1e6)
    
    # 1. Guardar imagen SIN landmarks
    img_filename = f'{gesture_label}_{count:04d}.jpg'
    img_path = f'dataset/images/{gesture_label}/{img_filename}'
    cv2.imwrite(img_path, frame_clean)
    
    # 2. Guardar landmarks en CSV
    landmarks_filename = f'{gesture_label}_{count:04d}_landmarks.csv'
    landmarks_path = f'dataset/landmarks/{landmarks_filename}'
    with open(landmarks_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['landmark_id', 'x', 'y', 'z']
        writer.writerow(header)
        for landmark in landmarks:
            writer.writerow([landmark[0], landmark[1], landmark[2], 0])
    
    # 3. Guardar EMG en CSV
    emg_filename = f'{gesture_label}_{count:04d}_emg.csv'
    emg_path = f'dataset/emg/{emg_filename}'
    emg_data = []
    emg_timestamps = myo_data['emg_timestamps']
    emg_channels = myo_data['emg_channels']
    
    max_len = max(len(emg_timestamps), max(len(ch) for ch in emg_channels) if emg_channels else 0)
    
    with open(emg_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['index', 'timestamp'] + [f'ch{i}' for i in range(8)]
        writer.writerow(header)
        
        for i in range(max_len):
            ts = emg_timestamps[i] if i < len(emg_timestamps) else 0
            row = [i, ts]
            for ch_idx in range(8):
                if i < len(emg_channels[ch_idx]):
                    row.append(emg_channels[ch_idx][i])
                else:
                    row.append(0)
            writer.writerow(row)
    
    # 4. Guardar IMU en CSV
    imu_filename = f'{gesture_label}_{count:04d}_imu.csv'
    imu_path = f'dataset/imu/{imu_filename}'
    
    imu_timestamps = myo_data['imu_timestamps']
    orientation = myo_data['orientation']
    accel = myo_data['accel']
    gyro = myo_data['gyro']
    
    max_len_imu = max(len(imu_timestamps), len(orientation), len(accel), len(gyro))
    
    with open(imu_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['index', 'timestamp', 
                  'ori_x', 'ori_y', 'ori_z', 'ori_w',
                  'acc_x', 'acc_y', 'acc_z',
                  'gyr_x', 'gyr_y', 'gyr_z']
        writer.writerow(header)
        
        for i in range(max_len_imu):
            ts = imu_timestamps[i] if i < len(imu_timestamps) else 0
            ori = orientation[i] if i < len(orientation) else [0, 0, 0, 0]
            acc = accel[i] if i < len(accel) else [0, 0, 0]
            gyr = gyro[i] if i < len(gyro) else [0, 0, 0]
            
            row = [i, ts] + ori + acc + gyr
            writer.writerow(row)
    
    # 5. Guardar metadata en JSON
    meta_filename = f'{gesture_label}_{count:04d}_meta.json'
    meta_path = f'dataset/landmarks/{meta_filename}'
    
    now = datetime.now()
    metadata = {
        'gesture': gesture_label,
        'sample_id': count,
        'timestamp': timestamp,
        'date': now.strftime("%d/%m/%Y"),
        'time': now.strftime("%H:%M:%S"),
        'files': {
            'image': img_filename,
            'landmarks': landmarks_filename,
            'emg': emg_filename,
            'imu': imu_filename
        },
        'data_counts': {
            'landmarks': len(landmarks),
            'emg_samples': len(emg_timestamps),
            'imu_samples': len(imu_timestamps)
        }
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Muestra sincronizada guardada: {gesture_label} #{count}")
    print(f"  - Landmarks: {len(landmarks)} puntos")
    print(f"  - EMG: {len(emg_timestamps)} muestras")
    print(f"  - IMU: {len(imu_timestamps)} muestras")

# ==================== ELIMINAR DATASET ====================
def delete_dataset():
    """Elimina completamente el dataset"""
    response = input("\n⚠️  ¿Estás seguro de que quieres ELIMINAR TODO el dataset? (sí/no): ")
    if response.lower() in ['sí', 'si', 'yes', 's', 'y']:
        try:
            if os.path.exists('dataset'):
                shutil.rmtree('dataset')
                print("\n✓ Dataset eliminado completamente")
                create_dataset_folders()
                global counters
                counters = {'rock': 0, 'paper': 0, 'scissors': 0, 'none': 0}
                print("✓ Carpetas recreadas y contadores reseteados")
            else:
                print("\n⚠️  No existe dataset para eliminar")
        except Exception as e:
            print(f"\n❌ Error al eliminar dataset: {e}")
    else:
        print("\n✓ Operación cancelada")

# ==================== INICIALIZAR MYO ====================
import os
from pathlib import Path

# 1. Obtener la ruta de la carpeta raíz del proyecto (subiendo un nivel desde 'getdata')
# __file__ es la ubicación de dataset_creator_myo.py
BASE_DIR = Path(__file__).resolve().parent.parent 

# 2. Construir la ruta al SDK de forma relativa
sdk_path = os.path.join(BASE_DIR, "MYO_armband_SDK", "myo-sdk-win-0.9.0")

def initialize_myo():
    """Inicializa el Myo Armband"""
    try:
        myo.init(sdk_path=sdk_path)
        hub = myo.Hub()
        listener = MyoListener()
        
        def hub_loop():
            while True:
                hub.run(listener, 20)
        
        threading.Thread(target=hub_loop, daemon=True).start()
        print("[INFO] Myo Hub iniciado. Esperando conexión...")
        
        # Esperar conexión
        timeout = 10
        start_time = time.time()
        while not listener.connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if listener.connected:
            return listener
        else:
            print("[WARNING] No se pudo conectar al Myo Armband en 10s")
            return None
            
    except Exception as e:
        print(f"[ERROR] No se pudo inicializar Myo: {e}")
        print("[INFO] El programa continuará sin datos de Myo")
        return None

# ==================== PROGRAMA PRINCIPAL ====================
if __name__ == "__main__":
    # Inicializar Myo
    myo_listener = initialize_myo()
    
    # Inicializar contadores
    counters = get_current_counters()
    
    # Variables de control
    current_mode = 'rock'
    capturing = False
    last_capture_time = 0
    capture_interval = 0.5
    
    print("\n" + "="*60)
    print("CREADOR DE DATASET SINCRONIZADO")
    print("MEDIAPIPE + MYO ARMBAND (EMG + IMU)")
    print("="*60)
    print("\nControles:")
    print("  1 - Modo ROCK")
    print("  2 - Modo PAPER")
    print("  3 - Modo SCISSORS")
    print("  4 - Modo NONE (sin gesto)")
    print("  ESPACIO - Iniciar/Pausar captura automática")
    print("  S - Capturar una muestra individual")
    print("  D - ELIMINAR TODO el dataset")
    print("  Q - Salir")
    print("\nDatos capturados por muestra:")
    print("  ✓ Imagen de la mano")
    print("  ✓ 21 landmarks de MediaPipe")
    print("  ✓ 8 canales EMG del Myo")
    print("  ✓ Datos IMU (orientación, aceleración, giroscopio)")
    print("="*60)
    
    if any(counters.values()):
        print("\n📊 Dataset existente detectado:")
        for gesture, count in counters.items():
            print(f"   {gesture.capitalize()}: {count} muestras")
        print(f"   TOTAL: {sum(counters.values())} muestras")
    
    print("\n" + "="*60)
    
    with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7,
                           min_tracking_confidence=0.7, max_num_hands=1) as hands:
        
        while True:
            ret, frame = cap.read()
            frame_original = frame.copy()
            frame_display = cv2.resize(frame, (640, 480))
            
            # Procesar con MediaPipe
            results = hands.process(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))
            
            hand_detected = False
            landmarks = []
            
            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    drawingModule.draw_landmarks(frame_display, handLandmarks, 
                                                handsModule.HAND_CONNECTIONS)
                    landmarks = get_landmarks_from_mediapipe(handLandmarks)
                    hand_detected = True
            
            # Captura automática
            current_time = time.time()
            if capturing and hand_detected and (current_time - last_capture_time) >= capture_interval:
                # Obtener datos del Myo
                if myo_listener and myo_listener.connected:
                    myo_data = myo_listener.get_snapshot()
                else:
                    # Datos vacíos si no hay Myo
                    myo_data = {
                        'emg_timestamps': [],
                        'emg_channels': [[] for _ in range(8)],
                        'imu_timestamps': [],
                        'orientation': [],
                        'accel': [],
                        'gyro': []
                    }
                
                # Guardar muestra
                counters[current_mode] += 1
                frame_clean = cv2.resize(frame_original, (640, 480))
                save_synchronized_sample(frame_clean, landmarks, myo_data, 
                                        current_mode, counters[current_mode])
                last_capture_time = current_time
            
            # Información en pantalla
            status_color = (0, 255, 0) if capturing else (0, 0, 255)
            status_text = "CAPTURANDO" if capturing else "PAUSADO"
            
            # Estado Myo
            myo_status = "✓ CONECTADO" if (myo_listener and myo_listener.connected) else "✗ DESCONECTADO"
            myo_color = (0, 255, 0) if (myo_listener and myo_listener.connected) else (0, 0, 255)
            
            cv2.putText(frame_display, f"Modo: {current_mode.upper()}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_display, f"Estado: {status_text}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame_display, f"Myo: {myo_status}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, myo_color, 2)
            
            # Contadores
            y_pos = 130
            for gesture, count in counters.items():
                cv2.putText(frame_display, f"{gesture.capitalize()}: {count}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 25
            
            if not hand_detected:
                cv2.putText(frame_display, "SIN MANO DETECTADA", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Dataset Creator - Sincronizado", frame_display)
            key = cv2.waitKey(1) & 0xFF
            
            # Controles de teclado
            if key == ord('1'):
                current_mode = 'rock'
                print(f"\n>>> Modo: ROCK")
            elif key == ord('2'):
                current_mode = 'paper'
                print(f"\n>>> Modo: PAPER")
            elif key == ord('3'):
                current_mode = 'scissors'
                print(f"\n>>> Modo: SCISSORS")
            elif key == ord('4'):
                current_mode = 'none'
                print(f"\n>>> Modo: NONE")
            elif key == ord(' '):
                capturing = not capturing
                status = "INICIADA" if capturing else "PAUSADA"
                print(f"\n>>> Captura {status}")
                if capturing and myo_listener:
                    myo_listener.reset_buffers()
            elif key == ord('s') or key == ord('S'):
                if hand_detected:
                    if myo_listener and myo_listener.connected:
                        myo_data = myo_listener.get_snapshot()
                    else:
                        myo_data = {
                            'emg_timestamps': [],
                            'emg_channels': [[] for _ in range(8)],
                            'imu_timestamps': [],
                            'orientation': [],
                            'accel': [],
                            'gyro': []
                        }
                    
                    counters[current_mode] += 1
                    frame_clean = cv2.resize(frame_original, (640, 480))
                    save_synchronized_sample(frame_clean, landmarks, myo_data,
                                            current_mode, counters[current_mode])
                else:
                    print("⚠ No hay mano detectada")
            elif key == ord('d') or key == ord('D'):
                capturing = False
                cv2.destroyAllWindows()
                delete_dataset()
                cv2.namedWindow("Dataset Creator - Sincronizado")
            elif key == ord('q') or key == ord('Q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("RESUMEN DEL DATASET FINAL")
    print("="*60)
    for gesture, count in counters.items():
        print(f"{gesture.capitalize()}: {count} muestras")
    print(f"TOTAL: {sum(counters.values())} muestras sincronizadas")
    print("\nArchivos guardados en:")
    print("  - dataset/images/[gesture]/     (imágenes sin landmarks)")
    print("  - dataset/landmarks/            (coordenadas 21 puntos + metadata)")
    print("  - dataset/emg/                  (8 canales EMG)")
    print("  - dataset/imu/                  (orientación, aceleración, giroscopio)")
    print("="*60)
