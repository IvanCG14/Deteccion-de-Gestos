import mediapipe
import cv2
import math
import os
import csv
import time
import shutil
from pathlib import Path  # <-- Importamos Pathlib

# MediaPipe setup
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

# --- Configuración de Rutas con pathlib ---
DATASET_ROOT = Path('dataset')
IMAGES_ROOT = DATASET_ROOT / 'images'
LANDMARKS_ROOT = DATASET_ROOT / 'landmarks'

# Configuración de la cámara
cap = cv2.VideoCapture(0)
h = 480
w = 640

# Crear carpetas para el dataset
# Usamos mkdir(parents=True, exist_ok=True) para crear carpetas de forma segura
GESTURES = ['rock', 'paper', 'scissors', 'none']

for gesture in GESTURES:
    (IMAGES_ROOT / gesture).mkdir(parents=True, exist_ok=True)

LANDMARKS_ROOT.mkdir(parents=True, exist_ok=True)


# Leer contadores existentes del dataset
def get_current_counters():
    """Lee los contadores actuales del dataset existente"""
    counters = {gesture: 0 for gesture in GESTURES}
    
    # Contar archivos existentes en cada carpeta
    for gesture in counters.keys():
        folder = IMAGES_ROOT / gesture
        if folder.exists():
            # Usamos glob para encontrar archivos .jpg de forma segura
            files = [f for f in folder.glob('*.jpg')]
            if files:
                # Extraer el número más alto
                numbers = []
                for file_path in files:
                    # El nombre del archivo es una cadena que podemos dividir
                    filename = file_path.name
                    try:
                        # Asume el formato 'gesto_XXXX.jpg'
                        num = int(filename.split('_')[1].split('.')[0])
                        numbers.append(num)
                    except:
                        pass
                if numbers:
                    counters[gesture] = max(numbers)
    
    return counters

# Inicializar contadores con valores existentes
counters = get_current_counters()

# Modo actual (qué gesto estás capturando)
current_mode = 'rock'

# Control de captura
capturing = False
last_capture_time = 0
capture_interval = 0.5  # Segundos entre capturas


def calculate_distance(point1, point2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def save_sample(frame_clean, landmarks, gesture_label):
    """
    Guarda una muestra (imagen SIN landmarks + datos en CSV) en el dataset
    frame_clean: frame sin dibujar landmarks
    """
    global counters
    
    # Incrementar contador
    counters[gesture_label] += 1
    count = counters[gesture_label]
    
    # --- Guardar imagen SIN landmarks (Usando Path) ---
    img_filename_path = IMAGES_ROOT / gesture_label / f'{gesture_label}_{count:04d}.jpg'
    cv2.imwrite(str(img_filename_path), frame_clean) # cv2.imwrite requiere una cadena (str)
    
    # --- Guardar landmarks en CSV (Usando Path) ---
    csv_filename_path = LANDMARKS_ROOT / f'{gesture_label}_landmarks.csv'
    file_exists = csv_filename_path.exists()  # Usamos .exists()
    
    # Abrimos el archivo usando Path.open() en modo 'a' (append)
    with csv_filename_path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Escribir encabezado solo si el archivo no existe
        if not file_exists:
            header = ['image_file', 'label']
            for i in range(21):
                header.extend([f'x{i}', f'y{i}', f'z{i}'])
            writer.writerow(header)
        
        # Escribir datos
        # Guardamos la ruta relativa de la imagen como cadena
        row = [str(img_filename_path), gesture_label] 
        for landmark in landmarks:
            # x, y, z (z=0 en 2D) - Asumo que el valor z se mantiene en 0 para compatibilidad con el código original
            row.extend([landmark[1], landmark[2], 0]) 
        writer.writerow(row)
    
    print(f"✓ Guardado: {gesture_label} #{count}")


def get_landmarks_from_mediapipe(handLandmarks):
    """Extrae landmarks de MediaPipe"""
    landmarks = []
    for id, pt in enumerate(handLandmarks.landmark):
        x = int(pt.x * w)
        y = int(pt.y * h)
        landmarks.append([id, x, y])
    return landmarks


def delete_dataset():
    """Elimina completamente el dataset"""
    response = input("\n⚠️  ¿Estás seguro de que quieres ELIMINAR TODO el dataset? (sí/no): ")
    if response.lower() in ['sí', 'si', 'yes', 's', 'y']:
        try:
            # Usamos Path.exists() y shutil.rmtree para eliminación
            if DATASET_ROOT.exists():
                shutil.rmtree(DATASET_ROOT)
                print("\n✓ Dataset eliminado completamente")
                
                # Recrear carpetas vacías (reutilizamos la lógica de creación inicial)
                for gesture in GESTURES:
                    (IMAGES_ROOT / gesture).mkdir(parents=True, exist_ok=True)
                LANDMARKS_ROOT.mkdir(parents=True, exist_ok=True)
                
                # Resetear contadores
                global counters
                counters = {gesture: 0 for gesture in GESTURES}
                print("✓ Carpetas recreadas y contadores reseteados")
            else:
                print("\n⚠️  No existe dataset para eliminar")
        except Exception as e:
            print(f"\n❌ Error al eliminar dataset: {e}")
    else:
        print("\n✓ Operación cancelada")


print("="*60)
print("CREADOR DE DATASET - ROCK PAPER SCISSORS")
print("="*60)
print("\nControles:")
print("  1 - Modo ROCK")
print("  2 - Modo PAPER")
print("  3 - Modo SCISSORS")
print("  4 - Modo NONE (sin gesto)")
print("  ESPACIO - Iniciar/Pausar captura automática")
print("  S - Capturar una imagen individual")
print("  D - ELIMINAR TODO el dataset")
print("  Q - Salir")
print("\nInstrucciones:")
print("1. Presiona el número del gesto que vas a mostrar")
print("2. Presiona ESPACIO para iniciar captura automática")
print("3. Mueve tu mano en diferentes posiciones y ángulos")
print("4. El sistema capturará imágenes cada 0.5 segundos")
print("5. Las imágenes se guardan SIN landmarks visibles")
print("="*60)

# Mostrar contadores iniciales
if any(counters.values()):
    print("\n📊 Dataset existente detectado:")
    print(f"  Rock:     {counters['rock']} imágenes")
    print(f"  Paper:    {counters['paper']} imágenes")
    print(f"  Scissors: {counters['scissors']} imágenes")
    print(f"  None:     {counters['none']} imágenes")
    print(f"  TOTAL:    {sum(counters.values())} imágenes")
    print("  Las nuevas capturas se agregarán al dataset existente")
    print("="*60)

with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, 
                        min_tracking_confidence=0.7, max_num_hands=1) as hands:
    
    while True:
        ret, frame = cap.read()
        frame_original = frame.copy()  # Copia sin landmarks para guardar
        frame_display = cv2.resize(frame, (640, 480))
        
        # Procesar con MediaPipe
        results = hands.process(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))
        
        hand_detected = False
        landmarks = []
        
        # Detectar mano y dibujar landmarks SOLO en el frame de visualización
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                # Dibujar landmarks SOLO en frame_display (no en frame_original)
                drawingModule.draw_landmarks(frame_display, handLandmarks, handsModule.HAND_CONNECTIONS)
                landmarks = get_landmarks_from_mediapipe(handLandmarks)
                hand_detected = True
        
        # Captura automática
        current_time = time.time()
        if capturing and hand_detected and (current_time - last_capture_time) >= capture_interval:
            # Guardar el frame ORIGINAL (sin landmarks)
            frame_clean = cv2.resize(frame_original, (640, 480))
            save_sample(frame_clean, landmarks, current_mode)
            last_capture_time = current_time
        
        # Mostrar información en pantalla
        status_color = (0, 255, 0) if capturing else (0, 0, 255)
        status_text = "CAPTURANDO" if capturing else "PAUSADO"
        
        cv2.putText(frame_display, f"Modo: {current_mode.upper()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_display, f"Estado: {status_text}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame_display, f"Rock: {counters['rock']}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_display, f"Paper: {counters['paper']}", (10, 125), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_display, f"Scissors: {counters['scissors']}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_display, f"None: {counters['none']}", (10, 175), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if not hand_detected:
            cv2.putText(frame_display, "SIN MANO DETECTADA", (200, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar frame CON landmarks (solo visualización)
        cv2.imshow("Dataset Creator", frame_display)
        key = cv2.waitKey(1) & 0xFF
        
        # Controles de teclado
        if key == ord('1'):
            current_mode = 'rock'
            print(f"\n>>> Modo cambiado a: ROCK")
        elif key == ord('2'):
            current_mode = 'paper'
            print(f"\n>>> Modo cambiado a: PAPER")
        elif key == ord('3'):
            current_mode = 'scissors'
            print(f"\n>>> Modo cambiado a: SCISSORS")
        elif key == ord('4'):
            current_mode = 'none'
            print(f"\n>>> Modo cambiado a: NONE")
        elif key == ord(' '):
            capturing = not capturing
            status = "INICIADA" if capturing else "PAUSADA"
            print(f"\n>>> Captura {status}")
        elif key == ord('s') or key == ord('S'):
            if hand_detected:
                frame_clean = cv2.resize(frame_original, (640, 480))
                save_sample(frame_clean, landmarks, current_mode)
            else:
                print("⚠ No hay mano detectada")
        elif key == ord('d') or key == ord('D'):
            capturing = False  # Pausar captura antes de eliminar
            cv2.destroyAllWindows()  # Cerrar ventana
            delete_dataset()
            # Reabrir ventana
            cv2.namedWindow("Dataset Creator")
        elif key == ord('q') or key == ord('Q'):
            break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("RESUMEN DEL DATASET FINAL")
print("="*60)
print(f"Rock:     {counters['rock']} imágenes")
print(f"Paper:    {counters['paper']} imágenes")
print(f"Scissors: {counters['scissors']} imágenes")
print(f"None:     {counters['none']} imágenes")
print(f"TOTAL:    {sum(counters.values())} imágenes")
print("\nArchivos guardados en:")
print(f"  - {IMAGES_ROOT}/[rock|paper|scissors|none]/")
print("    (imágenes SIN landmarks visibles)")
print(f"  - {LANDMARKS_ROOT}/[gestos]_landmarks.csv")
print("    (coordenadas de los 21 puntos)")
print("="*60)