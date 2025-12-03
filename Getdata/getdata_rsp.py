import mediapipe
import cv2
import math
import os
import csv
import time

# MediaPipe setup
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

# Configuración de la cámara
cap = cv2.VideoCapture(0)
h = 480
w = 640

# Crear carpetas para el dataset
os.makedirs('dataset/images/rock', exist_ok=True)
os.makedirs('dataset/images/paper', exist_ok=True)
os.makedirs('dataset/images/scissors', exist_ok=True)
os.makedirs('dataset/images/none', exist_ok=True)
os.makedirs('dataset/landmarks', exist_ok=True)

# Contador de imágenes por clase
counters = {
    'rock': 0,
    'paper': 0,
    'scissors': 0,
    'none': 0
}

# Modo actual (qué gesto estás capturando)
current_mode = 'rock'  # Cambia entre: 'rock', 'paper', 'scissors', 'none'

# Control de captura
capturing = False
last_capture_time = 0
capture_interval = 0.5  # Segundos entre capturas


def calculate_distance(point1, point2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def save_sample(frame, landmarks, gesture_label):
    """Guarda una muestra (imagen + landmarks) en el dataset"""
    global counters
    
    # Incrementar contador
    counters[gesture_label] += 1
    count = counters[gesture_label]
    
    # Guardar imagen
    img_filename = f'dataset/images/{gesture_label}/{gesture_label}_{count:04d}.jpg'
    cv2.imwrite(img_filename, frame)
    
    # Guardar landmarks en CSV
    csv_filename = f'dataset/landmarks/{gesture_label}_landmarks.csv'
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Escribir encabezado si es un archivo nuevo
        if not file_exists:
            header = ['image_file', 'label']
            for i in range(21):
                header.extend([f'x{i}', f'y{i}', f'z{i}'])
            writer.writerow(header)
        
        # Escribir datos
        row = [img_filename, gesture_label]
        for landmark in landmarks:
            row.extend([landmark[1], landmark[2], 0])  # x, y, z (z=0 en 2D)
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
print("  Q - Salir")
print("\nInstrucciones:")
print("1. Presiona el número del gesto que vas a mostrar")
print("2. Presiona ESPACIO para iniciar captura automática")
print("3. Mueve tu mano en diferentes posiciones y ángulos")
print("4. El sistema capturará imágenes cada 0.5 segundos")
print("="*60)

with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, 
                       min_tracking_confidence=0.7, max_num_hands=1) as hands:
    
    while True:
        ret, frame = cap.read()
        frame1 = cv2.resize(frame, (640, 480))
        
        # Procesar con MediaPipe
        results = hands.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        
        hand_detected = False
        landmarks = []
        
        # Detectar mano
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)
                landmarks = get_landmarks_from_mediapipe(handLandmarks)
                hand_detected = True
        
        # Captura automática
        current_time = time.time()
        if capturing and hand_detected and (current_time - last_capture_time) >= capture_interval:
            save_sample(frame1, landmarks, current_mode)
            last_capture_time = current_time
        
        # Mostrar información en pantalla
        status_color = (0, 255, 0) if capturing else (0, 0, 255)
        status_text = "CAPTURANDO" if capturing else "PAUSADO"
        
        cv2.putText(frame1, f"Modo: {current_mode.upper()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame1, f"Estado: {status_text}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame1, f"Rock: {counters['rock']}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame1, f"Paper: {counters['paper']}", (10, 125), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame1, f"Scissors: {counters['scissors']}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame1, f"None: {counters['none']}", (10, 175), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if not hand_detected:
            cv2.putText(frame1, "SIN MANO DETECTADA", (200, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar frame
        cv2.imshow("Dataset Creator", frame1)
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
        elif key == ord('s'):
            if hand_detected:
                save_sample(frame1, landmarks, current_mode)
            else:
                print("⚠ No hay mano detectada")
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("RESUMEN DEL DATASET CREADO")
print("="*60)
print(f"Rock:     {counters['rock']} imágenes")
print(f"Paper:    {counters['paper']} imágenes")
print(f"Scissors: {counters['scissors']} imágenes")
print(f"None:     {counters['none']} imágenes")
print(f"TOTAL:    {sum(counters.values())} imágenes")
print("\nArchivos guardados en:")
print("  - dataset/images/[rock|paper|scissors|none]/")
print("  - dataset/landmarks/[gestos]_landmarks.csv")
print("="*60)