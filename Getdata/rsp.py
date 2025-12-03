# Import the necessary Packages for this software to run
import mediapipe
import cv2
import math

# Use MediaPipe to draw the hand framework over the top of hands it identifies in Real-Time
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

# Use CV2 Functionality to create a Video stream and add some values
cap = cv2.VideoCapture(0)

h = 480
w = 640

# Variable para almacenar el gesto anterior
previous_gesture = None


def calculate_distance(point1, point2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def is_finger_extended(landmarks, tip_id, pip_id, mcp_id, wrist_id=0):
    """
    Determina si un dedo está extendido usando múltiples criterios
    """
    tip = landmarks[tip_id][1:]
    pip = landmarks[pip_id][1:]
    mcp = landmarks[mcp_id][1:]
    wrist = landmarks[wrist_id][1:]
    
    # Distancia de la punta a la muñeca vs articulación PIP a la muñeca
    tip_to_wrist = calculate_distance(tip, wrist)
    pip_to_wrist = calculate_distance(pip, wrist)
    
    # Distancia de la punta a MCP vs PIP a MCP
    tip_to_mcp = calculate_distance(tip, mcp)
    pip_to_mcp = calculate_distance(pip, mcp)
    
    # El dedo está extendido si:
    # 1. La punta está más lejos de la muñeca que PIP
    # 2. La distancia punta-MCP es mayor que PIP-MCP (dedo estirado vs doblado)
    condition1 = tip_to_wrist > pip_to_wrist * 1.0
    condition2 = tip_to_mcp > pip_to_mcp * 0.9
    
    return condition1 and condition2


def is_thumb_extended(landmarks):
    """Detección especial para el pulgar"""
    thumb_tip = landmarks[4][1:]
    thumb_ip = landmarks[3][1:]
    thumb_mcp = landmarks[2][1:]
    wrist = landmarks[0][1:]
    
    # Distancia de la punta del pulgar a la muñeca
    tip_to_wrist = calculate_distance(thumb_tip, wrist)
    ip_to_wrist = calculate_distance(thumb_ip, wrist)
    
    # El pulgar está extendido si su punta está más lejos que su articulación IP
    return tip_to_wrist > ip_to_wrist * 1.1


def detect_gesture(landmarks):
    """Detecta el gesto basándose en los dedos levantados"""
    if len(landmarks) == 0:
        return "none"
    
    # Detectar cada dedo
    thumb_extended = is_thumb_extended(landmarks)
    index_extended = is_finger_extended(landmarks, 8, 6, 5)
    middle_extended = is_finger_extended(landmarks, 12, 10, 9)
    ring_extended = is_finger_extended(landmarks, 16, 14, 13)
    pinky_extended = is_finger_extended(landmarks, 20, 18, 17)
    
    # Contar dedos extendidos (sin el pulgar)
    fingers = [index_extended, middle_extended, ring_extended, pinky_extended]
    extended_count = sum(fingers)
    
    # ROCK: Ningún dedo extendido o máximo 1
    if extended_count == 0:
        return "rock"
    
    # SCISSORS: Solo índice y medio extendidos
    elif index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "scissors"
    
    # PAPER: 3 o 4 dedos extendidos
    elif extended_count >= 3:
        return "paper"
    
    # Si tiene 1-2 dedos pero no es tijera, considerarlo rock si están poco extendidos
    elif extended_count <= 2:
        # Verificar si realmente están bien extendidos o solo parcialmente
        wrist = landmarks[0][1:]
        
        # Calcular distancias promedio de dedos "extendidos"
        avg_distance = 0
        count = 0
        if index_extended:
            avg_distance += calculate_distance(landmarks[8][1:], wrist)
            count += 1
        if middle_extended:
            avg_distance += calculate_distance(landmarks[12][1:], wrist)
            count += 1
        if ring_extended:
            avg_distance += calculate_distance(landmarks[16][1:], wrist)
            count += 1
        if pinky_extended:
            avg_distance += calculate_distance(landmarks[20][1:], wrist)
            count += 1
        
        if count > 0:
            avg_distance /= count
            # Si la distancia promedio es baja, probablemente es rock
            palm_size = calculate_distance(landmarks[0][1:], landmarks[9][1:])
            if avg_distance < palm_size * 1.5:
                return "rock"
    
    return "none"


with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, 
                       min_tracking_confidence=0.7, max_num_hands=1) as hands:
    
    while True:
        ret, frame = cap.read()
        frame1 = cv2.resize(frame, (640, 480))
        
        # Process the hand framework overlay
        results = hands.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        
        current_gesture = "none"
        
        # If hand is detected
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)
                
                # Get landmarks positions
                landmarks = []
                for id, pt in enumerate(handLandmarks.landmark):
                    x = int(pt.x * w)
                    y = int(pt.y * h)
                    landmarks.append([id, x, y])
                
                # Detect gesture
                current_gesture = detect_gesture(landmarks)
        
        # Solo imprime si el gesto cambió
        if current_gesture != previous_gesture and current_gesture != "none":
            print(f"Gesto detectado: {current_gesture.upper()}")
            previous_gesture = current_gesture
        elif current_gesture == "none" and previous_gesture != "none":
            print("Ningún gesto reconocido")
            previous_gesture = "none"
        
        # Show the current frame
        cv2.imshow("Frame", frame1)
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to quit
        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()