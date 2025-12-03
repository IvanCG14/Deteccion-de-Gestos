# Deteccion-de-Gestos

## Creación de dataset
**Paso 1: Preparación**
1. Ejecuta el script
2. Verás la ventana de la cámara con información en pantalla

**Paso 2: Captura de cada gesto**

*Para ROCK (Piedra):*
1. Presiona `1` (modo rock)
2. Cierra el puño
3. Presiona `ESPACIO` para iniciar captura automática
4. Mueve tu mano en diferentes:
   - Ángulos (horizontal, vertical, diagonal)
   - Posiciones (cerca, lejos, izquierda, derecha)
   - Rotaciones
5. Deja que capture ~100-200 imágenes
6. Presiona `ESPACIO` para pausar

*Para PAPER (Papel):*
1. Presiona `2` (modo paper)
2. Abre todos los dedos
3. Presiona `ESPACIO`
4. Repite variaciones
5. Captura ~100-200 imágenes

*Para SCISSORS (Tijera):*
1. Presiona `3` (modo scissors)
2. Extiende índice y medio
3. Presiona `ESPACIO`
4. Repite variaciones
5. Captura ~100-200 imágenes

*Para NONE (Sin gesto):*
1. Presiona `4` (modo none)
2. Haz gestos raros, dedos parcialmente extendidos, etc.
3. Captura ~50-100 imágenes

**Paso 3: Captura manual**
- Si quieres capturar solo una imagen específica, presiona `S`

**Paso 4: Finalizar**
- Presiona `Q` para salir
- Verás un resumen de cuántas imágenes capturaste

## Data augmentation

## Modelo Multimodal
