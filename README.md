# Deteccion-de-Gestos

## Modelo Multimodal

Este repositorio contiene el código para un modelo de Deep Learning multimodal diseñado para la clasificación de gestos, utilizando dos fuentes de datos complementarias: imágenes RGB y datos vectoriales de landmarks (puntos clave de pose).

### 📁 Estructura del dataset esperada

El dataset debe estar organizado en carpetas:
```
 dataset/                  
    ├── images/           
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

### ⚙️ Dependencias

```python
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
```

Para instalarlas:
```nginx
pip install torch torchvision seaborn tqdm pillow scikit-learn matplotlib
```

### 📦 Clases y Componentes Principales

🔹 1. Custom Dataset – RPSDataset

Lee imágenes desde directorios y devuelve:
- image transformada (resize, normalización)
- label numérico

Las transformaciones incluyen:
- Resize (224×224)
- ToTensor
- Normalization

🔹 2. Modelo – ResNet18 Frozen

El modelo base utilizado es:
```
models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
```

Las capas convolucionales están congeladas, solo se entrena:
- fc → capa final completamente conectada

Se reemplaza fc para predecir 3 clases:
```
model.fc = nn.Linear(model.fc.in_features, 3)
```
🔹 3. Entrenamiento

El entrenamiento incluye:
- Optimizer: Adam
- Scheduler: CosineAnnealingLR
- Loss: CrossEntropyLoss
- DataLoader con batchs y shuffling

Se ejecuta en GPU si está disponible.

🔹 4. Métricas y gráficos

El Notebook genera:
- Curva de entrenamiento y validación
- Confusion Matrix
- Classification Report (precision, recall, f1-score)
- Ejemplos de predicciones

### ▶️ Cómo entrenar el modelo

1. Asegúrate de tener el dataset en el formato esperado.
2. Define la ruta del dataset:
```
root_dir = "ruta/a/tu/dataset"
```
3. Ejecuta todas las celdas del notebook.

El entrenamiento iniciará y verás una barra de progreso de tqdm.

### 📊 Resultados

El notebook muestra:
- Precisión por clase
- Accuracy general
- Matriz de confusión
- Pérdidas por época

Esto permite evaluar si el modelo está clasificando correctamente cada gesto.

### 💾 Guardado del modelo

El modelo final se guarda normalmente como:
```
best_model.pth
```
Y puede cargarse después para inferencia.

### 🤖 Uso del modelo entrenado
```python
model = torch.load("model.pth")
model.eval()

img = Image.open("mi_imagen.jpg")
tensor = transform(img).unsqueeze(0)
pred = model(tensor)
print(torch.argmax(pred))
```
---

## Data augmentation

El Aumento de Datos previene el sobreajuste (overfitting) al simular variaciones del mundo real y hacer el modelo más robusto a cambios en la captura (iluminación, ángulo, tamaño).

![Augmentation](imgs/dataset_samples.png)

### Transformaciones Aplicadas (Rama RGB)

#### Transformación: Propósito
- Resize + RandomCrop: Simula variaciones en el zoom y la posición del gesto.
- RandomHorizontalFlip: Enseña a reconocer el gesto independientemente de la lateralidad (mano izquierda/derecha).
- ColorJitter:"Simula cambios en las condiciones de iluminación (brillo, contraste, saturación)."
- RandomRotation: Acepta ligeros cambios en el ángulo o inclinación de la cámara/mano.
- Normalize: Estandariza la imagen con los valores de ImageNet para compatibilidad con ResNet-18.


