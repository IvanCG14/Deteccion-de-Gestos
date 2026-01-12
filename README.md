# Detección de Gestos Multimodal (Rock, Paper, Scissors)

Este proyecto implementa un sistema de reconocimiento de gestos mediante **Deep Learning Multimodal**, fusionando visión artificial (RGB + Landmarks) y señales bioeléctricas (EMG + IMU) del brazalete **MYO Armband**.

---

## 🛠️ 1. Instalación del Entorno

Sigue estos pasos para configurar tu ambiente de desarrollo con soporte para GPU.

### Preparación Inicial

```bash
# en la carpeta de preferencia
git clone https://github.com/IvanCG14/Deteccion-de-Gestos.git
```

Navega al directorio `environment` que contiene los archivos de configuración:

```bash
cd ./environment/
```

Los archivos `environment.yml` y `requirements.txt` permiten reproducir el entorno completo.

### Opción A: Instalación automática (Recomendada)

Usa el fichero `environment.yml` para crear el entorno con todas las dependencias especificadas.

```bash
conda env create -f environment.yml
conda activate cti_env_gpu
```

### Opción B: Instalación manual (Paso a paso)

```bash
# 1. Crear entorno
conda create -n cti_env_gpu python=3.11.14 pip -y
conda activate cti_env_gpu

# 2. Instalar PyTorch con soporte CUDA 13.0
pip install torch==2.9.0+cu130 torchvision==0.24.0+cu130 --index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)

# 3. Instalar dependencias del proyecto
pip install -r requirements.txt
```

### Verificación de Hardware:

```bash
# Comprobar estado de la GPU
nvidia-smi

# Verificar PyTorch en Python
python -c "import torch; print(f'Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
```

---

## 📸 2. Generación de Datasets

En la carpeta `getdata/` se encuentran las herramientas necesarias para construir el dataset, permitiendo elegir entre un flujo de trabajo puramente visual o uno multimodal avanzado.

### 1. Dataset de 2 Modalidades (Básico)
Utiliza el script `getdata_rsp.py` para capturas basadas únicamente en visión artificial.
* **Ramas:** Imagen RGB y Marcadores 3D (Landmarks).
* **Tecnologías:** OpenCV y MediaPipe.
* **Uso:** Ideal para modelos que no requieren sensores externos.

### 2. Dataset de 4 Ramas (Multimodal - Myo Armband)
Utiliza el script `dataset_creator_myo.py` para una captura completa y sincronizada de bioseñales y visión.
* **Ramas:**
    1.  **RGB:** Imágenes de alta definición.
    2.  **Marcadores 3D:** Coordenadas espaciales de la mano.
    3.  **EMG:** 8 canales de actividad eléctrica muscular.
    4.  **IMU:** Datos inerciales (orientación, aceleración y giroscopio).
* **Sincronización:** El script gestiona hilos independientes para asegurar que los datos de los sensores coincidan exactamente con el frame capturado por la cámara, generando un archivo `metadata.json` como índice maestro.

### 📂 Recursos y Referencias
* **Scripts de captura:** [Carpeta getdata/](getdata/)
* **Dataset de ejemplo:** [Dataset Multimodal Sincronizado](https://1drv.ms/f/c/66c04837d2873fa4/IgCSyiKERBCESYB2pku-jSTYAdretsgtq320lxWYOVtWO4M?e=l4WRRh)

> **Nota:** Para el uso del sistema de 4 ramas, asegúrate de tener el SDK de Myo y el brazalete correctamente calibrado en el antebrazo.

---

## 🧠 3. Modelo Multimodal

Este proyecto implementa una arquitectura de **Deep Learning Multimodal** diseñada para el reconocimiento de gestos en tiempo real. El modelo utiliza una estrategia de **Fusión Tardía (Late Fusion)**, donde cada modalidad es procesada por una rama especializada antes de combinarse en una capa de clasificación común.

### Arquitectura de 4 Ramas
Basado en el núcleo de `model_training.ipynb`, el sistema integra:

* **Rama Visual (CNN):** Utiliza una **ResNet-18** (Transfer Learning) para extraer características espaciales de imágenes RGB redimensionadas a `128x128`.
* **Rama Esquelética (3D):** Un bloque de capas densas (MLP) que procesa los 21 landmarks (63 coordenadas) extraídos por MediaPipe.
* **Rama EMG (Bioseñales):** Procesa los 8 canales de electromiografía del brazalete Myo para detectar la intensidad de la contracción muscular.
* **Rama IMU (Inercial):** Analiza la orientación (cuaterniones), aceleración y velocidad angular para capturar la dinámica del movimiento.

### Características Principales

- 🚀 **Fusión Sincronizada**: El modelo procesa muestras donde la imagen y las señales de los sensores ocurren en la misma ventana temporal mediante el archivo `metadata.json`.
- 🧬 **Regularización con Mixup**: Implementa aumento de datos por mezcla lineal de muestras, lo que mejora drásticamente la generalización y reduce el overfitting.
- ⚖️ **Optimización Avanzada**: Uso de `CosineAnnealingLR` para un decaimiento suave de la tasa de aprendizaje y `Adam` como optimizador.
- 📊 **Evaluación Exhaustiva**: Generación automática de matrices de confusión y reportes de clasificación (Precision, Recall, F1) para cada gesto.
- 💾 **Gestión de Checkpoints**: El sistema monitorea el *Validation Loss* y guarda automáticamente el estado óptimo en `best_model_synchronized.pth`.

---

### Requisitos del Sistema

#### Hardware
- **Mínimo**: CPU (funcional pero lento ~2 min/epoch)
- **Recomendado**: GPU NVIDIA con CUDA (10x más rápido)
- **RAM**: 8GB mínimo, 16GB recomendado
- **Almacenamiento**: ~2GB para dataset + modelos

#### Software
- **Sistema Operativo**: Windows 10/11, Linux, macOS
- **Python**: 3.8 - 3.11 (recomendado 3.10)
- **CUDA** (opcional): 11.8+ para aceleración GPU

---

## Referencias

### Papers
- 

---

📧 Contacto e Investigación
Proyecto desarrollado para la investigación en interfaces hombre-máquina y fusión sensorial. 
**Licencia:** Open Source para fines educativos.



