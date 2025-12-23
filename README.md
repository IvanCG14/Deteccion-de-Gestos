# Deteccion de Gestos

## Instlación de Environment

### Clonar/Descargar el proyecto

```bash
# en la carpeta de preferencia
git clone https://github.com/IvanCG14/Deteccion-de-Gestos.git
```

Navega al directorio `environment` que contiene los archivos de configuración:

```bash
cd ./environment/
```

Los archivos `environment.yml` y `requirements.txt` permiten reproducir el entorno completo.

### Opción 1 — Instalación automática (recomendada)
Usa el fichero `environment.yml` para crear el entorno con todas las dependencias especificadas.

1. Desde la carpeta que contiene `environment.yml`, ejecuta:

```bash
conda env create -f environment.yml
conda activate cti_env_gpu
```

2. Verifica la instalación de PyTorch y la disponibilidad de la GPU:

```bash
python -c "import torch; print(f'Torch: {torch.__version__} | CUDA disponible: {torch.cuda.is_available()}')"
```

Si `CUDA disponible` es `True`, la GPU está configurada correctamente.

---

### Opción 2 — Instalación manual
Si la instalación automática falla o prefieres controlar cada paso:

1. Crear y activar el entorno base:

```bash
conda create -n cti_env_gpu python=3.11.14 pip -y
conda activate cti_env_gpu
```

2. Instalar PyTorch compatible con CUDA 13.0 (rueda oficial):

```bash
pip install torch==2.9.0+cu130 torchvision==0.24.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

3. Instalar dependencias adicionales:

```bash
pip install -r requirements.txt
```

4. Verificación adicional (opcional):

```bash
# Comprobar versión de CUDA en el sistema
nvidia-smi

# Probar desde Python
python -c "import torch; print('Torch', torch.__version__); print('CUDA disponible:', torch.cuda.is_available()); print('Dispositivos CUDA:', torch.cuda.device_count())"
```

Si no dispone de GPU o prefieres instalar la versión CPU-only de PyTorch, usa:

```bash
pip install torch==2.9.0+cpu torchvision==0.24.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

---

## Creación de dataset

En la carpeta getdata se encuentra el script para generar un dataset de 2 modalidades:
- RGB
- Maracadores 3D

Trabaja con opencv y mediapipe para detectar el tipo de gesto y la posición de los marcadores. El script es el archivo llamado [getdata_rsp.py](getdata/getdata_rsp.py)

Link de dataset ejemplo: [Dataset_ejemplo](https://1drv.ms/f/c/66c04837d2873fa4/IgCSyiKERBCESYB2pku-jSTYAdretsgtq320lxWYOVtWO4M?e=l4WRRh)

## Modelo Multimodal

Este proyecto implementa un modelo de deep learning multimodal para reconocimiento de gestos de mano. Combina:
- **Modalidad Visual (RGB)**: Imágenes procesadas con ResNet-18 preentrenado
- **Modalidad Esquelética (3D)**: 21 landmarks de MediaPipe procesados con MLP
- **Modalidad EMG**: 8 canales EMG procesados con LTSM
- **Modalidad IMU**: Datos IMU de (orientación, aceleración, giroscopio)

Sistema de clasificación de gestos de mano (Rock, Paper, Scissors, None) usando Deep Learning multimodal que fusiona imágenes RGB y landmarks 3D de MediaPipe.

### Características

- ✅ **Modelo Multimodal**: Fusión de RGB + Skeleton 3D
- ✅ **Transfer Learning**: ResNet-18 preentrenado en ImageNet
- ✅ **Manejo de Desbalance**: Class weighting automático
- ✅ **Pipeline Completo**: Desde datos crudos hasta modelo entrenado
- ✅ **Visualizaciones**: Gráficas de entrenamiento y matriz de confusión
- ✅ **Checkpoints**: Guardado automático del mejor modelo
- ✅ **Reproducibilidad**: Seeds fijadas para resultados consistentes

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

### Código Base
- PyTorch: https://pytorch.org/
- TorchVision: https://pytorch.org/vision/
- MediaPipe: https://google.github.io/mediapipe/

---



