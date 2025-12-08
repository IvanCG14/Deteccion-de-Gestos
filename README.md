# Deteccion de Gestos

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
- He, K., et al. (2016). "Deep Residual Learning for Image Recognition". CVPR.
- Ioffe, S., & Szegedy, C. (2015). "Batch Normalization". ICML.
- Baltrusaitis, T., et al. (2018). "Multimodal Machine Learning: A Survey". IEEE TPAMI.

### Código Base
- PyTorch: https://pytorch.org/
- TorchVision: https://pytorch.org/vision/
- MediaPipe: https://google.github.io/mediapipe/

---



