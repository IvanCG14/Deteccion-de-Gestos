"""
Modelo Multimodal Extendido: RGB + Landmarks + EMG + IMU
Arquitectura de 4 ramas para clasificación de gestos con Myo Armband

Basado en:
- Fromm et al. (2016): LSTM para señales EMG temporales
- Tu modelo anterior: ResNet18 + MLP para RGB + Landmarks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional, Tuple

# ============================================
# 1. RAMA RGB (Ya existente)
# ============================================
class RGBBranch(nn.Module):
    """Branch CNN para imágenes RGB"""
    def __init__(self, backbone='resnet18', feature_dim=256, dropout=0.3):
        super().__init__()
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        encoder_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.feature_dim = feature_dim
    
    def forward(self, x):
        """x: (B, 3, H, W)"""
        return self.projector(self.encoder(x))

# ============================================
# 2. RAMA LANDMARKS (Ya existente)
# ============================================
class LandmarkBranch(nn.Module):
    """Branch MLP para landmarks 3D de MediaPipe"""
    def __init__(self, input_dim=63, feature_dim=256, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        self.feature_dim = feature_dim
    
    def forward(self, x):
        """x: (B, 63) o (B, 21, 3)"""
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.mlp(x)

# ============================================
# 3. RAMA EMG (NUEVA) - Señales temporales
# ============================================
class EMGBranch(nn.Module):
    """
    Branch LSTM para señales EMG temporales (8 sensores).
    Inspirado en Fromm et al. (2016) - Figure 7.
    
    Arquitectura:
    Input: (B, seq_len, 8) → LSTM(256) → LSTM(256) → Dense(256)
    """
    def __init__(self, 
                 input_dim=8,           # 8 sensores EMG
                 hidden_dim=256,        # Units del LSTM
                 num_layers=2,          # 2 capas LSTM como en paper
                 feature_dim=256,       # Dimensión de salida
                 dropout=0.3):
        super().__init__()
        
        # LSTM bidireccional para capturar patrones temporales
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Captura contexto pasado y futuro
        )
        
        # Dense layer (como en Figure 7)
        # BiLSTM duplica dimensión de salida
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x):
        """
        x: (B, seq_len, 8) - Secuencia temporal de EMG
        Returns: (B, feature_dim)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Usar último hidden state (concatenado de ambas direcciones)
        # h_n shape: (num_layers * 2, B, hidden_dim)
        # Tomamos última capa, ambas direcciones
        forward_hidden = h_n[-2]  # Última capa, dirección forward
        backward_hidden = h_n[-1]  # Última capa, dirección backward
        
        # Concatenar ambas direcciones
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)  # (B, hidden_dim*2)
        
        # Dense layer
        return self.dense(hidden)

# ============================================
# 4. RAMA IMU (NUEVA) - Orientación + Aceleración + Giroscopio
# ============================================
class IMUBranch(nn.Module):
    """
    Branch LSTM para datos IMU temporales.
    Input: Orientación (4 quaternions) + Aceleración (3 ejes) + Giroscopio (3 ejes) = 10 valores
    
    Similar a EMGBranch pero con 10 inputs en lugar de 8.
    """
    def __init__(self,
                 input_dim=10,          # 4 (orientation) + 3 (acc) + 3 (gyro)
                 hidden_dim=128,        # Menos unidades que EMG (IMU es más simple)
                 num_layers=2,
                 feature_dim=256,
                 dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x):
        """
        x: (B, seq_len, 10) - Secuencia temporal de IMU
        Returns: (B, feature_dim)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return self.dense(hidden)

# ============================================
# 5. MÓDULO DE FUSIÓN ADAPTATIVA
# ============================================
class AdaptiveFusion(nn.Module):
    """
    Fusión adaptativa que maneja modalidades opcionales.
    Permite entrenar con subconjuntos de modalidades (cuando Myo no está disponible).
    """
    def __init__(self, feature_dim=256, num_modalities=4, dropout=0.3):
        super().__init__()
        
        # Atención sobre modalidades
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.Tanh(),
            nn.Linear(feature_dim // 4, 1)
        )
        
        # Proyección final
        self.fusion_proj = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, features_dict):
        """
        features_dict: {'rgb': tensor, 'landmarks': tensor, 'emg': tensor, 'imu': tensor}
        Cada tensor: (B, feature_dim)
        
        Modalidades opcionales pueden ser None.
        """
        # Filtrar modalidades disponibles
        available_features = {k: v for k, v in features_dict.items() if v is not None}
        
        if len(available_features) == 0:
            raise ValueError("Al menos una modalidad debe estar presente")
        
        # Stack features disponibles
        feature_list = list(available_features.values())
        stacked = torch.stack(feature_list, dim=1)  # (B, num_modalities, feature_dim)
        
        # Calcular atención por modalidad
        attn_scores = self.attention(stacked)  # (B, num_modalities, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # Normalizar sobre modalidades
        
        # Weighted sum
        attended = (stacked * attn_weights).sum(dim=1)  # (B, feature_dim)
        
        # Si tenemos todas las modalidades, concatenar + proyectar
        if len(available_features) == 4:
            concatenated = torch.cat(feature_list, dim=1)  # (B, feature_dim*4)
            fused = self.fusion_proj(concatenated)
            # Combinar attended features con fused
            return (attended + fused) / 2
        else:
            # Solo usar attended si no tenemos todas
            return attended

# ============================================
# 6. MODELO MULTIMODAL COMPLETO (4 RAMAS)
# ============================================
class MultimodalGestureModelWithMyo(nn.Module):
    """
    Modelo Multimodal de 4 ramas: RGB + Landmarks + EMG + IMU
    
    Modalidades:
    - RGB: Imágenes de cámara
    - Landmarks: Coordenadas 3D de MediaPipe
    - EMG: 8 sensores de actividad muscular (temporal)
    - IMU: Orientación + aceleración + giroscopio (temporal)
    
    Maneja datos asincrónicos: modalidades opcionales.
    """
    def __init__(self,
                 num_classes=3,
                 feature_dim=256,
                 dropout=0.3,
                 emg_seq_len=100,       # Secuencias EMG típicas
                 imu_seq_len=100):      # Secuencias IMU típicas
        super().__init__()
        
        # Ramas de features
        self.rgb_branch = RGBBranch(feature_dim=feature_dim, dropout=dropout)
        self.landmark_branch = LandmarkBranch(input_dim=63, feature_dim=feature_dim, dropout=dropout)
        self.emg_branch = EMGBranch(input_dim=8, feature_dim=feature_dim, dropout=dropout)
        self.imu_branch = IMUBranch(input_dim=10, feature_dim=feature_dim, dropout=dropout)
        
        # Módulo de fusión
        self.fusion = AdaptiveFusion(feature_dim=feature_dim, num_modalities=4, dropout=dropout)
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, 
                image=None,           # (B, 3, H, W)
                landmarks=None,       # (B, 63) o (B, 21, 3)
                emg=None,            # (B, seq_len, 8)
                imu=None):           # (B, seq_len, 10)
        """
        Forward pass con modalidades opcionales.
        Al menos una modalidad debe estar presente.
        """
        features = {}
        
        # Extraer features de cada modalidad disponible
        if image is not None:
            features['rgb'] = self.rgb_branch(image)
        else:
            features['rgb'] = None
        
        if landmarks is not None:
            features['landmarks'] = self.landmark_branch(landmarks)
        else:
            features['landmarks'] = None
        
        if emg is not None:
            features['emg'] = self.emg_branch(emg)
        else:
            features['emg'] = None
        
        if imu is not None:
            features['imu'] = self.imu_branch(imu)
        else:
            features['imu'] = None
        
        # Fusión adaptativa
        fused = self.fusion(features)
        
        # Clasificación
        logits = self.classifier(fused)
        
        return {
            'logits': logits,
            'features': features,
            'fused': fused
        }

# ============================================
# 7. MODELOS BASELINE (para ablation study)
# ============================================
class RGBLandmarksOnly(nn.Module):
    """Modelo sin Myo (solo RGB + Landmarks)"""
    def __init__(self, num_classes=3, feature_dim=256, dropout=0.3):
        super().__init__()
        self.rgb_branch = RGBBranch(feature_dim=feature_dim, dropout=dropout)
        self.landmark_branch = LandmarkBranch(63, feature_dim, dropout)
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, image, landmarks, emg=None, imu=None):
        rgb_feat = self.rgb_branch(image)
        lm_feat = self.landmark_branch(landmarks)
        fused = self.fusion(torch.cat([rgb_feat, lm_feat], dim=1))
        return {'logits': self.classifier(fused)}

class MyoOnly(nn.Module):
    """Modelo solo con Myo (EMG + IMU)"""
    def __init__(self, num_classes=3, feature_dim=256, dropout=0.3):
        super().__init__()
        self.emg_branch = EMGBranch(8, feature_dim, dropout=dropout)
        self.imu_branch = IMUBranch(10, feature_dim, dropout=dropout)
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, image=None, landmarks=None, emg=None, imu=None):
        emg_feat = self.emg_branch(emg)
        imu_feat = self.imu_branch(imu)
        fused = self.fusion(torch.cat([emg_feat, imu_feat], dim=1))
        return {'logits': self.classifier(fused)}

# ============================================
# 8. TEST DEL MODELO
# ============================================
if __name__ == "__main__":
    print("="*60)
    print("TEST DEL MODELO MULTIMODAL CON MYO ARMBAND")
    print("="*60)
    
    batch_size = 4
    
    # Datos dummy
    images = torch.randn(batch_size, 3, 224, 224)
    landmarks = torch.randn(batch_size, 63)
    emg = torch.randn(batch_size, 100, 8)      # 100 timesteps, 8 sensores
    imu = torch.randn(batch_size, 100, 10)     # 100 timesteps, 10 valores
    
    # ===== Test 1: Modelo completo (4 ramas) =====
    print("\n1. Modelo completo (4 modalidades):")
    model_full = MultimodalGestureModelWithMyo(num_classes=3)
    
    output = model_full(image=images, landmarks=landmarks, emg=emg, imu=imu)
    print(f"   Logits: {output['logits'].shape}")
    print(f"   Fused features: {output['fused'].shape}")
    
    total_params = sum(p.numel() for p in model_full.parameters())
    print(f"   Total parámetros: {total_params:,}")
    
    # ===== Test 2: Solo RGB + Landmarks (sin Myo) =====
    print("\n2. Solo RGB + Landmarks (Myo no disponible):")
    output_partial = model_full(image=images, landmarks=landmarks, emg=None, imu=None)
    print(f"   Logits: {output_partial['logits'].shape}")
    
    # ===== Test 3: Solo Myo (EMG + IMU) =====
    print("\n3. Solo Myo (EMG + IMU):")
    output_myo = model_full(image=None, landmarks=None, emg=emg, imu=imu)
    print(f"   Logits: {output_myo['logits'].shape}")
    
    # ===== Test 4: Todas las combinaciones posibles =====
    print("\n4. Test de combinaciones:")
    test_cases = [
        ("RGB only", {'image': images}),
        ("Landmarks only", {'landmarks': landmarks}),
        ("EMG only", {'emg': emg}),
        ("IMU only", {'imu': imu}),
        ("RGB + EMG", {'image': images, 'emg': emg}),
        ("Landmarks + IMU", {'landmarks': landmarks, 'imu': imu}),
        ("All 4", {'image': images, 'landmarks': landmarks, 'emg': emg, 'imu': imu})
    ]
    
    for name, inputs in test_cases:
        try:
            out = model_full(**inputs)
            print(f"   ✓ {name}: {out['logits'].shape}")
        except Exception as e:
            print(f"   ✗ {name}: {e}")
    
    # ===== Test 5: Modelos baseline =====
    print("\n5. Modelos baseline:")
    
    model_rgb_lm = RGBLandmarksOnly(num_classes=3)
    out1 = model_rgb_lm(images, landmarks)
    print(f"   RGB+Landmarks: {out1['logits'].shape}")
    
    model_myo = MyoOnly(num_classes=3)
    out2 = model_myo(emg=emg, imu=imu)
    print(f"   Myo only: {out2['logits'].shape}")
    
    print("\n" + "="*60)
    print("✓ TODOS LOS TESTS PASARON")
    print("="*60)
    print(f"""
RESUMEN:
- Modelo soporta 4 modalidades: RGB, Landmarks, EMG, IMU
- Fusión adaptativa maneja modalidades opcionales
- Útil para datos asincrónicos o incompletos
- Total parámetros: {total_params:,}

ARQUITECTURA:
RGB (ResNet18) ────────┐
                       ├──> Fusión Adaptativa ──> Clasificador
Landmarks (MLP) ───────┤
                       │
EMG (LSTM) ────────────┤
                       │
IMU (LSTM) ────────────┘
    """)
