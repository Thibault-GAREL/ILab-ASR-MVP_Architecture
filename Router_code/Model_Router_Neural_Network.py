"""
Model Router Neural Network
Prend en entrée les log mel spectrogrammes et les métadonnées audio
pour prédire quel modèle ASR utiliser
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from pathlib import Path


class SpectrogramEncoder(nn.Module):
    """
    Encodeur CNN pour les log mel spectrogrammes
    Extrait des features à partir des spectrogrammes (n_mels, time)
    """

    def __init__(
            self,
            n_mels: int = 80,
            hidden_dims: list = [64, 128, 256],
            dropout: float = 0.3
    ):
        super().__init__()

        self.n_mels = n_mels

        # Couches convolutionnelles
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, hidden_dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Dropout2d(dropout)
        )

        self.output_dim = hidden_dims[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_mels, time)

        Returns:
            features: (batch, output_dim)
        """
        # Ajouter dimension channel
        x = x.unsqueeze(1)  # (batch, 1, n_mels, time)

        # Convolutions
        x = self.conv1(x)  # (batch, 64, n_mels/2, time/2)
        x = self.conv2(x)  # (batch, 128, n_mels/4, time/4)
        x = self.conv3(x)  # (batch, 256, 1, 1)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 256)

        return x


class MetadataEncoder(nn.Module):
    """
    Encodeur MLP pour les métadonnées audio
    (SNR, duration, RMS energy, zero crossing rate, spectral centroid)
    """

    def __init__(
            self,
            input_dim: int = 5,
            hidden_dims: list = [32, 64],
            dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 5) - [snr, duration, rms_energy, zcr, spectral_centroid]

        Returns:
            features: (batch, output_dim)
        """
        return self.mlp(x)


class ModelRouter(nn.Module):
    """
    Model Router complet
    Combine spectrogramme et métadonnées pour prédire le meilleur modèle ASR
    """

    def __init__(
            self,
            n_mels: int = 80,
            n_models: int = 3,  # Nombre de modèles ASR disponibles
            spec_hidden_dims: list = [64, 128, 256],
            meta_hidden_dims: list = [32, 64],
            fusion_hidden_dim: int = 128,
            dropout: float = 0.3
    ):
        """
        Args:
            n_mels: Nombre de filtres mel
            n_models: Nombre de modèles ASR à choisir
            spec_hidden_dims: Dimensions cachées pour l'encodeur de spectrogramme
            meta_hidden_dims: Dimensions cachées pour l'encodeur de métadonnées
            fusion_hidden_dim: Dimension de la couche de fusion
            dropout: Taux de dropout
        """
        super().__init__()

        self.n_models = n_models

        # Encodeurs
        self.spec_encoder = SpectrogramEncoder(
            n_mels=n_mels,
            hidden_dims=spec_hidden_dims,
            dropout=dropout
        )

        self.meta_encoder = MetadataEncoder(
            input_dim=5,
            hidden_dims=meta_hidden_dims,
            dropout=dropout
        )

        # Couche de fusion
        fusion_input_dim = self.spec_encoder.output_dim + self.meta_encoder.output_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_hidden_dim // 2),
            nn.Dropout(dropout)
        )

        # Tête de classification
        self.classifier = nn.Linear(fusion_hidden_dim // 2, n_models)

    def forward(
            self,
            log_mel_specs: torch.Tensor,
            metadata: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            log_mel_specs: (batch, n_mels, time)
            metadata: (batch, 5)

        Returns:
            logits: (batch, n_models) - scores bruts pour chaque modèle
            probs: (batch, n_models) - probabilités softmax
        """
        # Encoder spectrogramme
        spec_features = self.spec_encoder(log_mel_specs)  # (batch, 256)

        # Encoder métadonnées
        meta_features = self.meta_encoder(metadata)  # (batch, 64)

        # Concaténer les features
        combined = torch.cat([spec_features, meta_features], dim=1)  # (batch, 320)

        # Fusion
        fused = self.fusion(combined)  # (batch, 64)

        # Classification
        logits = self.classifier(fused)  # (batch, n_models)
        probs = F.softmax(logits, dim=1)  # (batch, n_models)

        return logits, probs

    def predict(
            self,
            log_mel_specs: torch.Tensor,
            metadata: torch.Tensor
    ) -> torch.Tensor:
        """
        Prédit le modèle à utiliser (classe avec probabilité max)

        Args:
            log_mel_specs: (batch, n_mels, time)
            metadata: (batch, 5)

        Returns:
            predictions: (batch,) - indices des modèles prédits
        """
        self.eval()
        with torch.no_grad():
            _, probs = self.forward(log_mel_specs, metadata)
            predictions = torch.argmax(probs, dim=1)
        return predictions

    def predict_with_confidence(
            self,
            log_mel_specs: torch.Tensor,
            metadata: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prédit le modèle avec score de confiance

        Args:
            log_mel_specs: (batch, n_mels, time)
            metadata: (batch, 5)

        Returns:
            predictions: (batch,) - indices des modèles prédits
            confidences: (batch,) - scores de confiance (proba max)
        """
        self.eval()
        with torch.no_grad():
            _, probs = self.forward(log_mel_specs, metadata)
            confidences, predictions = torch.max(probs, dim=1)
        return predictions, confidences


class ModelRouterTrainer:
    """
    Trainer pour le Model Router
    """

    def __init__(
            self,
            model: ModelRouter,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.criterion = nn.CrossEntropyLoss()

        # Historique
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train_step(
            self,
            log_mel_specs: torch.Tensor,
            metadata: torch.Tensor,
            labels: torch.Tensor
    ) -> float:
        """
        Un pas d'entraînement

        Args:
            log_mel_specs: (batch, n_mels, time)
            metadata: (batch, 5)
            labels: (batch,) - indices des vrais modèles

        Returns:
            loss: float
        """
        self.model.train()

        # Move to device
        log_mel_specs = log_mel_specs.to(self.device)
        metadata = metadata.to(self.device)
        labels = labels.to(self.device)

        # Forward
        logits, _ = self.model(log_mel_specs, metadata)
        loss = self.criterion(logits, labels)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(
            self,
            val_loader
    ) -> Tuple[float, float]:
        """
        Validation sur un dataloader

        Returns:
            avg_loss: float
            accuracy: float
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                log_mel_specs = batch['log_mel_specs'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits, probs = self.model(log_mel_specs, metadata)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * log_mel_specs.size(0)

                predictions = torch.argmax(probs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def save_checkpoint(self, path: str):
        """Sauvegarde le modèle"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }, path)
        print(f"✅ Modèle sauvegardé: {path}")

    def load_checkpoint(self, path: str):
        """Charge le modèle"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        print(f"✅ Modèle chargé: {path}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer le model router
    model = ModelRouter(
        n_mels=80,
        n_models=3,  # Par exemple: Whisper-tiny, Whisper-base, Whisper-large
        spec_hidden_dims=[64, 128, 256],
        meta_hidden_dims=[32, 64],
        fusion_hidden_dim=128,
        dropout=0.3
    )

    print("🧠 Model Router créé")
    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    n_mels = 80
    time_frames = 1500  # ~15s à 16kHz avec hop_length=160

    dummy_specs = torch.randn(batch_size, n_mels, time_frames)
    dummy_metadata = torch.randn(batch_size, 5)

    logits, probs = model(dummy_specs, dummy_metadata)

    print(f"\n📊 Test forward pass:")
    print(f"  Input specs: {dummy_specs.shape}")
    print(f"  Input metadata: {dummy_metadata.shape}")
    print(f"  Output logits: {logits.shape}")
    print(f"  Output probs: {probs.shape}")
    print(f"  Probs sum: {probs.sum(dim=1)}")  # Doit être ~1.0

    # Test prediction
    predictions, confidences = model.predict_with_confidence(dummy_specs, dummy_metadata)
    print(f"\n🎯 Prédictions:")
    print(f"  Modèles prédits: {predictions}")
    print(f"  Confiances: {confidences}")

    # Créer un trainer
    trainer = ModelRouterTrainer(
        model=model,
        learning_rate=1e-3,
        device="cpu"
    )

    # Test training step
    dummy_labels = torch.randint(0, 3, (batch_size,))
    loss = trainer.train_step(dummy_specs, dummy_metadata, dummy_labels)
    print(f"\n🏋️ Test training step:")
    print(f"  Loss: {loss:.4f}")