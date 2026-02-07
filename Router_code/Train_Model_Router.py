"""
Script d'entraînement du Model Router
Charge les données prétraitées depuis data_chew/ et entraîne le réseau
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Importer vos modules
from Model_Router_Neural_Network import ModelRouter, ModelRouterTrainer
from Process_audio import AudioPreprocessor


class SpectrogramDataset(Dataset):
    """
    Dataset PyTorch pour charger les spectrogrammes et métadonnées
    depuis data_chew/
    """

    def __init__(
            self,
            data_dir: str = "data_chew",
            labels_file: Optional[str] = None,
    ):
        """
        Args:
            data_dir: Dossier contenant spectrograms/ et metadata/
            labels_file: Fichier JSON avec les labels {filename: model_id}
                        Si None, utilise des labels aléatoires (pour test)
        """
        self.data_dir = Path(data_dir)
        self.spec_dir = self.data_dir / "spectrograms"
        self.meta_dir = self.data_dir / "metadata"

        # Charger tous les fichiers disponibles
        self.spec_files = sorted(list(self.spec_dir.glob("*.pt")))

        if len(self.spec_files) == 0:
            raise ValueError(f"Aucun spectrogramme trouvé dans {self.spec_dir}")

        print(f"📊 Dataset chargé: {len(self.spec_files)} spectrogrammes")

        # Charger les labels
        self.labels = self._load_labels(labels_file)

    def _load_labels(self, labels_file: Optional[str]) -> Dict[str, int]:
        """
        Charge les labels depuis un fichier JSON
        Format attendu: {"filename_seg000": 0, "filename_seg001": 1, ...}

        Si labels_file est None, génère des labels aléatoires pour test
        """
        if labels_file is None or not Path(labels_file).exists():
            print("⚠️  Pas de fichier labels fourni, génération de labels aléatoires")
            # Labels aléatoires pour chaque fichier (0, 1, ou 2)
            labels = {}
            for spec_file in self.spec_files:
                labels[spec_file.stem] = np.random.randint(0, 3)
            return labels

        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)

        print(f"✅ Labels chargés depuis {labels_file}")
        return labels

    def __len__(self) -> int:
        return len(self.spec_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retourne un item du dataset

        Returns:
            Dict avec:
                - log_mel_specs: (n_mels, time)
                - metadata: (5,) [snr, duration, rms, zcr, centroid]
                - labels: (1,) classe du modèle à utiliser
        """
        spec_file = self.spec_files[idx]
        filename = spec_file.stem

        # Charger le spectrogramme
        log_mel_spec = torch.load(spec_file)

        # Charger les métadonnées
        meta_file = self.meta_dir / f"{filename}.json"
        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)

        # Extraire les features métadonnées (5 features)
        metadata = torch.tensor([
            metadata_dict['snr'],
            metadata_dict['duration'],
            metadata_dict['rms_energy'],
            metadata_dict['zero_crossing_rate'],
            metadata_dict['spectral_centroid']
        ], dtype=torch.float32)

        # Label
        label = self.labels.get(filename, 0)  # Default 0 si pas trouvé
        label = torch.tensor(label, dtype=torch.long)

        return {
            'log_mel_specs': log_mel_spec,
            'metadata': metadata,
            'labels': label
        }


def create_dataloaders(
        data_dir: str = "data_chew",
        labels_file: Optional[str] = None,
        train_split: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 4,
        seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Crée les dataloaders train et validation

    Args:
        data_dir: Dossier contenant les données
        labels_file: Fichier JSON avec les labels
        train_split: Proportion pour l'entraînement
        batch_size: Taille des batches
        num_workers: Nombre de workers pour le chargement
        seed: Random seed pour la reproductibilité

    Returns:
        train_loader, val_loader
    """
    # Créer le dataset complet
    dataset = SpectrogramDataset(data_dir, labels_file)

    # Split train/val
    torch.manual_seed(seed)
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"📦 Train: {train_size} samples | Val: {val_size} samples")

    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def train_epoch(
        trainer: ModelRouterTrainer,
        train_loader: DataLoader,
        epoch: int
) -> float:
    """
    Entraîne une epoch

    Returns:
        avg_loss: Loss moyenne sur l'epoch
    """
    trainer.model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        loss = trainer.train_step(
            batch['log_mel_specs'],
            batch['metadata'],
            batch['labels']
        )

        total_loss += loss
        pbar.set_postfix({'loss': f'{loss:.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def train_model(
        model: ModelRouter,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "checkpoints",
        early_stopping_patience: int = 10
):
    """
    Entraîne le model router

    Args:
        model: Model Router à entraîner
        train_loader: DataLoader d'entraînement
        val_loader: DataLoader de validation
        num_epochs: Nombre d'epochs
        learning_rate: Learning rate
        device: Device (cuda/cpu)
        save_dir: Dossier pour sauvegarder les checkpoints
        early_stopping_patience: Nombre d'epochs sans amélioration avant arrêt
    """
    # Créer le dossier de sauvegarde
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Initialiser le trainer
    trainer = ModelRouterTrainer(
        model=model,
        learning_rate=learning_rate,
        device=device
    )

    print(f"\n🚀 Début de l'entraînement")
    print(f"   Device: {device}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Batch size: {train_loader.batch_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # Variables pour early stopping
    best_val_acc = 0.0
    patience_counter = 0

    # Historique
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'=' * 60}")

        # Entraînement
        train_loss = train_epoch(trainer, train_loader, epoch)
        trainer.train_losses.append(train_loss)
        history['train_loss'].append(train_loss)

        # Validation
        val_loss, val_acc = trainer.validate(val_loader)
        trainer.val_losses.append(val_loss)
        trainer.val_accuracies.append(val_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\n📊 Résultats Epoch {epoch}:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   Val Acc:    {val_acc:.4f} ({val_acc * 100:.2f}%)")

        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            best_model_path = save_dir / "best_model.pth"
            trainer.save_checkpoint(str(best_model_path))
            print(f"   🌟 Nouveau meilleur modèle sauvegardé! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n⏹️  Early stopping après {epoch} epochs")
            print(f"   Meilleure Val Acc: {best_val_acc:.4f}")
            break

        # Sauvegarder checkpoint régulier
        if epoch % 10 == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pth"
            trainer.save_checkpoint(str(checkpoint_path))

    # Sauvegarder le dernier modèle
    final_model_path = save_dir / "final_model.pth"
    trainer.save_checkpoint(str(final_model_path))

    print(f"\n✅ Entraînement terminé!")
    print(f"   Meilleure Val Acc: {best_val_acc:.4f}")
    print(f"   Modèles sauvegardés dans {save_dir}")

    # Tracer les courbes
    plot_training_history(history, save_dir)

    return history


def plot_training_history(history: Dict, save_dir: Path):
    """
    Trace les courbes d'entraînement
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(epochs, history['val_acc'], 'g-', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=150)
    plt.close()

    print(f"📈 Courbes d'entraînement sauvegardées dans {save_dir / 'training_history.png'}")


def generate_dummy_labels(
        data_dir: str = "data_chew",
        output_file: str = "labels.json"
):
    """
    Génère un fichier de labels dummy pour tester
    Règle simple: utilise le SNR pour choisir le modèle
      - SNR < 10 dB -> modèle 0 (Whisper-large)
      - SNR 10-20 dB -> modèle 1 (Whisper-base)
      - SNR > 20 dB -> modèle 2 (Whisper-tiny)
    """
    meta_dir = Path(data_dir) / "metadata"
    labels = {}

    for meta_file in meta_dir.glob("*.json"):
        with open(meta_file, 'r') as f:
            metadata = json.load(f)

        snr = metadata['snr']
        filename = meta_file.stem

        # Règle basée sur SNR
        if snr < 10:
            label = 0  # Whisper-large pour audio difficile
        elif snr < 20:
            label = 1  # Whisper-base pour audio moyen
        else:
            label = 2  # Whisper-tiny pour audio propre

        labels[filename] = label

    # Sauvegarder
    with open(output_file, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"✅ Labels générés: {output_file}")
    print(f"   {sum(1 for v in labels.values() if v == 0)} fichiers -> modèle 0")
    print(f"   {sum(1 for v in labels.values() if v == 1)} fichiers -> modèle 1")
    print(f"   {sum(1 for v in labels.values() if v == 2)} fichiers -> modèle 2")


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    DATA_DIR = "data_chew"
    LABELS_FILE = "labels.json"  # Ou None pour labels aléatoires
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Étape 1: Générer des labels dummy si besoin
    if not Path(LABELS_FILE).exists():
        print("🏷️  Génération de labels dummy basés sur SNR...")
        generate_dummy_labels(DATA_DIR, LABELS_FILE)

    # Étape 2: Créer les dataloaders
    print("\n📦 Création des dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir=DATA_DIR,
        labels_file=LABELS_FILE,
        train_split=0.8,
        batch_size=BATCH_SIZE,
        num_workers=4
    )

    # Étape 3: Créer le modèle
    print("\n🧠 Création du Model Router...")
    model = ModelRouter(
        n_mels=80,
        n_models=3,
        spec_hidden_dims=[64, 128, 256],
        meta_hidden_dims=[32, 64],
        fusion_hidden_dim=128,
        dropout=0.3
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Nombre de paramètres: {num_params:,}")

    # Étape 4: Entraîner
    print("\n" + "=" * 60)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        save_dir="checkpoints",
        early_stopping_patience=10
    )

    print("\n🎉 Entraînement terminé!")


#
# De plus, peux tu faire en sorte que l'erreur à rétro-propagé est le wer dans le code quand on fait tourner un modèle et faire en sorte que il y ait vraiment l'appel des différents modèles à chaque prédiction du model router (faire en sorte que ce soit simple d'ajouter des modèles).
#
#
#
# Peux tu faire en sorte de mettre tout les paramètres important dans un config pour que je puisse faire plein de changement et avoir une vision globale ?