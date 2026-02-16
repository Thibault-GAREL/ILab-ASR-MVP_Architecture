"""
Exemple d'utilisation du WhisperModel
Montre comment charger et utiliser le modèle Whisper pour transcrire l'audio
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au path pour permettre l'import de 'models'
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
import librosa
from models import WhisperModel, BatchInfo

# Charger un vrai fichier audio depuis data_audio/
audio_file = (
    root_dir / "data_audio" / "20090202-0900-PLENARY-12-fr_20090202-21_22_29_2.wav"
)

print(f"Chargement du fichier audio: {audio_file.name}")
audio, sample_rate = librosa.load(
    audio_file, sr=None
)  # sr=None garde le sample rate original
duration = len(audio) / sample_rate

print(f"  - Durée: {duration:.2f}s")
print(f"  - Sample rate: {sample_rate} Hz")
print(f"  - Forme: {audio.shape}")

# Créer les informations du batch
batch_info = BatchInfo(
    batch_id=audio_file.stem,
    language="fr",  # Audio en français
    duration=duration,
    start_time=0.0,
    end_time=duration,
    sample_rate=sample_rate,
    context="European Parliament plenary session",
)

# Initialiser le modèle Whisper
# Options de taille: 'tiny', 'base', 'small', 'medium', 'large-v3'
# Plus le modèle est grand, plus il est précis mais lent
model = WhisperModel(
    model_size="base",  # Bon compromis vitesse/précision
    device="cpu",  # Utiliser CPU (CUDA nécessite cublas)
    compute_type="int8",  # int8 est optimal sur CPU
)

# Transcrire l'audio
print("Transcription en cours...")
result = model.infer(
    audio=audio, batch_info=batch_info, language="fr"  # Optionnel, aide le modèle
)

# Afficher les résultats
print("\n=== Résultats de la transcription ===")
print(f"Texte: {result.text}")
print(f"Modèle: {result.model.name}")
print(f"Device: {result.model.device}")
print(f"Temps d'inférence: {result.model.inference_time:.3f}s")
print(f"Nombre de mots: {len(result.words)}")

print("\n=== Mots avec timestamps ===")
# Afficher tous les mots s'il y en a moins de 20, sinon les 20 premiers
num_words_to_show = min(20, len(result.words))
for word in result.words[:num_words_to_show]:
    print(
        f"  '{word.word}' [{word.start:.2f}s - {word.end:.2f}s] (conf: {word.confidence:.2f})"
    )
if len(result.words) > num_words_to_show:
    print(f"  ... ({len(result.words) - num_words_to_show} mots supplémentaires)")

# Convertir en JSON (afficher uniquement un résumé)
json_output = result.to_dict()
print("\n=== Informations du batch ===")
import json

print(json.dumps(json_output["batch"], indent=2, ensure_ascii=False))
print(f"\n=== Stats modèle ===")
print(json.dumps(json_output["model"], indent=2, ensure_ascii=False))

# Décharger le modèle de la mémoire
model.unload_model()
print("\nModèle déchargé de la mémoire")
