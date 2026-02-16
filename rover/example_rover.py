"""
Exemple d'utilisation du système ROVER avec 3 modèles Whisper
Teste la combinaison de transcriptions de tiny, base et small
"""

import sys
from pathlib import Path
import json

# Ajouter le dossier parent au path pour les imports
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import librosa
from models import WhisperModel, BatchInfo
from rover import ROVER

print("=" * 80)
print("ROVER - Recognizer Output Voting Error Reduction")
print("Test avec 3 modèles Whisper: tiny, base, small")
print("=" * 80)

# Charger un fichier audio réel
audio_file = root_dir / "data_audio" / "20090202-0900-PLENARY-9-fr_20090202-17_29_00_1.wav"

print(f"\n📁 Chargement du fichier audio: {audio_file.name}")
audio, sample_rate = librosa.load(audio_file, sr=None)
duration = len(audio) / sample_rate

print(f"  - Durée: {duration:.2f}s")
print(f"  - Sample rate: {sample_rate} Hz")

# Créer les informations du batch
batch_info = BatchInfo(
    batch_id=audio_file.stem,
    language="fr",
    duration=duration,
    start_time=0.0,
    end_time=duration,
    sample_rate=sample_rate,
    context="European Parliament plenary session",
)

# Initialiser 3 modèles Whisper de tailles différentes
print("\n🤖 Initialisation des modèles Whisper...")
models = {
    "tiny": WhisperModel(model_size="tiny", device="gpu", compute_type="float16"),
    "base": WhisperModel(model_size="base", device="gpu", compute_type="float16"),
    "small": WhisperModel(model_size="small", device="gpu", compute_type="float16"),
}

print(f"  ✓ {len(models)} modèles initialisés: {list(models.keys())}")

# Transcrire avec chaque modèle
print("\n🎙️  Transcription en cours avec chaque modèle...")
transcriptions = []

for model_name, model in models.items():
    print(f"\n  [{model_name.upper()}] Transcription...")
    result = model.infer(audio=audio, batch_info=batch_info, language="fr")
    transcriptions.append(result)

    print(f"    - Temps: {result.model.inference_time:.2f}s")
    print(f"    - Mots: {len(result.words)}")
    print(
        f"    - Confiance moyenne: {sum(w.confidence for w in result.words) / len(result.words) if result.words else 0:.2%}"
    )
    print(f"    - Texte: {result.text[:80]}{'...' if len(result.text) > 80 else ''}")

    # Décharger le modèle pour libérer la mémoire
    model.unload_model()

# Comparer les transcriptions individuelles
print("\n" + "=" * 80)
print("📊 COMPARAISON DES TRANSCRIPTIONS INDIVIDUELLES")
print("=" * 80)

for i, trans in enumerate(transcriptions):
    print(f"\n[{trans.model.name.upper()}]")
    print(f"Texte complet:\n{trans.text}")

# Initialiser le système ROVER
print("\n" + "=" * 80)
print("🗳️  ROVER - SYSTÈME DE VOTE")
print("=" * 80)

rover = ROVER(
    time_tolerance=0.3,  # Tolérance de 300ms pour l'alignement
    min_confidence=0.3,  # Filtrer les mots avec confiance < 30%
    confidence_weight=True,  # Pondérer les votes par confiance
    normalize_words=True,  # Normaliser les mots (lowercase)
)

print("\nParamètres ROVER:")
print(f"  - Tolérance temporelle: {rover.time_tolerance}s")
print(f"  - Confiance minimale: {rover.min_confidence}")
print(f"  - Vote pondéré: {rover.confidence_weight}")
print(f"  - Normalisation: {rover.normalize_words}")

# Combiner les transcriptions avec ROVER
print("\n🔄 Combinaison des transcriptions...")
rover_result = rover.combine(transcriptions)

print(f"\n✅ Transcription ROVER finale:")
print(f"\nTexte:\n{rover_result.text}")
print(f"\nStatistiques:")
print(f"  - Nombre de modèles: {rover_result.num_models}")
print(f"  - Modèles utilisés: {', '.join(rover_result.models_used)}")
print(f"  - Nombre de mots: {len(rover_result.aligned_words)}")
print(f"  - Score d'accord: {rover_result.agreement_score:.1%}")

# Analyser les désaccords
print("\n" + "=" * 80)
print("⚠️  ANALYSE DES DÉSACCORDS")
print("=" * 80)

disagreements = rover.get_disagreements(rover_result, threshold=0.6)

if disagreements:
    print(
        f"\n{len(disagreements)} mots avec désaccord significatif (< 60% d'accord):\n"
    )

    for i, word in enumerate(disagreements[:10], 1):  # Afficher les 10 premiers
        print(f"{i}. [{word.start:.2f}s] Mot sélectionné: '{word.word}'")
        print(f"   Vote: {word.vote_count}/{rover_result.num_models} modèles")
        print(f"   Candidats:")
        for cand_word, cand_conf, cand_model in word.candidates:
            print(f"     - {cand_model}: '{cand_word}' (conf: {cand_conf:.2%})")

    if len(disagreements) > 10:
        print(f"\n   ... et {len(disagreements) - 10} autres désaccords")
else:
    print("\n✅ Aucun désaccord significatif détecté!")

# Statistiques détaillées
print("\n" + "=" * 80)
print("📈 STATISTIQUES DÉTAILLÉES")
print("=" * 80)

stats = rover.compare_transcriptions(transcriptions)
print(f"\nNombre de mots par modèle:")
for model, count in zip(stats["models"], stats["word_counts"]):
    print(f"  - {model}: {count} mots")

print(f"\nTemps d'inférence:")
for model, time in zip(stats["models"], stats["inference_times"]):
    print(f"  - {model}: {time:.2f}s")

print(f"\nConfiance moyenne par modèle:")
for model, conf in zip(stats["models"], stats["avg_confidences"]):
    print(f"  - {model}: {conf:.1%}")

# Sauvegarder le résultat ROVER en JSON
output_file = root_dir / "rover_output.json"
print(f"\n💾 Sauvegarde du résultat dans {output_file.name}...")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(rover_result.to_dict(), f, indent=2, ensure_ascii=False)

print("\n✅ Test ROVER terminé avec succès!")
print("=" * 80)
