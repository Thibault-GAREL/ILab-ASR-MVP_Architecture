"""
Implémentation du modèle Whisper d'OpenAI pour l'ASR
Utilise faster-whisper pour des performances optimales
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from pathlib import Path

try:
    from faster_whisper import WhisperModel as FasterWhisperModel

    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("Warning: faster-whisper non installé. Utilisez: pip install faster-whisper")

from models.base_model import (
    BaseASRModel,
    BatchInfo,
    WordTranscription,
    TranscriptionOutput,
)


class WhisperModel(BaseASRModel):
    """
    Wrapper pour les modèles Whisper d'OpenAI
    Supporte tous les variants: tiny, base, small, medium, large-v3

    Utilise faster-whisper pour des performances optimales avec CTranslate2
    """

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: str = "float16",
        download_root: Optional[str] = None,
    ) -> None:
        """
        Initialise le modèle Whisper

        Args:
            model_size (str): Taille du modèle ('tiny', 'base', 'small', 'medium', 'large-v3')
            device (Optional[str]): Device à utiliser ('cpu', 'cuda', 'auto')
            compute_type (str): Type de calcul ('float16', 'int8', 'float32')
            download_root (Optional[str]): Dossier où télécharger les modèles
        """
        model_name = f"whisper-{model_size}"
        super().__init__(
            model_name=model_name,
            model_type="local",
            device=device,
            version="openai/whisper",
        )

        self.model_size = model_size
        self.compute_type = compute_type
        self.download_root = download_root
        self.target_sample_rate = 16000  # Whisper nécessite 16kHz

        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper n'est pas installé. "
                "Installez-le avec: pip install faster-whisper"
            )

    def load_model(self) -> None:
        """
        Charge le modèle Whisper en mémoire
        Utilise faster-whisper avec CTranslate2 pour l'optimisation
        """
        print(f"Chargement du modèle Whisper ({self.model_size}) sur {self.device}...")

        # Mapper le device pour faster-whisper
        device = "cuda" if self.device == "cuda" else "cpu"

        # Ajuster compute_type selon le device
        if device == "cpu":
            compute_type = "int8"  # Plus efficace sur CPU
        else:
            compute_type = self.compute_type

        self.model = FasterWhisperModel(
            model_size_or_path=self.model_size,
            device=device,
            compute_type=compute_type,
            download_root=self.download_root,
        )

        print(f"✓ Modèle Whisper chargé avec succès")

    def preprocess(
        self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int
    ) -> np.ndarray:
        """
        Prétraite l'audio pour Whisper
        - Resampling à 16kHz si nécessaire
        - Normalisation si nécessaire
        - Conversion en mono si stéréo

        Args:
            audio (Union[np.ndarray, torch.Tensor]): Signal audio brut
            sample_rate (int): Taux d'échantillonnage du signal audio

        Returns:
            np.ndarray: Audio prétraité (mono, 16kHz, float32)
        """
        # Convertir en numpy si c'est un tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Convertir en float32 si nécessaire
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Convertir en mono si stéréo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=0)

        # Resampling si nécessaire
        if sample_rate != self.target_sample_rate:
            try:
                import librosa

                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=self.target_sample_rate
                )
            except ImportError:
                raise ImportError(
                    "librosa est requis pour le resampling. "
                    "Installez-le avec: pip install librosa"
                )

        # Normaliser l'amplitude si elle dépasse [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        return audio

    def _run_inference(
        self, preprocessed_audio: np.ndarray, language: Optional[str] = None
    ) -> tuple:
        """
        Exécute l'inférence Whisper sur l'audio prétraité

        Args:
            preprocessed_audio (np.ndarray): Audio prétraité (16kHz, mono, float32)
            language (Optional[str]): Code langue pour guider la transcription (ex: 'fr', 'en')

        Returns:
            tuple: (segments, info) - Segments transcrits et informations de détection
        """
        # Paramètres d'inférence
        kwargs = {
            "beam_size": 5,
            "vad_filter": True,  # Utilise VAD pour filtrer les silences
            "vad_parameters": {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 100,
            },
            "word_timestamps": True,  # Activer les timestamps au niveau des mots
        }

        # Ajouter la langue si spécifiée
        if language:
            kwargs["language"] = language

        # Exécuter la transcription
        segments, info = self.model.transcribe(preprocessed_audio, **kwargs)

        # Convertir le générateur en liste
        segments = list(segments)

        return segments, info

    def postprocess(
        self, model_output: tuple, batch_info: BatchInfo, inference_time: float
    ) -> TranscriptionOutput:
        """
        Convertit la sortie Whisper au format standardisé TranscriptionOutput

        Args:
            model_output (tuple): (segments, info) - Sortie brute de Whisper
            batch_info (BatchInfo): Informations sur le batch traité
            inference_time (float): Temps d'inférence en secondes

        Returns:
            TranscriptionOutput: Transcription au format standardisé
        """
        segments, info = model_output

        # Extraire le texte complet
        full_text = " ".join([segment.text.strip() for segment in segments])

        # Extraire les mots avec timestamps
        words = []
        for segment in segments:
            if hasattr(segment, "words") and segment.words:
                for word in segment.words:
                    # Ajuster les timestamps avec le début du batch
                    adjusted_start = batch_info.start_time + word.start
                    adjusted_end = batch_info.start_time + word.end

                    words.append(
                        WordTranscription(
                            word=word.word.strip(),
                            start=adjusted_start,
                            end=adjusted_end,
                            confidence=word.probability,
                        )
                    )

        # Si aucun mot avec timestamp, créer des entrées basiques à partir du texte
        if not words and full_text:
            # Créer un seul mot pour tout le texte (fallback)
            words = [
                WordTranscription(
                    word=full_text,
                    start=batch_info.start_time,
                    end=batch_info.end_time,
                    confidence=0.9,  # confiance par défaut
                )
            ]

        # Créer l'objet ModelInfo
        model_info = self.get_model_info(inference_time)

        # Mettre à jour la langue détectée si disponible
        if hasattr(info, "language"):
            batch_info.language = info.language

        # Créer et retourner la transcription standardisée
        return TranscriptionOutput(
            batch=batch_info, text=full_text, words=words, model=model_info
        )
