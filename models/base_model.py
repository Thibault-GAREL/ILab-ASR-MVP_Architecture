"""
Classe de base abstraite pour tous les modèles ASR
Définit l'interface standardisée que chaque modèle doit implémenter
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import torch
import numpy as np
import time


@dataclass
class WordTranscription:
    """
    Représente un mot transcrit avec ses métadonnées temporelles

    Attributes:
        word (str): Le mot transcrit
        start (float): Temps de début en secondes
        end (float): Temps de fin en secondes
        confidence (float): Score de confiance entre 0 et 1
    """

    word: str
    start: float
    end: float
    confidence: float


@dataclass
class BatchInfo:
    """
    Informations sur le batch audio traité

    Attributes:
        batch_id (str): Identifiant unique du batch
        language (str): Code langue (ex: 'fr', 'en')
        duration (float): Durée totale du batch en secondes
        start_time (float): Temps de début du segment dans l'audio original
        end_time (float): Temps de fin du segment dans l'audio original
        sample_rate (int): Taux d'échantillonnage en Hz
        context (Optional[str]): Contexte additionnel (domaine, thème)
    """

    batch_id: str
    language: str
    duration: float
    start_time: float
    end_time: float
    sample_rate: int
    context: Optional[str] = None


@dataclass
class ModelInfo:
    """
    Informations sur le modèle ASR utilisé

    Attributes:
        name (str): Nom du modèle (ex: 'whisper-large-v3')
        type (str): Type de modèle ('local' ou 'api')
        inference_time (float): Temps d'inférence en secondes
        version (Optional[str]): Version du modèle
        device (Optional[str]): Device utilisé ('cpu', 'cuda', 'mps')
    """

    name: str
    type: str
    inference_time: float
    version: Optional[str] = None
    device: Optional[str] = None


@dataclass
class TranscriptionOutput:
    """
    Structure de sortie standardisée pour tous les modèles ASR

    Attributes:
        batch (BatchInfo): Informations sur le batch traité
        text (str): Transcription textuelle complète
        words (List[WordTranscription]): Liste des mots avec timestamps
        model (ModelInfo): Informations sur le modèle utilisé
    """

    batch: BatchInfo
    text: str
    words: List[WordTranscription]
    model: ModelInfo

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit la transcription en dictionnaire JSON

        Returns:
            Dict[str, Any]: Dictionnaire représentant la transcription
        """
        return {
            "batch": asdict(self.batch),
            "transcription": {
                "text": self.text,
                "words": [asdict(word) for word in self.words],
            },
            "model": asdict(self.model),
        }


class BaseASRModel(ABC):
    """
    Classe de base abstraite pour tous les modèles ASR
    Définit l'interface standardisée que chaque modèle doit implémenter

    Tous les modèles héritant de cette classe doivent implémenter:
    - load_model(): Charge le modèle en mémoire
    - preprocess(): Prétraite l'audio pour le modèle spécifique
    - _run_inference(): Exécute l'inférence du modèle
    - postprocess(): Convertit la sortie du modèle au format standardisé
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "local",
        device: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        """
        Initialise le modèle ASR de base

        Args:
            model_name (str): Nom du modèle
            model_type (str): Type de modèle ('local' ou 'api')
            device (Optional[str]): Device à utiliser ('cpu', 'cuda', 'mps')
            version (Optional[str]): Version du modèle
        """
        self.model_name = model_name
        self.model_type = model_type
        self.version = version

        # Déterminer le device automatiquement si non spécifié
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Le modèle sera chargé à la première utilisation (lazy loading)
        self.model = None
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """
        Charge le modèle en mémoire
        Doit être implémenté par chaque classe fille

        Raises:
            NotImplementedError: Si la méthode n'est pas implémentée
        """
        pass

    @abstractmethod
    def preprocess(
        self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int
    ) -> Any:
        """
        Prétraite l'audio pour le modèle spécifique
        Chaque modèle peut avoir des besoins différents (resampling, normalisation, etc.)

        Args:
            audio (Union[np.ndarray, torch.Tensor]): Signal audio brut
            sample_rate (int): Taux d'échantillonnage du signal audio

        Returns:
            Any: Audio prétraité dans le format attendu par le modèle

        Raises:
            NotImplementedError: Si la méthode n'est pas implémentée
        """
        pass

    @abstractmethod
    def _run_inference(
        self, preprocessed_audio: Any, language: Optional[str] = None
    ) -> Any:
        """
        Exécute l'inférence du modèle sur l'audio prétraité

        Args:
            preprocessed_audio (Any): Audio prétraité par la méthode preprocess
            language (Optional[str]): Code langue pour guider la transcription

        Returns:
            Any: Résultat brut du modèle (format spécifique au modèle)

        Raises:
            NotImplementedError: Si la méthode n'est pas implémentée
        """
        pass

    @abstractmethod
    def postprocess(
        self, model_output: Any, batch_info: BatchInfo, inference_time: float
    ) -> TranscriptionOutput:
        """
        Convertit la sortie brute du modèle au format standardisé TranscriptionOutput

        Args:
            model_output (Any): Sortie brute du modèle
            batch_info (BatchInfo): Informations sur le batch traité
            inference_time (float): Temps d'inférence en secondes

        Returns:
            TranscriptionOutput: Transcription au format standardisé

        Raises:
            NotImplementedError: Si la méthode n'est pas implémentée
        """
        pass

    def infer(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        batch_info: BatchInfo,
        language: Optional[str] = None,
    ) -> TranscriptionOutput:
        """
        Pipeline complet d'inférence: prétraitement -> inférence -> post-traitement

        Args:
            audio (Union[np.ndarray, torch.Tensor]): Signal audio brut
            batch_info (BatchInfo): Informations sur le batch
            language (Optional[str]): Code langue pour guider la transcription

        Returns:
            TranscriptionOutput: Transcription au format standardisé
        """
        # Charger le modèle si ce n'est pas déjà fait (lazy loading)
        if not self.is_loaded:
            self.load_model()
            self.is_loaded = True

        # Prétraitement
        preprocessed_audio = self.preprocess(audio, batch_info.sample_rate)

        # Inférence avec mesure du temps
        start_time = time.time()
        model_output = self._run_inference(preprocessed_audio, language)
        inference_time = time.time() - start_time

        # Post-traitement
        transcription = self.postprocess(model_output, batch_info, inference_time)

        return transcription

    def unload_model(self) -> None:
        """
        Décharge le modèle de la mémoire pour libérer des ressources
        Utile pour le lazy loading et la gestion mémoire
        """
        if self.is_loaded:
            del self.model
            self.model = None
            self.is_loaded = False

            # Nettoyer le cache GPU si applicable
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def validate_audio(
        self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Valide que l'audio est dans un format acceptable

        Args:
            audio (Union[np.ndarray, torch.Tensor]): Signal audio
            sample_rate (int): Taux d'échantillonnage

        Returns:
            Tuple[bool, Optional[str]]: (est_valide, message_erreur)
        """
        # Convertir en numpy si nécessaire
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Vérifier que l'audio n'est pas vide
        if audio.size == 0:
            return False, "L'audio est vide"

        # Vérifier que le sample rate est raisonnable
        if sample_rate < 8000 or sample_rate > 48000:
            return False, f"Sample rate invalide: {sample_rate} Hz"

        # Vérifier que l'audio n'est pas trop court (< 0.1s)
        duration = len(audio) / sample_rate
        if duration < 0.1:
            return False, f"Audio trop court: {duration:.2f}s"

        return True, None

    def get_model_info(self, inference_time: float) -> ModelInfo:
        """
        Crée un objet ModelInfo avec les informations du modèle

        Args:
            inference_time (float): Temps d'inférence en secondes

        Returns:
            ModelInfo: Informations sur le modèle
        """
        return ModelInfo(
            name=self.model_name,
            type=self.model_type,
            inference_time=inference_time,
            version=self.version,
            device=self.device,
        )

    def __repr__(self) -> str:
        """
        Représentation textuelle du modèle

        Returns:
            str: Description du modèle
        """
        return (
            f"{self.__class__.__name__}("
            f"name='{self.model_name}', "
            f"type='{self.model_type}', "
            f"device='{self.device}', "
            f"loaded={self.is_loaded})"
        )
