"""
ROVER (Recognizer Output Voting Error Reduction)
Combine plusieurs transcriptions ASR en utilisant un vote au niveau des mots
pour produire une transcription finale optimisée
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
import sys

# Ajouter le dossier parent au path pour les imports
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from models.base_model import TranscriptionOutput, WordTranscription


@dataclass
class AlignedWord:
    """
    Représente un mot aligné avec les votes de plusieurs modèles

    Attributes:
        word (str): Le mot final sélectionné par vote
        start (float): Temps de début moyen en secondes
        end (float): Temps de fin moyen en secondes
        confidence (float): Confiance moyenne pondérée
        candidates (List[Tuple[str, float, str]]): Liste des (mot, confiance, modèle) candidats
        vote_count (int): Nombre de modèles ayant voté pour ce mot
    """

    word: str
    start: float
    end: float
    confidence: float
    candidates: List[Tuple[str, float, str]] = field(default_factory=list)
    vote_count: int = 1


@dataclass
class ROVEROutput:
    """
    Sortie du système ROVER

    Attributes:
        text (str): Transcription finale après vote
        aligned_words (List[AlignedWord]): Mots alignés avec informations de vote
        num_models (int): Nombre de modèles ayant participé au vote
        models_used (List[str]): Noms des modèles utilisés
        agreement_score (float): Score d'accord entre les modèles (0-1)
    """

    text: str
    aligned_words: List[AlignedWord]
    num_models: int
    models_used: List[str]
    agreement_score: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit la sortie ROVER en dictionnaire

        Returns:
            Dict[str, Any]: Dictionnaire représentant la sortie ROVER
        """
        return {
            "text": self.text,
            "num_models": self.num_models,
            "models_used": self.models_used,
            "agreement_score": self.agreement_score,
            "num_words": len(self.aligned_words),
            "aligned_words": [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                    "vote_count": w.vote_count,
                    "candidates": [
                        {"word": cand[0], "confidence": cand[1], "model": cand[2]}
                        for cand in w.candidates
                    ],
                }
                for w in self.aligned_words
            ],
        }


class ROVER:
    """
    Implémentation du système ROVER pour combiner plusieurs transcriptions ASR

    Le ROVER fonctionne en 3 étapes:
    1. Alignement temporel des mots de différentes transcriptions
    2. Regroupement des mots se chevauchant temporellement
    3. Vote pour sélectionner le meilleur mot dans chaque groupe
    """

    def __init__(
        self,
        time_tolerance: float = 0.3,
        min_confidence: float = 0.0,
        confidence_weight: bool = True,
        normalize_words: bool = True,
    ) -> None:
        """
        Initialise le système ROVER

        Args:
            time_tolerance (float): Tolérance temporelle pour l'alignement (secondes)
            min_confidence (float): Confiance minimale pour qu'un mot soit considéré
            confidence_weight (bool): Pondérer les votes par la confiance
            normalize_words (bool): Normaliser les mots (lowercase, strip)
        """
        self.time_tolerance = time_tolerance
        self.min_confidence = min_confidence
        self.confidence_weight = confidence_weight
        self.normalize_words = normalize_words

    def combine(self, transcriptions: List[TranscriptionOutput]) -> ROVEROutput:
        """
        Combine plusieurs transcriptions en utilisant le vote ROVER

        Args:
            transcriptions (List[TranscriptionOutput]): Liste des transcriptions à combiner

        Returns:
            ROVEROutput: Transcription combinée après vote
        """
        if not transcriptions:
            raise ValueError("Au moins une transcription est requise")

        if len(transcriptions) == 1:
            # Cas trivial: une seule transcription
            trans = transcriptions[0]
            aligned_words = [
                AlignedWord(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    confidence=w.confidence,
                    candidates=[(w.word, w.confidence, trans.model.name)],
                    vote_count=1,
                )
                for w in trans.words
            ]
            return ROVEROutput(
                text=trans.text,
                aligned_words=aligned_words,
                num_models=1,
                models_used=[trans.model.name],
                agreement_score=1.0,
            )

        # Extraire tous les mots de toutes les transcriptions
        all_words = self._extract_all_words(transcriptions)

        # Aligner les mots temporellement
        aligned_groups = self._align_words(all_words)

        # Voter pour chaque groupe de mots alignés
        voted_words = self._vote_on_groups(aligned_groups)

        # Dédupliquer les mots consécutifs identiques
        voted_words = self._deduplicate_consecutive(voted_words)

        # Construire la transcription finale
        raw_text = " ".join([w.word for w in voted_words])
        final_text = self._clean_text(raw_text)

        # Calculer le score d'accord
        agreement_score = self._calculate_agreement(voted_words, len(transcriptions))

        # Extraire les noms des modèles
        models_used = [t.model.name for t in transcriptions]

        return ROVEROutput(
            text=final_text,
            aligned_words=voted_words,
            num_models=len(transcriptions),
            models_used=models_used,
            agreement_score=agreement_score,
        )

    def _extract_all_words(
        self, transcriptions: List[TranscriptionOutput]
    ) -> List[Tuple[WordTranscription, str]]:
        """
        Extrait tous les mots de toutes les transcriptions avec leur source

        Args:
            transcriptions (List[TranscriptionOutput]): Liste des transcriptions

        Returns:
            List[Tuple[WordTranscription, str]]: Liste de (mot, nom_modèle)
        """
        all_words = []
        for trans in transcriptions:
            for word in trans.words:
                # Filtrer les mots avec confiance trop faible
                if word.confidence >= self.min_confidence:
                    all_words.append((word, trans.model.name))

        # Trier par temps de début
        all_words.sort(key=lambda x: x[0].start)

        return all_words

    def _align_words(
        self, all_words: List[Tuple[WordTranscription, str]]
    ) -> List[List[Tuple[WordTranscription, str]]]:
        """
        Aligne les mots temporellement en groupes qui se chevauchent
        Utilise une approche de clustering pour éviter complètement les doublons
        Garantit qu'un mot de chaque modèle n'apparaît qu'une seule fois

        Args:
            all_words (List[Tuple[WordTranscription, str]]): Tous les mots triés par temps

        Returns:
            List[List[Tuple[WordTranscription, str]]]: Groupes de mots alignés
        """
        if not all_words:
            return []

        # Séparer les mots par modèle
        words_by_model = defaultdict(list)
        for word, model in all_words:
            words_by_model[model].append(word)

        # Indices de consommation pour chaque modèle
        indices = {model: 0 for model in words_by_model}
        aligned_groups = []

        # Tant qu'il reste des mots à traiter
        while any(indices[model] < len(words_by_model[model]) for model in indices):
            # Trouver le mot non traité avec le start le plus précoce
            earliest_time = float("inf")
            earliest_model = None

            for model, idx in indices.items():
                if idx < len(words_by_model[model]):
                    if words_by_model[model][idx].start < earliest_time:
                        earliest_time = words_by_model[model][idx].start
                        earliest_model = model

            if earliest_model is None:
                break

            # Créer un groupe avec ce mot comme centre
            anchor_word = words_by_model[earliest_model][indices[earliest_model]]
            group = [(anchor_word, earliest_model)]
            indices[earliest_model] += 1

            anchor_center = (anchor_word.start + anchor_word.end) / 2

            # Pour chaque autre modèle, voir si le prochain mot s'aligne
            for model in words_by_model:
                if model == earliest_model:
                    continue

                idx = indices[model]
                if idx >= len(words_by_model[model]):
                    continue

                candidate_word = words_by_model[model][idx]
                candidate_center = (candidate_word.start + candidate_word.end) / 2

                # Vérifier l'alignement temporel
                time_diff = abs(anchor_center - candidate_center)
                overlap = (
                    candidate_word.start <= anchor_word.end + self.time_tolerance
                    and anchor_word.start <= candidate_word.end + self.time_tolerance
                )

                # Si aligné, ajouter au groupe et incrémenter l'index
                if time_diff <= self.time_tolerance or overlap:
                    group.append((candidate_word, model))
                    indices[model] += 1

            aligned_groups.append(group)

        return aligned_groups

    def _vote_on_groups(
        self, aligned_groups: List[List[Tuple[WordTranscription, str]]]
    ) -> List[AlignedWord]:
        """
        Vote pour sélectionner le meilleur mot dans chaque groupe aligné

        Args:
            aligned_groups (List[List[Tuple[WordTranscription, str]]]): Groupes alignés

        Returns:
            List[AlignedWord]: Mots finaux sélectionnés par vote
        """
        voted_words = []

        for group in aligned_groups:
            # Normaliser les mots si demandé
            word_scores = defaultdict(lambda: {"score": 0.0, "count": 0, "data": []})

            # Compter le nombre unique de modèles dans ce groupe
            unique_models = set([model for _, model in group])
            num_models_in_group = len(unique_models)

            for word, model in group:
                word_text = word.word
                if self.normalize_words:
                    word_text = word_text.lower().strip()

                # Calculer le score (vote pondéré par confiance ou simple comptage)
                if self.confidence_weight:
                    score = word.confidence
                else:
                    score = 1.0

                word_scores[word_text]["score"] += score
                word_scores[word_text]["count"] += 1
                word_scores[word_text]["data"].append((word, model))

            # Sélectionner le mot avec le meilleur score
            if not word_scores:
                continue

            best_word = max(word_scores.items(), key=lambda x: x[1]["score"])
            word_text = best_word[0]
            word_info = best_word[1]

            # Calculer les statistiques moyennes
            all_word_data = word_info["data"]
            avg_start = np.mean([w.start for w, _ in all_word_data])
            avg_end = np.mean([w.end for w, _ in all_word_data])
            avg_confidence = word_info["score"] / word_info["count"]

            # Créer la liste des candidats (tous les mots du groupe)
            candidates = []
            for word_cand, model in group:
                cand_text = word_cand.word
                if self.normalize_words:
                    cand_text = cand_text.lower().strip()
                candidates.append((cand_text, word_cand.confidence, model))

            # Créer le mot aligné
            # vote_count = nombre de modèles uniques qui ont voté pour le mot gagnant
            models_voting_for_winner = set([model for _, model in word_info["data"]])

            aligned_word = AlignedWord(
                word=word_text,
                start=float(avg_start),
                end=float(avg_end),
                confidence=float(avg_confidence),
                candidates=candidates,
                vote_count=len(models_voting_for_winner),
            )

            voted_words.append(aligned_word)

        return voted_words

    def _deduplicate_consecutive(
        self, voted_words: List[AlignedWord]
    ) -> List[AlignedWord]:
        """
        Fusionne les mots consécutifs identiques (après normalisation)
        Garde le mot avec la meilleure confiance et fusionne les candidats

        Args:
            voted_words (List[AlignedWord]): Liste des mots votés

        Returns:
            List[AlignedWord]: Liste dédupliquée
        """
        if len(voted_words) <= 1:
            return voted_words

        deduplicated = []
        i = 0

        while i < len(voted_words):
            current = voted_words[i]

            # Chercher les mots consécutifs identiques
            duplicates = [current]
            j = i + 1

            while j < len(voted_words):
                next_word = voted_words[j]

                # Vérifier si les mots sont identiques (après normalisation)
                if current.word.lower().strip() == next_word.word.lower().strip():
                    duplicates.append(next_word)
                    j += 1
                else:
                    break

            # Si on a trouvé des doublons, fusionner
            if len(duplicates) > 1:
                # Garder le mot avec la meilleure confiance
                best = max(duplicates, key=lambda w: w.confidence)

                # Fusionner tous les candidats
                all_candidates = []
                for dup in duplicates:
                    all_candidates.extend(dup.candidates)

                # Calculer les nouvelles statistiques
                avg_start = np.mean([w.start for w in duplicates])
                avg_end = np.mean([w.end for w in duplicates])
                total_vote_count = sum([w.vote_count for w in duplicates])

                # Créer le mot fusionné
                merged = AlignedWord(
                    word=best.word,
                    start=float(avg_start),
                    end=float(avg_end),
                    confidence=best.confidence,
                    candidates=all_candidates,
                    vote_count=min(
                        total_vote_count, len(all_candidates)
                    ),  # Limité par le nombre de candidats
                )

                deduplicated.append(merged)
            else:
                deduplicated.append(current)

            i = j

        return deduplicated

    def _calculate_agreement(
        self, voted_words: List[AlignedWord], num_models: int
    ) -> float:
        """
        Calcule le score d'accord entre les modèles

        Args:
            voted_words (List[AlignedWord]): Mots votés
            num_models (int): Nombre de modèles

        Returns:
            float: Score d'accord entre 0 et 1
        """
        if not voted_words or num_models == 0:
            return 0.0

        # Score basé sur le nombre de modèles d'accord pour chaque mot
        agreement_scores = [w.vote_count / num_models for w in voted_words]

        return float(np.mean(agreement_scores))

    def _clean_text(self, text: str) -> str:
        """
        Nettoie le texte final en corrigeant les espaces autour des apostrophes
        et de la ponctuation

        Args:
            text (str): Texte brut à nettoyer

        Returns:
            str: Texte nettoyé
        """
        import re

        # Corriger les espaces avant les apostrophes: "d '" -> "d'"
        text = re.sub(r"\s+'", "'", text)

        # Corriger les espaces après les apostrophes: "' un" -> "'un"
        text = re.sub(r"'\s+", "'", text)

        # Corriger les espaces avant la ponctuation
        text = re.sub(r"\s+([,.:;!?])", r"\1", text)

        # Supprimer les espaces multiples
        text = re.sub(r"\s+", " ", text)

        # Nettoyer les espaces au début et à la fin
        text = text.strip()

        return text

    def get_disagreements(
        self, rover_output: ROVEROutput, threshold: float = 0.5
    ) -> List[AlignedWord]:
        """
        Identifie les mots où il y a désaccord entre les modèles

        Args:
            rover_output (ROVEROutput): Sortie du ROVER
            threshold (float): Seuil de désaccord (proportion de votes)

        Returns:
            List[AlignedWord]: Mots avec désaccord significatif
        """
        disagreements = []

        for word in rover_output.aligned_words:
            vote_ratio = word.vote_count / rover_output.num_models
            if vote_ratio < threshold:
                disagreements.append(word)

        return disagreements

    def compare_transcriptions(
        self, transcriptions: List[TranscriptionOutput]
    ) -> Dict[str, Any]:
        """
        Compare plusieurs transcriptions et fournit des statistiques

        Args:
            transcriptions (List[TranscriptionOutput]): Liste des transcriptions

        Returns:
            Dict[str, Any]: Statistiques de comparaison
        """
        if len(transcriptions) < 2:
            return {
                "error": "Au moins 2 transcriptions sont requises pour la comparaison"
            }

        stats = {
            "num_models": len(transcriptions),
            "models": [t.model.name for t in transcriptions],
            "word_counts": [len(t.words) for t in transcriptions],
            "texts": [t.text for t in transcriptions],
            "inference_times": [t.model.inference_time for t in transcriptions],
            "avg_confidences": [
                np.mean([w.confidence for w in t.words]) if t.words else 0.0
                for t in transcriptions
            ],
        }

        return stats
