"""
Audio Preprocessing Module pour Model Router ASR
Transforme des fichiers audio en log mel spectrogrammes avec métadonnées
Découpe en segments de 15s avec 5s d'overlap
Sauvegarde les résultats dans data_chew/
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass, asdict
import librosa
import json
import pickle


@dataclass
class AudioMetadata:
    """Métadonnées extraites de l'audio"""
    snr: float  # Signal-to-Noise Ratio en dB
    duration: float  # Durée en secondes
    sample_rate: int
    num_channels: int
    rms_energy: float  # Root Mean Square energy
    zero_crossing_rate: float
    spectral_centroid: float  # Centre de gravité spectral
    original_filename: str  # Nom du fichier original
    segment_index: int  # Index du segment (0, 1, 2, ...)
    segment_start_time: float  # Temps de début du segment en secondes
    segment_end_time: float  # Temps de fin du segment en secondes
    total_segments: int  # Nombre total de segments pour ce fichier


class AudioPreprocessor:
    """
    Préprocesseur audio pour le model router ASR
    Supporte .wav, .mp3, .flac, .ogg, .m4a, .mp4, etc.
    Découpe en segments de 15s avec 5s d'overlap
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = 8000.0,
        batch_size: int = 8,
        segment_duration: float = 15.0,  # Durée de chaque segment
        segment_overlap: float = 5.0,    # Overlap entre segments
        output_dir: str = "data_chew",
    ):
        """
        Args:
            target_sample_rate: Taux d'échantillonnage cible
            n_fft: Taille de la FFT
            hop_length: Hop length pour STFT
            n_mels: Nombre de filtres mel
            f_min: Fréquence minimale
            f_max: Fréquence maximale
            batch_size: Taille des batches
            segment_duration: Durée de chaque segment en secondes (15s)
            segment_overlap: Overlap entre segments en secondes (5s)
            output_dir: Dossier de sortie pour sauvegarder les données
        """
        self.target_sample_rate = target_sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or target_sample_rate / 2
        self.batch_size = batch_size
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap
        self.segment_hop = segment_duration - segment_overlap  # 10s entre chaque début de segment
        self.output_dir = Path(output_dir)

        # Créer les dossiers de sortie
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "spectrograms").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)

        # Transformation mel spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=self.f_max,
        )

    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        """
        Charge un fichier audio (supporte .wav, .mp3, .flac, .ogg, .m4a, .mp4, etc.)

        Args:
            audio_path: Chemin vers le fichier audio

        Returns:
            Tuple (waveform, sample_rate)
            waveform shape: (channels, samples)
        """
        audio_path = Path(audio_path)

        try:
            # torchaudio supporte nativement .wav, .flac, .ogg, .mp3 (avec backend approprié)
            waveform, sr = torchaudio.load(str(audio_path))
        except Exception as e:
            # Fallback sur librosa pour formats plus exotiques (.m4a, .mp4, etc.)
            print(f"Torchaudio failed, using librosa fallback: {e}")
            waveform_np, sr = librosa.load(str(audio_path), sr=None, mono=False)

            # Convertir en tensor et assurer shape (channels, samples)
            if waveform_np.ndim == 1:
                waveform_np = waveform_np[np.newaxis, :]
            waveform = torch.from_numpy(waveform_np).float()

        return waveform, sr

    def preprocess_audio(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Prétraite l'audio: resample, mono, normalisation

        Args:
            waveform: Tensor audio (channels, samples)
            sr: Sample rate original

        Returns:
            waveform prétraité (1, samples)
        """
        # Resample si nécessaire
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            waveform = resampler(waveform)

        # Convertir en mono si stéréo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Normalisation
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        return waveform

    def segment_audio(self, waveform: torch.Tensor) -> List[Tuple[torch.Tensor, float, float]]:
        """
        Découpe l'audio en segments de segment_duration avec segment_overlap

        Args:
            waveform: Tensor audio (1, samples)

        Returns:
            Liste de tuples (segment_waveform, start_time, end_time)
        """
        total_samples = waveform.shape[1]
        total_duration = total_samples / self.target_sample_rate

        segment_samples = int(self.segment_duration * self.target_sample_rate)
        hop_samples = int(self.segment_hop * self.target_sample_rate)

        segments = []
        start_sample = 0

        while start_sample < total_samples:
            end_sample = min(start_sample + segment_samples, total_samples)

            # Extraire le segment
            segment = waveform[:, start_sample:end_sample]

            # Calculer les temps
            start_time = start_sample / self.target_sample_rate
            end_time = end_sample / self.target_sample_rate

            # Si le segment est trop court (< 5s), on le pad ou on l'ignore
            segment_duration_actual = segment.shape[1] / self.target_sample_rate

            if segment_duration_actual >= 5.0:  # Minimum 5s pour garder un segment
                # Pad le dernier segment s'il est plus court que segment_duration
                if segment.shape[1] < segment_samples:
                    pad_length = segment_samples - segment.shape[1]
                    segment = torch.nn.functional.pad(segment, (0, pad_length), value=0)

                segments.append((segment, start_time, end_time))

            # Avancer de hop_samples (10s)
            start_sample += hop_samples

            # Sortir si on a dépassé la fin
            if end_sample >= total_samples:
                break

        return segments

    def calculate_snr(self, waveform: torch.Tensor) -> float:
        """
        Calcule le Signal-to-Noise Ratio (SNR) en dB
        Utilise une approximation: signal = variance globale, bruit = variance sur silences
        """
        wav_np = waveform.squeeze().numpy()

        # Détection des segments silencieux (threshold = 1% de l'amplitude max)
        threshold = 0.01 * np.max(np.abs(wav_np))
        noise_mask = np.abs(wav_np) < threshold

        if noise_mask.sum() == 0:
            return 40.0  # SNR très élevé si pas de silence détecté

        signal_power = np.var(wav_np)
        noise_power = np.var(wav_np[noise_mask]) + 1e-10

        snr_db = 10 * np.log10(signal_power / noise_power)
        return float(snr_db)

    def extract_metadata(
        self,
        waveform: torch.Tensor,
        sr: int,
        original_filename: str,
        segment_index: int,
        segment_start_time: float,
        segment_end_time: float,
        total_segments: int
    ) -> AudioMetadata:
        """
        Extrait les métadonnées de l'audio

        Args:
            waveform: Tensor audio (1, samples)
            sr: Sample rate
            original_filename: Nom du fichier original
            segment_index: Index du segment
            segment_start_time: Temps de début du segment
            segment_end_time: Temps de fin du segment
            total_segments: Nombre total de segments

        Returns:
            AudioMetadata
        """
        wav_np = waveform.squeeze().numpy()

        # SNR
        snr = self.calculate_snr(waveform)

        # Durée
        duration = waveform.shape[1] / sr

        # RMS Energy
        rms_energy = float(np.sqrt(np.mean(wav_np ** 2)))

        # Zero Crossing Rate
        zcr = float(np.mean(librosa.zero_crossings(wav_np)))

        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=wav_np, sr=sr, hop_length=self.hop_length
        )
        spectral_centroid = float(np.mean(spectral_centroid))

        return AudioMetadata(
            snr=snr,
            duration=duration,
            sample_rate=sr,
            num_channels=waveform.shape[0],
            rms_energy=rms_energy,
            zero_crossing_rate=zcr,
            spectral_centroid=spectral_centroid,
            original_filename=original_filename,
            segment_index=segment_index,
            segment_start_time=segment_start_time,
            segment_end_time=segment_end_time,
            total_segments=total_segments,
        )

    def audio_to_log_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convertit l'audio en log mel spectrogram

        Args:
            waveform: Tensor audio (1, samples)

        Returns:
            Log mel spectrogram (n_mels, time_frames)
        """
        # Mel spectrogram
        mel_spec = self.mel_transform(waveform)

        # Log scale (avec epsilon pour stabilité numérique)
        log_mel_spec = torch.log(mel_spec + 1e-9)

        # Squeeze channel dimension
        log_mel_spec = log_mel_spec.squeeze(0)  # (n_mels, time)

        return log_mel_spec

    def save_processed_audio(
        self,
        log_mel_spec: torch.Tensor,
        metadata: AudioMetadata,
        output_name: str,
    ):
        """
        Sauvegarde le spectrogramme et les métadonnées

        Args:
            log_mel_spec: Log mel spectrogram
            metadata: Métadonnées audio
            output_name: Nom de base pour les fichiers de sortie (sans extension)
        """
        # Sauvegarder le spectrogramme en .pt (format PyTorch)
        spec_path = self.output_dir / "spectrograms" / f"{output_name}.pt"
        torch.save(log_mel_spec, spec_path)

        # Sauvegarder les métadonnées en JSON
        metadata_path = self.output_dir / "metadata" / f"{output_name}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)

        print(f"✅ Sauvegardé: {output_name}")

    def process_single_audio(
        self,
        audio_path: Union[str, Path],
        save: bool = True,
    ) -> List[Dict[str, Union[torch.Tensor, AudioMetadata]]]:
        """
        Traite un seul fichier audio et le découpe en segments

        Args:
            audio_path: Chemin vers le fichier audio
            save: Si True, sauvegarde les résultats dans data_chew/

        Returns:
            Liste de dicts avec 'log_mel_spec' et 'metadata' pour chaque segment
        """
        audio_path = Path(audio_path)

        # Charger l'audio
        waveform, sr = self.load_audio(audio_path)

        # Prétraiter
        waveform = self.preprocess_audio(waveform, sr)

        # Découper en segments
        segments = self.segment_audio(waveform)
        total_segments = len(segments)

        print(f"🔪 {audio_path.name}: {total_segments} segments créés")

        results = []

        for segment_idx, (segment_waveform, start_time, end_time) in enumerate(segments):
            # Extraire métadonnées
            metadata = self.extract_metadata(
                segment_waveform,
                self.target_sample_rate,
                audio_path.name,
                segment_idx,
                start_time,
                end_time,
                total_segments
            )

            # Générer log mel spectrogram
            log_mel_spec = self.audio_to_log_mel_spectrogram(segment_waveform)

            # Sauvegarder si demandé
            if save:
                output_name = f"{audio_path.stem}_seg{segment_idx:03d}"
                self.save_processed_audio(log_mel_spec, metadata, output_name)

            results.append({
                'log_mel_spec': log_mel_spec,
                'metadata': metadata,
            })

        return results

    def collate_fn(self, batch_data: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Fonction de collation pour créer des batches
        Tous les segments ont la même durée (15s) donc pas besoin de padding

        Args:
            batch_data: Liste de dicts avec 'log_mel_spec' et 'metadata'

        Returns:
            Dict avec tensors batchés
        """
        log_mel_specs = [item['log_mel_spec'] for item in batch_data]
        metadatas = [item['metadata'] for item in batch_data]

        # Stack en batch (pas de padding nécessaire car tous ont 15s)
        log_mel_specs_batch = torch.stack(log_mel_specs)  # (batch, n_mels, time)

        # Métadonnées en tensors
        metadata_tensor = torch.tensor([
            [m.snr, m.duration, m.rms_energy, m.zero_crossing_rate, m.spectral_centroid]
            for m in metadatas
        ])  # (batch, 5)

        return {
            'log_mel_specs': log_mel_specs_batch,
            'metadata': metadata_tensor,
        }

    def process_batch_from_paths(
        self,
        audio_paths: List[Union[str, Path]],
        save: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Traite un batch de fichiers audio

        Args:
            audio_paths: Liste de chemins vers fichiers audio
            save: Si True, sauvegarde chaque audio traité

        Returns:
            Liste de segments traités
        """
        all_segments = []

        for audio_path in audio_paths:
            try:
                segments = self.process_single_audio(audio_path, save=save)
                all_segments.extend(segments)
            except Exception as e:
                print(f"❌ Erreur lors du traitement de {audio_path}: {e}")
                continue

        return all_segments

    def process_directory(
        self,
        input_dir: Union[str, Path],
        audio_extensions: List[str] = None,
    ):
        """
        Traite tous les fichiers audio d'un dossier

        Args:
            input_dir: Dossier contenant les fichiers audio
            audio_extensions: Extensions à traiter (par défaut: wav, mp3, flac, ogg, m4a)
        """
        if audio_extensions is None:
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.mp4']

        input_dir = Path(input_dir)
        audio_files = []

        # Collecter tous les fichiers audio
        for ext in audio_extensions:
            audio_files.extend(input_dir.glob(f"*{ext}"))
            audio_files.extend(input_dir.glob(f"*{ext.upper()}"))

        print(f"📂 Trouvé {len(audio_files)} fichiers audio dans {input_dir}")

        # Traiter chaque fichier
        total_segments = 0
        for i, audio_file in enumerate(audio_files):
            print(f"\n📄 Traitement du fichier {i+1}/{len(audio_files)}: {audio_file.name}")

            try:
                segments = self.process_single_audio(audio_file, save=True)
                total_segments += len(segments)
            except Exception as e:
                print(f"❌ Erreur: {e}")

        print(f"\n✨ Traitement terminé!")
        print(f"   - {len(audio_files)} fichiers traités")
        print(f"   - {total_segments} segments créés")
        print(f"   - Fichiers sauvegardés dans {self.output_dir}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialiser le preprocessor avec segmentation 15s et overlap 5s
    preprocessor = AudioPreprocessor(
        target_sample_rate=16000,
        n_mels=80,
        batch_size=8,
        segment_duration=15.0,   # 15 secondes par segment
        segment_overlap=5.0,      # 5 secondes d'overlap
        output_dir="data_chew",
    )

    # Option 1: Traiter des fichiers individuels
    audio_paths = [
        "data_audio/sample1.wav",
        "data_audio/sample2.wav"
        # ... plus de fichiers
    ]

    # for audio_path in audio_paths:
    #     segments = preprocessor.process_single_audio(audio_path, save=True)
    #     print(f"Créé {len(segments)} segments pour {audio_path}")

    # Option 2: Traiter tout un dossier
    preprocessor.process_directory("data_audio")

    # Option 3: Traiter et obtenir des segments pour entraînement
    # all_segments = preprocessor.process_batch_from_paths(audio_paths[:2], save=True)
    # print(f"Total de {len(all_segments)} segments créés")

    # # Pour créer un batch pour l'entraînement
    # if len(all_segments) >= 4:
    #     batch = preprocessor.collate_fn(all_segments[:4])
    #     print(f"Batch shape: {batch['log_mel_specs'].shape}")