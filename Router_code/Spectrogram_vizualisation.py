"""
Visualiseur de Log Mel Spectrogrammes
Affiche les spectrogrammes sauvegardés dans data_chew/
"""

import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from typing import Optional, List
import seaborn as sns


class SpectrogramVisualizer:
    """Classe pour visualiser les log mel spectrogrammes"""

    def __init__(self, data_dir: str = "data_chew"):
        """
        Args:
            data_dir: Dossier contenant les données (spectrograms/ et metadata/)
        """
        self.data_dir = Path(data_dir)
        self.spec_dir = self.data_dir / "spectrograms"
        self.meta_dir = self.data_dir / "metadata"

        if not self.spec_dir.exists():
            raise ValueError(f"Dossier {self.spec_dir} introuvable!")

        # Configuration matplotlib pour de beaux graphiques
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def load_spectrogram(self, filename: str) -> torch.Tensor:
        """
        Charge un spectrogramme depuis le fichier .pt

        Args:
            filename: Nom du fichier (avec ou sans .pt)

        Returns:
            Log mel spectrogram (n_mels, time)
        """
        if not filename.endswith('.pt'):
            filename = f"{filename}.pt"

        spec_path = self.spec_dir / filename
        if not spec_path.exists():
            raise FileNotFoundError(f"Spectrogramme {spec_path} introuvable!")

        return torch.load(spec_path)

    def load_metadata(self, filename: str) -> dict:
        """
        Charge les métadonnées depuis le fichier .json

        Args:
            filename: Nom du fichier (avec ou sans .json)

        Returns:
            Dict avec les métadonnées
        """
        if not filename.endswith('.json'):
            filename = f"{filename}.json"

        meta_path = self.meta_dir / filename
        if not meta_path.exists():
            return None

        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def plot_single_spectrogram(
        self,
        filename: str,
        figsize: tuple = (12, 6),
        cmap: str = 'viridis',
        show_metadata: bool = True,
    ):
        """
        Affiche un seul spectrogramme avec ses métadonnées

        Args:
            filename: Nom du fichier (sans extension)
            figsize: Taille de la figure
            cmap: Colormap à utiliser
            show_metadata: Si True, affiche les métadonnées
        """
        # Charger données
        spec = self.load_spectrogram(filename)
        metadata = self.load_metadata(filename)

        # Convertir en numpy
        spec_np = spec.numpy()

        # Créer la figure
        if show_metadata and metadata:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                           gridspec_kw={'width_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)

        # Afficher le spectrogramme
        im = ax1.imshow(
            spec_np,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            interpolation='nearest'
        )

        ax1.set_xlabel('Temps (frames)', fontsize=12)
        ax1.set_ylabel('Filtres Mel', fontsize=12)
        ax1.set_title(f'Log Mel Spectrogram: {filename}', fontsize=14, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Log Magnitude', rotation=270, labelpad=20)

        # Afficher métadonnées
        if show_metadata and metadata:
            ax2.axis('off')

            metadata_text = f"""
METADONNEES
{'='*30}

Fichier: {metadata.get('original_filename', 'N/A')}

Duree: {metadata.get('duration', 0):.2f} s

SNR: {metadata.get('snr', 0):.2f} dB

RMS Energy: {metadata.get('rms_energy', 0):.4f}

Zero Crossing: {metadata.get('zero_crossing_rate', 0):.4f}

Spectral Centroid: {metadata.get('spectral_centroid', 0):.1f} Hz

Sample Rate: {metadata.get('sample_rate', 0)} Hz

Shape: {spec.shape[0]} mels x {spec.shape[1]} frames
            """

            ax2.text(0.1, 0.5, metadata_text,
                    fontsize=10,
                    verticalalignment='center',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.show()

    def plot_multiple_spectrograms(
        self,
        filenames: List[str],
        cols: int = 2,
        figsize: tuple = (15, 10),
        cmap: str = 'viridis',
    ):
        """
        Affiche plusieurs spectrogrammes en grille

        Args:
            filenames: Liste des noms de fichiers
            cols: Nombre de colonnes
            figsize: Taille de la figure
            cmap: Colormap
        """
        n = len(filenames)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # S'assurer que axes est toujours un array 2D
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, filename in enumerate(filenames):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            try:
                spec = self.load_spectrogram(filename)
                spec_np = spec.numpy()

                im = ax.imshow(
                    spec_np,
                    aspect='auto',
                    origin='lower',
                    cmap=cmap,
                    interpolation='nearest'
                )

                ax.set_title(filename, fontsize=10)
                ax.set_xlabel('Temps')
                ax.set_ylabel('Mels')

                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            except Exception as e:
                ax.text(0.5, 0.5, f'Erreur:\n{filename}\n{str(e)}',
                       ha='center', va='center', fontsize=8)
                ax.axis('off')

        # Cacher les axes vides
        for idx in range(n, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

    def compare_spectrograms(
        self,
        filenames: List[str],
        figsize: tuple = (15, 5),
        cmap: str = 'viridis',
    ):
        """
        Compare plusieurs spectrogrammes côte à côte

        Args:
            filenames: Liste des noms de fichiers (max 4 recommandé)
            figsize: Taille de la figure
            cmap: Colormap
        """
        n = len(filenames)
        fig, axes = plt.subplots(1, n, figsize=figsize)

        if n == 1:
            axes = [axes]

        for idx, (ax, filename) in enumerate(zip(axes, filenames)):
            try:
                spec = self.load_spectrogram(filename)
                metadata = self.load_metadata(filename)
                spec_np = spec.numpy()

                im = ax.imshow(
                    spec_np,
                    aspect='auto',
                    origin='lower',
                    cmap=cmap,
                    interpolation='nearest'
                )

                title = f"{filename}"
                if metadata:
                    title += f"\nSNR: {metadata['snr']:.1f}dB | Durée: {metadata['duration']:.1f}s"

                ax.set_title(title, fontsize=10)
                ax.set_xlabel('Temps (frames)')
                if idx == 0:
                    ax.set_ylabel('Filtres Mel')

                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            except Exception as e:
                ax.text(0.5, 0.5, f'Erreur: {str(e)}',
                       ha='center', va='center')
                ax.axis('off')

        plt.tight_layout()
        plt.show()

    def list_available_spectrograms(self) -> List[str]:
        """
        Liste tous les spectrogrammes disponibles

        Returns:
            Liste des noms de fichiers (sans extension)
        """
        spec_files = list(self.spec_dir.glob("*.pt"))
        return [f.stem for f in sorted(spec_files)]

    def plot_statistics(self, figsize: tuple = (15, 10)):
        """
        Affiche des statistiques sur tous les spectrogrammes
        """
        filenames = self.list_available_spectrograms()

        if not filenames:
            print("Aucun spectrogramme trouvé!")
            return

        # Collecter les métadonnées
        snrs = []
        durations = []
        energies = []
        zcrs = []
        centroids = []

        for filename in filenames:
            metadata = self.load_metadata(filename)
            if metadata:
                snrs.append(metadata['snr'])
                durations.append(metadata['duration'])
                energies.append(metadata['rms_energy'])
                zcrs.append(metadata['zero_crossing_rate'])
                centroids.append(metadata['spectral_centroid'])

        # Créer les graphiques
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # SNR
        axes[0, 0].hist(snrs, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Distribution du SNR', fontweight='bold')
        axes[0, 0].set_xlabel('SNR (dB)')
        axes[0, 0].set_ylabel('Fréquence')

        # Durée
        axes[0, 1].hist(durations, bins=20, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_title('Distribution de la Durée', fontweight='bold')
        axes[0, 1].set_xlabel('Durée (s)')
        axes[0, 1].set_ylabel('Fréquence')

        # RMS Energy
        axes[0, 2].hist(energies, bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[0, 2].set_title('Distribution du RMS Energy', fontweight='bold')
        axes[0, 2].set_xlabel('RMS Energy')
        axes[0, 2].set_ylabel('Fréquence')

        # Zero Crossing Rate
        axes[1, 0].hist(zcrs, bins=20, edgecolor='black', alpha=0.7, color='red')
        axes[1, 0].set_title('Distribution du Zero Crossing Rate', fontweight='bold')
        axes[1, 0].set_xlabel('ZCR')
        axes[1, 0].set_ylabel('Fréquence')

        # Spectral Centroid
        axes[1, 1].hist(centroids, bins=20, edgecolor='black', alpha=0.7, color='purple')
        axes[1, 1].set_title('Distribution du Spectral Centroid', fontweight='bold')
        axes[1, 1].set_xlabel('Fréquence (Hz)')
        axes[1, 1].set_ylabel('Fréquence')

        # Statistiques textuelles
        axes[1, 2].axis('off')
        stats_text = f"""
STATISTIQUES GLOBALES
{'='*35}

Nombre de fichiers: {len(filenames)}

SNR moyen: {np.mean(snrs):.2f} +/- {np.std(snrs):.2f} dB
Duree moyenne: {np.mean(durations):.2f} +/- {np.std(durations):.2f} s
RMS moyen: {np.mean(energies):.4f} +/- {np.std(energies):.4f}
ZCR moyen: {np.mean(zcrs):.4f} +/- {np.std(zcrs):.4f}
Centroid moyen: {np.mean(centroids):.1f} +/- {np.std(centroids):.1f} Hz
        """

        axes[1, 2].text(0.1, 0.5, stats_text,
                       fontsize=11,
                       verticalalignment='center',
                       family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.tight_layout()
        plt.show()


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialiser le visualiseur
    viz = SpectrogramVisualizer(data_dir="data_chew")

    # Lister les fichiers disponibles
    available = viz.list_available_spectrograms()
    print(f"Spectrogrammes disponibles ({len(available)}):")
    for filename in available[:10]:  # Afficher les 10 premiers
        print(f"  - {filename}")

    if available:
        # Afficher un spectrogramme avec métadonnées
        print(f"\nAffichage de: {available[0]}")
        viz.plot_single_spectrogram(available[0], show_metadata=True)

        # Afficher plusieurs spectrogrammes en grille (si plus de 1 disponible)
        if len(available) >= 4:
            print(f"\nAffichage de 4 spectrogrammes en grille")
            viz.plot_multiple_spectrograms(available[:4], cols=2)

        # Comparer 2-3 spectrogrammes
        if len(available) >= 2:
            print(f"\nComparaison de spectrogrammes")
            viz.compare_spectrograms(available[:min(3, len(available))])

        # Afficher les statistiques globales
        print(f"\nStatistiques globales")
        viz.plot_statistics()