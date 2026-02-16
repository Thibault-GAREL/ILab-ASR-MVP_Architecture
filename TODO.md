# Todo

- [] Gestion des inputs batch
    - [] tests sur la taille des batchs, comment on découpe l'audio en batchs ? *(mots, groupe de mots, etc.)*

- [] Construction du Router + gestion de la descente de gradient
- [] Choix des modèles (prendre des modèles sur hf déjà un peu fine tuned) et standardisation des outputs
- [] Implémenter du Rover pour les groupes de modèles
- [] Code pour calcul du wer et l'erreur *(pondération d'erreur, de la rapidité et de la frugalité)*

### Next Steps

- [ ] Add more model types (Parakeet, Wav2Vec2)
- [ ] Implement neural network router for automatic model selection
- [ ] Add WER evaluation metrics
- [ ] Batch processing with segment overlapping