# Concept GNN Pipeline

Pipeline complet pour générer, caractériser et entraîner un GNN capable de retrouver les chaînes d'effets bijectifs appliquées à des matrices binaires 256×256.

## Structure

```
concept-gnn-pipeline/
├── README.md
├── requirements.txt
├── config/
│   ├── bins.yml
│   ├── generator.yml
│   └── train.yml
├── data/
│   ├── logs/               # sorties, checkpoints, metrics
│   └── .gitignore
├── scripts/
│   ├── effect_metric_sweep.py
│   ├── generate_dataset.py
│   ├── train_gnn.py
│   ├── evaluate_gnn.py
│   └── analyse_metrics.py
├── src/
│   ├── __init__.py
│   ├── bijective_pipeline.py
│   └── metrics.py
├── tests/
│   ├── test_metrics.py
│   ├── test_generator.py
│   └── test_model.py
└── colab/
    └── concept_gnn_pipeline.ipynb
```

## Workflow
1. **Balayage métrique** : `python scripts/effect_metric_sweep.py --config config/bins.yml`.
2. **Génération dataset** : `python scripts/generate_dataset.py --config config/generator.yml --max-samples ...`.
3. **Entraînement** : `python scripts/train_gnn.py --config config/train.yml --dataset ...`.
4. **Évaluation / visualisation** : `scripts/evaluate_gnn.py`, `scripts/analyse_metrics.py`.

Le notebook Colab (`colab/concept_gnn_pipeline.ipynb`) orchestre toutes les étapes pour GPU A100.
