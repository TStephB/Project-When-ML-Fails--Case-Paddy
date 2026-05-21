# When Machine Learning Fails — Case Paddy

**Mini-Project | Introduction to AI and Machine Learning**
Ecole Centrale Casablanca — Spring 2026

**Equipe :**
- TONDE Stephane
- SANOU Jacquel
- LOMPO Ernest

*Etudiants en 2ème année — Ecole Centrale Casablanca*

**Encadrantes :** Kawtar Zerhouni & Rym Nassih — UTER MID@S

---

## Résumé

Ce projet étudie les modes de défaillance des modèles prédictifs de Machine Learning appliqués à l'agriculture, sur le jeu de données **Paddy** (UCI 1186). Nous mettons en évidence deux pannes classiques :

1. **Mode principal — Panne d'extrapolation (Feature Representation / Distribution Shift) :** Un Random Forest entraîné sur de petites parcelles (≤ 4 Ha) s'effondre catastrophiquement (R² = -10.077) sur des grandes parcelles (> 4 Ha) car les arbres sont structurellement incapables d'extrapoler au-delà du maximum observé en entraînement.

2. **Mode bonus — Shortcut Learning :** La variable `Hectares` (corrélée à 0.994 avec la cible) agit comme raccourci dominant, masquant l'absence d'apprentissage agronomique réel.

Une correction naïve par mise à l'échelle de la cible seule (R² = 0.706) est un **succès en trompe-l'œil** (corrélation du taux = -0.113). La correction robuste via un pipeline invariant d'échelle résout la panne (R² = 0.380 sur le taux, sans illusion mathématique).

---

## Structure du Projet

```
.
├── paddydataset.csv                        # Jeu de données (2789 parcelles agricoles)
├── data_loader.py                          # Chargement, split OOD, fonctions d'invariance d'échelle
├── Essai 1_overfitting (echec échoué).py   # Tentative 1 : shortcut learning masquant l'overfitting
├── Essai 2_extrapolation (echec réussi).py # Tentative 2 : panne d'extrapolation OOD
├── Correction.py                           # Double correction (Naïve vs Robuste)
├── Paddy_Failure_Analysis.ipynb            # Notebook de restitution (5 sections, grille académique)
├── Paddy_Failure_Report.tex                # Rapport scientifique LaTeX
├── requirements.txt                        # Dépendances Python
└── README.md
```

---

## Modèle utilisé

**Random Forest Regressor** (scikit-learn) — famille tree-based ensembles, conformément à la contrainte du projet (pas de régression linéaire/logistique comme modèle principal).

---

## Résultats clés

| Configuration | R² Test | RMSE Test | Corr. taux |
|---|---|---|---|
| Tentative 1 — Split aléatoire | 0.989 | 968.83 | — |
| Tentative 2 — Split OOD | **-10.077** | 8113.66 | — |
| Correction Naïve (target scaling seul) | 0.706 | 1322.53 | **-0.113** |
| Correction Robuste (pipeline invariant) | 0.380 | 1919.52 | -0.022 |

---

## Installation et Exécution

### Prérequis
Python 3.8+

### Installation
```bash
pip install -r requirements.txt
```

### Exécution

```bash
# Tentative 1 : Shortcut Learning
python "Essai 1_overfitting (echec échoué).py"

# Tentative 2 : Panne d'extrapolation OOD
python "Essai 2_extrapolation (echec réussi).py"

# Double correction (Naïve vs Robuste)
python Correction.py
```

Le notebook `Paddy_Failure_Analysis.ipynb` contient la restitution complète structurée selon la grille d'évaluation académique (Research Question → Symptom → Hypothesis → Correction → Threats to Validity).
