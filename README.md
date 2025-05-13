# Projet : Analyse des Signaux WiFi via CSI et Deep Learning pour la Reconnaissance d'Activités Humaines

## 🧠 Contexte

Les signeaux WiFi, via l'information d'état du canal **(channel state information - CSI)**, permettent de détecter
et d'analyser les mouvements humains dans un environnement sans recourir à des capteurs dédiés. Ce projet vise
à exploiter le CSI combiné à des modéles de deep learning **(CNN/LSTM)** pour estimer le nombre de personnes
ou reconnaître leurs activités. L'optimisation des hyperparamètres des modèles sera réalisée à l'aide 
di **Sine Cosine Algorithm (SCA)**, une métaheuristique inspirée des fonctions trigonométriques.
---

## 🎯 Objectifs

### 1. Prétraitement des données CSI
- Extraire et normaliser les données CSI à partir d’un jeu de données public :
  - [Dataset Wi-Fi CSI – Human Activity Recognition](https://figshare.com/articles/dataset/Dataset_for_Human_Activity_Recognition_using_Wi-Fi_Channel_State_Information_CSI_data/14386892/1?file=27485900)

### 2. Modélisation Deep Learning
- Concevoir une architecture **CNN** ou **LSTM** pour analyser des **séquences CSI**.

### 3. Optimisation des hyperparamètres

Adapter différents algorithmes pour optimiser les hyperparamètres clés tels que :
- Taux d’apprentissage
- Nombre de couches
- Taille des noyaux, etc.

Les algorithmes proposés :

- [**Sine Cosine Algorithm (SCA)**](https://github.com/thieu1995/mealpy/blob/master/mealpy/math_based/SCA.py)

- [**Grey Wolf Optimizer (GWO)**](https://github.com/thieu1995/mealpy/blob/master/mealpy/swarm_based/GWO.py)

- [**Arithmetic Optimization Algorithm (AOA)**](https://github.com/thieu1995/mealpy/blob/master/mealpy/math_based/AOA.py)

- [**Equilibrium Optimizer (EO)**](https://github.com/thieu1995/mealpy/blob/master/mealpy/physics_based/EO.py)

- [**Harris Hawks Optimization (HHO)**](https://github.com/thieu1995/mealpy/blob/master/mealpy/swarm_based/HHO.py)

### 4. Interface Utilisateur

Développer une interface en **Python (Streamlit)** ou **MATLAB (App Designer)** permettant :

- Visualisation des données CSI et des activités détectées
- Ajustement interactif des hyperparamètres et réentraînement du modèle
- Export des résultats :
  - Matrices de confusion
  - Courbes d’apprentissage
  - Métriques de performance (accuracy, précision...)

---

## 🔗 Liens Utiles

- [Dataset CSI – Human Activity Recognition (Figshare)](https://figshare.com/articles/dataset/Dataset_for_Human_Activity_Recognition_using_Wi-Fi_Channel_State_Information_CSI_data/14386892/1?file=27485900)  
- [Code de base CSI Activity Recognition](https://github.com/ludlows/CSI-Activity-Recognition/tree/master)  
- [Implémentation Deep Learning CSI](https://github.com/Retsediv/WIFI_CSI_based_HAR)  
- [Mealpy - Optimisation Metaheuristique](https://github.com/thieu1995/mealpy)

---

## 📌 Note

Les étudiants doivent se regrouper en **équipes de 4 à 5 personnes** et choisir **l’un des algorithmes d’optimisation** suivants :
- SCA
- GWO
- HHO
- AOA
- EO
