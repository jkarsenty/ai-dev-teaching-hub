# fichier .py qui reprend les étapes du notebook Alyra-pipelines-ml.ipynb
# sans pipeline pour l'instant, juste pour montrer le préprocessing et l'entraînement d'un modèle
# pou faire écho à la structure de dossier proposée dans le notebook Alyra-pipelines-ml.ipynb


# 0 setup + import des données
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score

# Préprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Outils pipelines
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

# Modèles
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Metrics et évaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1 Charger les données
adult = fetch_openml("adult", version=2, as_frame=True)
print(adult.keys())

# 2 Séparation des features et de la target (X, y)
X = adult.data
y = (adult.target == ">50K").astype(int)  # binaire 0/1
print("X shape:", X.shape, "| y shape:", y.shape)

# 3 Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

print("X_train shape:", X_train.shape, "| X_test shape:", X_test.shape)

# 4 Transformation des doonées

# 4.0 Séparation des colonnes numériques et catégorielles 
# (à creuser pour voir la bonne méthode de séparation)
cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns
num_cols = X.columns.difference(cat_cols)

print("Num cols:", len(num_cols), "| Cat cols:", len(cat_cols))
print ("Shape before preprocessing:", X_train[num_cols].shape, X_train[cat_cols].shape)

# 4.1 Cleaning (imputation + outliers)
# --- NUM : imputation + scaling (fit sur train, transform train/test)
num_imputer = SimpleImputer(strategy="median")
X_train_num = num_imputer.fit_transform(X_train[num_cols])
X_test_num  = num_imputer.transform(X_test[num_cols])

# --- CAT : imputation + one-hot
cat_imputer = SimpleImputer(strategy="most_frequent")
X_train_cat = cat_imputer.fit_transform(X_train[cat_cols])
X_test_cat  = cat_imputer.transform(X_test[cat_cols])

# 4.2 preprocessing (scaling + encoding)
num_scaler = StandardScaler()
X_train_num = num_scaler.fit_transform(X_train_num)
X_test_num  = num_scaler.transform(X_test_num)

ohe = OneHotEncoder(handle_unknown="ignore")
X_train_cat = ohe.fit_transform(X_train_cat)
X_test_cat  = ohe.transform(X_test_cat)

print ("Shape after preprocessing:", X_train_num.shape, X_train_cat.shape)

# --- CONCAT (attention: sparse)
from scipy.sparse import hstack

X_train_prepared = hstack([X_train_num, X_train_cat])
X_test_prepared  = hstack([X_test_num,  X_test_cat])
print("Shape after preprocessing:", X_train_prepared.shape)

# 5 Modèle
clf = LogisticRegression(max_iter=5000, solver="saga")
clf.fit(X_train_prepared, y_train)

# 6 Prédiction et évaluation
y_pred = clf.predict(X_test_prepared)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

# visualisation de la matrice de confusion graphique
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()