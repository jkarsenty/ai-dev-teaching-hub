from sklearn.datasets import fetch_openml

# Modèles
from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier

# Metrics et évaluation
from sklearn.metrics import accuracy_score #, classification_report, confusion_matrix

from preprocess import preprocess


# 1) Chargement dataset réel : Adult (Census Income)
adult = fetch_openml("adult", version=2, as_frame=True)

# séparation features / target
X = adult.data
y = (adult.target == ">50K").astype(int)  # binaire 0/1

X_train_prepared, X_test_prepared, y_train, y_test = preprocess(X, y)

# Modele et évaluation (même code que précédemment)
clf = LogisticRegression(max_iter=5000, solver="saga")
clf.fit(X_train_prepared, y_train)

# prediction et évaluation
y_pred = clf.predict(X_test_prepared)
print("Accuracy (pipeline preprocessing):", accuracy_score(y_test, y_pred))