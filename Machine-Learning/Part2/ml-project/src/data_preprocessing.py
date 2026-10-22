
from sklearn.model_selection import train_test_split

# Préprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Outils pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def filter_columns(X):
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns
    num_cols = X.columns.difference(cat_cols)

    return cat_cols, num_cols


def preprocess_data(X_train, X_test, num_cols, cat_cols):
    """Préprocess les données d'entrainement et de test"""
    # Préprocessing numérique
    numeric_preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Préprocessing catégoriel
    categorical_preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Préprocessing complet
    preprocess = ColumnTransformer([
        ("num", numeric_preprocess, num_cols),
        ("cat", categorical_preprocess, cat_cols)
    ])

    # Fit et transform sur les données d'entrainement
    X_train_prepared = preprocess.fit_transform(X_train)
    X_test_prepared = preprocess.transform(X_test)

    return X_train_prepared, X_test_prepared


def preprocess(X,y) :

    
    cat_cols, num_cols = filter_columns(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_prepared, X_test_prepared =  preprocess_data(X_train, X_test, num_cols, cat_cols)
    
    return X_train_prepared, X_test_prepared, y_train, y_test  