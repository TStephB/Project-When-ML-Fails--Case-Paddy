import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from data_loader import load_data, split_random

def build_pipeline(X_train):
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42,
        n_jobs=-1
    )

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

def run_attempt1():
    print("--- Tentative 1 : Provocation d'Overfitting via OHE ---")
    df = load_data('paddydataset.csv')
    X_train, X_test, y_train, y_test = split_random(df)
    
    # Construction et ajustement du pipeline autonome
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    
    r2_train = r2_score(y_train, pipeline.predict(X_train))
    r2_test = r2_score(y_test, pipeline.predict(X_test))
    
    print(f"R2 Train : {r2_train:.3f}")
    print(f"R2 Test  : {r2_test:.3f}")
    print("Constat : L'écart est minime. L'overfitting attendu n'a pas eu lieu.")
    print("Explication : Les variables catégorielles d'origine ont une très basse cardinalité")
    print("  (Variety: 3, Agriblock: 6) et la variable 'Hectares' (fortement corrélée avec la cible)")
    print("  sert de raccourci d'apprentissage dominant, masquant tout risque de sur-apprentissage.")

if __name__ == "__main__":
    run_attempt1()
