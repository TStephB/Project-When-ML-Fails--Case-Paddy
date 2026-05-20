import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from data_loader import load_data, split_extrapolation

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

def run_attempt2():
    print("--- Tentative 2 : Provocation de panne par Extrapolation (OOD Split) ---")
    df = load_data('paddydataset.csv')
    X_train, X_test, y_train, y_test = split_extrapolation(df)
    
    # Construction d'un pipeline indépendant ajusté uniquement sur X_train
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    
    r2_train = r2_score(y_train, pipeline.predict(X_train))
    r2_test = r2_score(y_test, pipeline.predict(X_test))
    
    print(f"R2 Train : {r2_train:.3f}")
    print(f"R2 Test  : {r2_test:.3f}")
    print("Constat : Effondrement total du R2 de Test dû à l'incapacité d'extrapoler des forêts aléatoires.")

if __name__ == "__main__":
    run_attempt2()
