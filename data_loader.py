import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier {filepath} introuvable.")
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip()
    return df

def split_random(df, target_col='Paddy yield(in Kg)', test_size=0.2, random_state=42):
    """Split aléatoire standard pour la Tentative 1 (Overfitting)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def split_extrapolation(df, target_col='Paddy yield(in Kg)'):
    """Split Out-Of-Distribution basé sur Hectares pour la Tentative 2 (Extrapolation)."""
    median_hectares = df['Hectares'].median()
    df_train = df[df['Hectares'] <= median_hectares].copy()
    df_test = df[df['Hectares'] > median_hectares].copy()
    
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    return X_train, X_test, y_train, y_test

def get_scale_dependent_features(df, threshold=0.8):
    """Identifie les variables fortement corrélées avec Hectares (seuil > threshold)."""
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if 'Hectares' not in numeric_df.columns:
        return []
    corrs = numeric_df.corr()['Hectares']
    scale_features = corrs[corrs > threshold].index.tolist()
    scale_features.remove('Hectares')
    # On exclut la cible
    target_col = 'Paddy yield(in Kg)'
    if target_col in scale_features:
        scale_features.remove(target_col)
    return scale_features

def scale_features_by_hectares(df, scale_cols):
    """Normalise les colonnes d'intrants par rapport aux Hectares pour les rendre invariants d'échelle."""
    df_scaled = df.copy()
    for col in scale_cols:
        if col in df_scaled.columns:
            df_scaled[f'{col}_per_Ha'] = df_scaled[col] / df_scaled['Hectares']
    return df_scaled.drop(columns=scale_cols)
