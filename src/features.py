import pandas as pd
from sklearn.preprocessing import StandardScaler

NUMERIC_COLUMNS = [
    'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality',
    'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
    'MMSE', 'FunctionalAssessment', 'ADL'
]

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=['Ethnicity'], dtype=int)

def scale_numeric_features(df: pd.DataFrame, columns: list = NUMERIC_COLUMNS) -> pd.DataFrame:
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled

def split_features_target(df: pd.DataFrame, target_col: str = 'Diagnosis'):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y

def run_feature_engineering_pipeline(df: pd.DataFrame):
    df_encoded = encode_categorical(df)
    df_scaled = scale_numeric_features(df_encoded)
    X, y = split_features_target(df_scaled)
    return X, y