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

def scale_numeric_features(df_train: pd.DataFrame, df_test: pd.DataFrame, columns: list = NUMERIC_COLUMNS):
    scaler = StandardScaler()
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    df_train_scaled[columns] = scaler.fit_transform(df_train[columns])
    df_test_scaled[columns] = scaler.transform(df_test[columns])
    return df_train_scaled, df_test_scaled, scaler

def split_features_target(df: pd.DataFrame, target_col: str = 'Diagnosis'):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y

def run_feature_engineering_pipeline(df: pd.DataFrame):
    df_encoded = encode_categorical(df)
    df_scaled = scale_numeric_features(df_encoded)
    X, y = split_features_target(df_scaled)
    return X, y