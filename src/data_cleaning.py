import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = ['PatientID', 'DoctorInCharge']
    df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    return df_clean