import pandas as pd

alzheimer_data = pd.read_csv('data/raw/alzheimers_disease_data.csv')

print(alzheimer_data.head())

alzheimer_data_clean = alzheimer_data.drop(columns=['PatientID','DoctorInCharge'])

alzheimer_data_clean.to_csv("data/processed/alzheimers_disease_data.csv", encoding= "utf-8")