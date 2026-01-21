import pandas as pd

def load_data():
    df = pd.read_csv('data/cancer_mama.csv')
    return df