import pandas as pd

def load_data(df):
    df = pd.read_csv('data/dados_brutos/cancer_mama.csv')
    return df