import pandas as pd

def load_data(df):
    df = pd.read_csv('data/cancer_mama.csv')
    return df