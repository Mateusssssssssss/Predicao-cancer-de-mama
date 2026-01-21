from data.dados import load_data
from logger_config import setup_logger

# Dados
dados = load_data('data/cancer_mama.csv')

# logger = setup_logger("EDA", "logs/eda.log")

# Função para informação
def informacoes(df):
    return {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "nulls": df.isnull().sum(),
        "duplicates": df.duplicated().sum()
    }

info = informacoes(dados)
print(info)

dados = dados.drop(columns=["Unnamed: 32", "id"])
print(f'Dados após remoção de colunas: \n{dados.head()}')

## Retirada a coluna categorica para o boxplot
dados_boxplot = dados.drop(columns=['diagnosis'])
