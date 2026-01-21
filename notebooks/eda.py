from data.dados import load_data
from preprocess import encode_dados
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

# Quantidade de classes
class_counts = dados['diagnosis'].value_counts()
print(f'Contagem de classes:\n{class_counts}')


# Transformar coluna categorica em numérica
dados['diagnosis'], encoder = encode_dados(dados['diagnosis'])

# Correlação
correlation_matrix = dados.corr()['diagnosis'].sort_values(ascending=False)
print(f'Matriz de Correlação:\n{correlation_matrix}')