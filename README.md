# Previsão cancer de mama: Malignos ou Benignos

Este projeto utiliza aprendizado de máquina para identificar tumores benignos e malignos com base em um conjunto de dados que contém várias características do tumor. A análise é feita com a utilização de machine learning.

# Modelo com melhor Metrica
XGBOOST

# Bibliotecas e Ferramentas Utilizadas

Pandas: Para manipulação e análise de dados.
Numpy: Para operações matemáticas e manipulação de arrays.
Scikit-learn: Para pré-processamento de dados, como Label Encoding e One-Hot Encoding, e divisão de dados em treino e teste.
Matplotlib: Para visualização de gráficos e métricas.
Seaborn: Para visualização de gráficos.


# 1. Leitura do Dataset

O conjunto de dados foi carregado a partir de um arquivo CSV contendo características morfológicas dos tumores.
```python
dados = pd.read_csv('dataset/cancer_mama.csv')
```

# 2. Análise de Dados (EDA)

Informações sobre o dataset: shape, dtypes, valores nulos e dados duplicados:
```python
def informacoes(df):
    return {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "nulls": df.isnull().sum(),
        "duplicates": df.duplicated().sum()
    }
```

Identificado que as colunas 'Unnamed 32' e 'id'não é importante para o treinamento:
```python
   dados = dados.drop(columns=["Unnamed: 32", "id"])
```

Retirada da coluna categorica para visualização boxplot, para identificar possiveis outliers:
``` python
dados_boxplot = dados.drop(columns=['diagnosis'])
```

Contagem de cada classe da tabela target: Malignos (M) e Benignos (B).
``` python
class_counts = dados['diagnosis'].value_counts()
``` 
Transformando a coluna 'diagnosis' categorica para numerica: (B) = 0 e (M) = 1:

```python
dados['diagnosis'], encoder = encode_dados(dados['diagnosis'])
```

Função criada para a visualização das correlações entre diagnosis e as demais colunas:
```python
correlation_matrix = dados.corr()['diagnosis'].sort_values(ascending=False)
```
# Pré-processamento

Definição da coluna target e dos previsores:

```python
target = dados['diagnosis']
previsores = dados.drop(columns=['diagnosis'])
```
Função para transformação de dados categoricos em numericos.

```python
def encode_dados(y):
    le = LabelEncoder()
    dados_encoded = le.fit_transform(y)
    return dados_encoded, le
```

Divisão dos dados em treino e teste
``` python
x_train, x_test, y_train, y_test = train_test_split(
   previsores, target, test_size=0.4, random_state=42
)
```
# Parametros do Modelos

``` python

xg = XGBClassifier(objective='binary:logistic', # Objetivo de classificação binária
    eval_metric='auc',            # Métrica de avaliação
    n_estimators=1000,             # Número de árvores
    learning_rate=1e-2,           # Taxa de aprendizado
    max_depth=30,                  # Profundidade das árvores
    subsample=0.5,                # Amostragem para evitar overfitting
    colsample_bytree=0.5,         # Porcentagem de colunas usadas
    gamma=1,                      # Evita overfitting
    reg_lambda=0,                 # Regularização L2
    reg_alpha=1,                   # Regularização L1
    
)

```
``` python

rf = RandomForestClassifier(
    n_estimators=1000,        # Número de árvores na floresta
)

```
# Estrutura do Projeto
``` python
triagem_assertiva/
│
│
├── dataset/
│   ├── cancer_mama.csv  # Dataset original
│   ├── dados.py               # Carregamento e manipulação dos dados
│
├── notebooks/
│   ├── eda.py                 # Análise exploratória de dados
│   ├── preprocess.py          # split treino/teste
│
├── src/
│   ├── models/            # Treinamento com Modelos Diferentes(model_forest, model_keras, etc ..)
│   ├── utils/             # Pasta Metrics e Predicts
|        ├── metrics/            # Metricas de todos os modelos (metrics_forest, metrics_keras, etc ..)
│        └── predict/            # Predição de todos os modelos (predict_forest, predict_keras, etc ..)
│
├── best_model/
│   ├── triagem_assertiva.pkl  # Melhor modelo serializado
│   ├── model.py           # Lógica para escolha e exportação do melhor modelo
|
└── README.md              # Documentação do projeto
```

# Dataset
O conjunto de dados contém variáveis relacionadas às características morfológicas de tumores de mama, incluindo medidas de raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão fractal, calculadas a partir de imagens digitalizadas de massas mamárias.

# Modelagem
Melhor modelo usado: xgboost

# Técnicas aplicadas:
Train/test split (60% treino, 40% teste)
Parâmetros do Modelo
xgboost


# Funcionalidades
Previsão de classificação de tumores malignos e benignos.
Retorna a classificação dos tumores.
Utiliza modelo de machine learning serializado com joblib

# Métricas Usadas
Usando F1-score, accuracy, precision e auc como avaliação do modelo.

AUC-ROC (Área sob a curva ROC): mede a capacidade do modelo de distinguir entre classes.
Varia de 0 a 1, onde 1 indica um modelo perfeito

Precision (Precisão): proporção de predições positivas que estavam corretas.
Quanto menor o falso positivo, maior a precisão.

Recall (Sensibilidade): proporção de positivos reais que foram corretamente identificados.
#Quanto menor o falso negativo, maior o recall.

F1-Score: média harmônica entre Precision e Recall.
Balanceia precisão e sensibilidade, útil quando as classes são desbalanceadas.


# Predição
Foram implementadas funções utilitárias para cálculo de métricas e geração de previsões com threshold ajustável.



# Métricas de Avaliação do Modelo

Cross Validation: [0.96491228 0.97368421 0.94690265]
AUC-ROC: 0.998
Relatório de Classificacao:

``` python
               precision    recall  f1-score   support

           0      0.980     0.993     0.987       148
           1      0.987     0.963     0.975        80

    accuracy                          0.982       228
   macro avg      0.984     0.978     0.981       228
weighted avg      0.983     0.982     0.982       228


#Matriz de Confusão:
[[147   1]
 [  3  77]]
```
Excelente performance nas classes Benignas e Malignas.   
Recall para classe Benigna <=: 99%
Recall para classe Maligna <=: 96%
Precision para classes Benigna e Malignas <=: 98% 

# Tecnologias Utilizadas  
Python 3.10+  
XGBoost, RandomForest  
Pandas 
Scikit-learn  
Matplotlib
Seaborn

# Visualizações

