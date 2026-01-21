from notebooks.eda import *
from sklearn.model_selection import train_test_split


# matrix
target = dados['diagnosis']
previsores = dados.drop(columns=['diagnosis'])

# Divisão dos dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(
   previsores, target, test_size=0.4, random_state=42
)