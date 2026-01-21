from data.dados import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Dados
dados = load_data('data/cancer_mama.csv')

# matrix
target = dados['diagnosis']
previsores = dados.drop(columns=['diagnosis'])

def encode_target(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le

target, encoder = encode_target(target)
encoder.classes_
print(f'Classes do target: {encoder.classes_}')

# Divisão dos dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(
   previsores, target, test_size=0.4, random_state=42
)