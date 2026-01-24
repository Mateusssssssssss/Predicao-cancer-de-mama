from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import cross_val_score
from notebooks.preprocess import * 

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

#Validação cruzada
results = cross_val_score(xg, x_train, y_train, cv=3)
print(f'Cross Validation: {results}')

# Treinamento do modelo
xg.fit(x_train, y_train)

# Salvando o modelo treinado
joblib.dump(xg,"best_model/predicao_cancer_mama.pkl")