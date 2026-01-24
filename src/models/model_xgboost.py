from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from notebooks.preprocess import * 
from logger_config import setup_logger

logger = setup_logger("EDA", "logs/model_xgboost.log")

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

logger.info(f'Cross Validation Results: {results}')

# Treinamento do modelo
xg.fit(x_train, y_train)

# Log de sucesso
logger.info("Modelo XGBoost treinado com sucesso.")