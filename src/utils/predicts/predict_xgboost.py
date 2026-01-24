from src.models.model_xgboost import *
from logger_config import setup_logger
import numpy as np
from sklearn.metrics import precision_recall_curve

logger = setup_logger("Predict_XGBoost", "logs/predict_xgboost.log")

#Previsao para classes binárias

def melhor_threshold_recall(y_true, y_proba, min_precision=0.90):
    """
    Encontra o threshold que maximiza o recall
    garantindo uma precisão mínima.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # precision e recall têm tamanho thresholds + 1
    precision = precision[:-1]
    recall = recall[:-1]

    valid = precision >= min_precision
    if not valid.any():
        return 0.5
    
    best_idx = np.argmax(recall[valid])
    return thresholds[valid][best_idx]


# Função para prever rótulos e probabilidades com um dado threshold
def predict_labels(model, x, threshold):
    pred_proba = model.predict_proba(x)[:, 1]
    pred_labels = (pred_proba >= threshold).astype(int)
    return pred_labels, pred_proba


# Probabilidades no TREINO
y_proba_train = xg.predict_proba(x_train)[:, 1]

best_threshold = melhor_threshold_recall(
    y_train,
    y_proba_train,
    min_precision=0.90
)

print(f"Threshold escolhido: {best_threshold:.3f}") 

#  Aplicando no TESTE
y_pred_test, y_proba_test = predict_labels(
    model=xg,
    x=x_test,
    threshold=best_threshold
)

logger.info(f"Threshold escolhido: {best_threshold:.3f}")
logger.info("Previsoes realizadas com sucesso.")