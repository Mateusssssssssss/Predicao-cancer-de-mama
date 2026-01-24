from src.models.model_xgboost import *
from logger_config import setup_logger

logger = setup_logger("Predict_XGBoost", "logs/predict_xgboost.log")

#Previsao para classes binárias
def predict_labels(model, X, threshold=0.5):
    pred_proba = model.predict_proba(X)[:, 1]
    pred_labels = (pred_proba >= threshold).astype(int)
    return pred_labels, pred_proba


pred_labels_xg, pred_proba_xg = predict_labels(xg, x_test, threshold=0.5)

logger.info("Previsoes realizadas com sucesso.")