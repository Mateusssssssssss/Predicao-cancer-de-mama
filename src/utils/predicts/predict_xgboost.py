from src.models.model_xgboost import *
from logger_config import setup_logger

logger = setup_logger("Predict_XGBoost", "logs/predict_xgboost.log")

#Previsao para classes binárias

def prev_trashold(probabilities, threshold=0.5):

    return (probabilities >= threshold).astype(int)

# Probabilidade da classe positiva (classe 1)
pred_proba_xg = xg.predict_proba(x_test)[:, 1]

# Previsão usando threshold 0.5
pred_labels_xg = prev_trashold(pred_proba_xg, threshold=0.5)
