from src.models.model_xgboost import *
import numpy as np

#Previsao para classes binárias

# Probabilidade da classe positiva (classe 1)
pred_proba_xg = xg.predict_proba(x_test)[:, 1]

# Previsão usando threshold 0.5
pred_labels_xg = (pred_proba_xg > 0.5).astype(int)
