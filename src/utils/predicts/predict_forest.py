from src.models.model_xgboost import *
import numpy as np

#Previsao para classes binárias

# Probabilidade da classe positiva (classe 1)
pred_proba_rf = xg.predict_proba(x_test)[:, 1]

# Previsão usando threshold 0.5
pred_labels_rf = (pred_proba_rf > 0.5).astype(int)
