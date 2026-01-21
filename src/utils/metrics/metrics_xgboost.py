from notebooks.preprocess import *
from src.utils.predicts.predict_xgboost import *
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score)


def metrics(y_true, pred_labels, pred_proba):
    """
    Avalia o desempenho de um modelo de classificação binária usando AUC.

    Parâmetros:
    - y_true: valores reais (0 ou 1)
    - pred_labels: rótulos preditos (0 ou 1)
    - pred_proba: probabilidades preditas da classe positiva (float entre 0 e 1)

    Exibe:
    - AUC-ROC
    - Classification Report
    - Matriz de Confusão
    """
    # AUC-ROC
    auc = roc_auc_score(y_true, pred_proba)
    print(f"AUC-ROC: {auc:.3f}")
    
    # Classification report
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, pred_labels, digits=3))
    
    # Confusion matrix
    print("Matriz de Confusão:")
    print(confusion_matrix(y_true, pred_labels))

# Metricas para o modelo xgboost.
metrics(y_test, pred_labels_xg, pred_proba_xg)