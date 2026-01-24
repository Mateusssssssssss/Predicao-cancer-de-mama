from notebooks.preprocess import *
from src.utils.predicts.predict_forest import *
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

    # Checagem simples de segurança
    if len(set(y_true)) < 2:
        raise ValueError("y_true precisa ter pelo menos duas classes para calcular AUC.")
    if len(y_true) != len(pred_labels) or len(y_true) != len(pred_proba):
        raise ValueError("y_true, pred_labels e pred_proba devem ter o mesmo tamanho.")
    
    # AUC-ROC
    auc = roc_auc_score(y_true, pred_proba)
    
    # Classification report
    report = classification_report(y_true, pred_labels, digits=3)

    # Confusion matrix
    cm = confusion_matrix(y_true, pred_labels)

    return {
        "auc-roc": auc,
        "classification_report": report,
        "confusion_matrix": cm
    }


# Metricas para o modelo xgboost.
results_rf = metrics(y_test, pred_labels_rf, pred_proba_rf)
# Exibição das métricas
print(f"AUC-ROC: {results_rf['auc-roc']:.3f}")
print("Relatório de Classificacao:")
print(results_rf["classification_report"])
print("Matriz de Confusão:")
print(results_rf["confusion_matrix"])