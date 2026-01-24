from notebooks.preprocess import *
from src.utils.predicts.predict_xgboost import *
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score)
from logger_config import setup_logger

logger = setup_logger("Metrics_XGBoost", "logs/metrics_xgboost.log")

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



def log_metrics(metrics, model_name="Modelo"):
    """
    Loga métricas de forma estruturada e legível.
    """
    logger.info("Metricas %s", model_name)
    logger.info("AUC-ROC: %.3f", metrics["auc-roc"])
    logger.info("Relatorio de classificacao:\n%s", metrics["classification_report"])
    logger.info("Matriz de confusao:\n%s", metrics["confusion_matrix"])


# Explicação das métricas:
# - AUC-ROC (Área sob a curva ROC): mede a capacidade do modelo de distinguir entre classes.
#   > Varia de 0 a 1, onde 1 indica um modelo perfeito

# - Precision (Precisão): proporção de predições positivas que estavam corretas.
#   > Quanto menor o falso positivo, maior a precisão.

# - Recall (Sensibilidade): proporção de positivos reais que foram corretamente identificados.
#   > Quanto menor o falso negativo, maior o recall.

# - F1-Score: média harmônica entre Precision e Recall.
#   > Balanceia precisão e sensibilidade, útil quando as classes são desbalanceadas.

# Metricas para o modelo xgboost.
results_xg = metrics(y_test, pred_labels_xg, pred_proba_xg)
# Exibição das métricas
print(f"AUC-ROC: {results_xg['auc-roc']:.3f}")
print("Relatório de Classificacao:")
print(results_xg["classification_report"])
print("Matriz de Confusão:")
print(results_xg["confusion_matrix"])

# Log das métricas
log_metrics(results_xg, model_name="XGBoost")
