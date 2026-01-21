from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from notebooks.preprocess import *

rf = RandomForestClassifier(
    n_estimators=1000,        # Número de árvores na floresta
)

# Validação cruzada
results = cross_val_score(rf, x_train, y_train, cv=3)

# Treinoamento do modelo
rf.fit(x_train, y_train)