from notebooks.eda import dados_boxplot, dados
import matplotlib.pyplot as plt
import seaborn as sns

# Grafico para verificar normalização
dados.hist(figsize=(15, 12), bins=30)
plt.suptitle("Distribuição das variáveis")
plt.show()

# Visualização do bloxplot
sns.boxplot(data=dados_boxplot, orient="h", palette="Set2")
plt.title("Boxplot das variáveis numéricas")
plt.show()
