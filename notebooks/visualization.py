from notebooks.eda import dados_boxplot, dados, correlation_matrix
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


# Matriz de correlação
# Plot
plt.figure(figsize=(10, 8))
ax = correlation_matrix.plot(kind='barh')
plt.title('Correlação das Variáveis com o Diagnosis')
plt.xlabel('Correlação')
plt.ylabel('Variáveis')
plt.gca().invert_yaxis()  # deixa as maiores correlações no topo
plt.grid(axis='x', linestyle='--', alpha=0.6)
# Adiciona os valores nas barras (forma simples)
ax.bar_label(ax.containers[0], fmt='%.3f')
plt.show()


#  grafico da quantidade de classes
class_counts = dados['diagnosis'].value_counts()
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('Distribuição das Classes de Diagnóstico')
plt.xlabel('Diagnóstico')
plt.ylabel('Quantidade')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()