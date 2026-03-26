import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['Gatunek'] = [target_names[i] for i in y]

print("--- PODSTAWOWE STATYSTYKI ZBIORU IRIS ---")
print(df.describe())
print("\n")

plt.figure(figsize=(10, 8))
sns.pairplot(df.drop('target', axis=1), hue='Gatunek', palette='viridis', markers=["o", "s", "D"])
plt.suptitle('Rozklad cech i ich zaleznosci w zbiorze Iris', y=1.02)
plt.show()

k_values = range(1, 11)
accuracies = []

for k in k_values:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters_temp = kmeans_temp.fit_predict(X)

    mapped_labels_temp = np.zeros_like(clusters_temp)
    for i in range(k):
        mask = (clusters_temp == i)
        if np.sum(mask) > 0:
            dominant_label = pd.Series(y[mask]).mode()[0]
            mapped_labels_temp[mask] = dominant_label

    accuracies.append(accuracy_score(y, mapped_labels_temp))

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b', linewidth=2)
plt.title('Wplyw liczby klastrow (k) na dokladnosc (Accuracy)')
plt.xlabel('Parametr k (liczba klastrow)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_3 = kmeans_3.fit_predict(X)
centroids = kmeans_3.cluster_centers_

plt.figure(figsize=(10, 6))
markers = ['o', 's', '^']

for i, marker in enumerate(markers):
    mask = (y == i)
    plt.scatter(X[mask, 2], X[mask, 3], c=clusters_3[mask], cmap='viridis', vmin=0, vmax=2,
                marker=marker, edgecolor='k', s=80, label=f'Prawdziwa klasa: {target_names[i]}')

plt.scatter(centroids[:, 2], centroids[:, 3], c='red', marker='X', s=250, label='Centroidy', edgecolor='white',
            linewidths=2)

plt.title('Dzialanie modelu: Kolory = Klastry K-Means, Ksztalty = Prawdziwe Klasy')
plt.xlabel('Dlugosc platka (petal length)')
plt.ylabel('Szerokosc platka (petal width)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

mapped_labels = np.zeros_like(clusters_3)
for i in range(3):
    mask = (clusters_3 == i)
    mapped_labels[mask] = pd.Series(y[mask]).mode()[0]

print("--- OCENA SKUTECZNOŚCI PRZYPISANIA KLASTRÓW (dla k=3) ---")
print(f"Accuracy: {accuracy_score(y, mapped_labels):.4f}")
print(f"Precision: {precision_score(y, mapped_labels, average='weighted'):.4f}")
print(f"Recall: {recall_score(y, mapped_labels, average='weighted'):.4f}")
print(f"F1-Score: {f1_score(y, mapped_labels, average='weighted'):.4f}")

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=mapped_labels, cmap='viridis', edgecolor='k', s=60)
plt.title('Wizualizacja t-SNE po grupowaniu K-Means na zbiorze Iris')
plt.xlabel('Wymiar t-SNE 1')
plt.ylabel('Wymiar t-SNE 2')
handles, _ = scatter.legend_elements()
plt.legend(handles, target_names, title="Grupowanie (zmapowane)")
plt.show()