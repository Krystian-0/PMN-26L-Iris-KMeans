import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("--- PODSTAWOWE STATYSTYKI ZBIORU IRIS ---")
print(df.describe())
print("\n")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

mapped_labels = np.zeros_like(clusters)
for i in range(3):
    mask = (clusters == i)
    dominant_label = pd.Series(y[mask]).mode()[0]
    mapped_labels[mask] = dominant_label

acc = accuracy_score(y, mapped_labels)
prec = precision_score(y, mapped_labels, average='weighted')
rec = recall_score(y, mapped_labels, average='weighted')
f1 = f1_score(y, mapped_labels, average='weighted')

print("--- OCENA SKUTECZNOŚCI PRZYPISANIA KLASTRÓW ---")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=mapped_labels, cmap='viridis', edgecolor='k', s=60)
plt.title('Wizualizacja t-SNE po grupowaniu K-Means na zbiorze Iris')
plt.xlabel('Wymiar t-SNE 1')
plt.ylabel('Wymiar t-SNE 2')

handles, _ = scatter.legend_elements()
plt.legend(handles, iris.target_names, title="Grupowanie (zmapowane)")
plt.show()