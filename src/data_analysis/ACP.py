import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

# Supprime les colonnes non numériques ou peu pertinentes pour l’ACP
df_num = df.select_dtypes(include=[np.number])
print(df_num.columns)



X = df_num.drop(columns=['label'])
y = df_num['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)  # garde 95 % de la variance
X_pca = pca.fit_transform(X_scaled)

print(f"Nombre de composantes retenues : {pca.n_components_}")
print("Variance expliquée cumulée :", pca.explained_variance_ratio_.sum())

'''

plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance expliquée cumulée')
plt.title('Scree plot, ACP')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm', alpha=0.6)
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.title('ACP - Visualisation des deux premières composantes')
plt.colorbar(label='Label (1=Phishing, 0=Légitime)')
plt.show()

'''


pca_components = pd.DataFrame(
    pca.components_,
    columns=X.columns,
    index=[f"PC{i+1}" for i in range(pca.n_components_)]
)

pca_components.head(2)

pc1_loadings = pca_components.loc["PC1"].abs().sort_values(ascending=False).head(10)
pc2_loadings = pca_components.loc["PC2"].abs().sort_values(ascending=False).head(10)

print("Top 10 variables influentes sur PC1 :")
print(pc1_loadings)
print("\nTop 10 variables influentes sur PC2 :")
print(pc2_loadings)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
pc1_loadings.head(10).plot(kind='bar')
plt.title("Top 10 variables les plus influentes sur la 1ère composante principale")
plt.ylabel("Contribution absolue")
plt.xticks(rotation=10, ha='right')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
pc2_loadings.head(10).plot(kind='bar', color='orange')
plt.title("Top 10 variables les plus influentes sur la 2ème composante principale")
plt.ylabel("Contribution absolue")
plt.xticks(rotation=10, ha='right')
plt.grid(True)
plt.show()
