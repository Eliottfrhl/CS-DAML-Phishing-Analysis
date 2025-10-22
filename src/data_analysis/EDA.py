import pandas as pd
import numpy as np

# Chargement du dataset
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

# Aperçu général
print("Dimensions du dataset :", df.shape)
print("\nColonnes disponibles :")
print(df.columns.tolist())

# Types de données
print("\nTypes de variables :")
print(df.dtypes.value_counts())

# Aperçu des 5 premières lignes
df.head()

# Vérification des valeurs manquantes
missing = df.isna().sum()
missing_cols = missing[missing > 0]
print("\nColonnes avec valeurs manquantes :")
print(missing_cols if not missing_cols.empty else "Aucune valeur manquante détectée.")

# Statistiques descriptives sur les colonnes numériques
desc = df.describe().T
desc[['mean', 'std', 'min', 'max']].head(10)

# Répartition de la variable cible
label_counts = df['label'].value_counts(normalize=True) * 100
print("\nRépartition des classes (en %):")
print(label_counts)

# Exemple : quelques variables intéressantes
df[['URLLength', 'DomainLength', 'NoOfJS', 'NoOfImage', 'NoOfExternalRef']].describe()

import matplotlib.pyplot as plt

df['label'].value_counts().plot(
    kind='bar',
    color=['tab:red', 'tab:blue'],
    rot=0
)
plt.xticks([0, 1], ['Phishing (1)', 'Légitime (0)'])
plt.ylabel('Nombre de sites')
plt.title('Distribution de la variable cible (label)')
plt.grid(axis='y', alpha=0.4)
plt.savefig('distribution_label.png', dpi=300, bbox_inches='tight')
plt.show()

