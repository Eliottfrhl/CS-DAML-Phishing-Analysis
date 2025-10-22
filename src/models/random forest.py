

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)



df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

# Échantillon stratifié (10 000 lignes)
_, df_small = train_test_split(
    df, test_size=10000, stratify=df['label'], random_state=42
)

print(f"Échantillon sélectionné : {df_small.shape[0]} lignes\n")



cols_to_keep = [
    'URLLength', 'DomainLength', 'NoOfSubDomain', 'IsDomainIP',
    'NoOfLettersInURL', 'NoOfDegitsInURL', 'NoOfEqualsInURL',
    'NoOfQMarkInURL', 'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL', 'TLDLength'
]

cols_to_keep = [c for c in cols_to_keep if c in df_small.columns]
X = df_small[cols_to_keep]
y = df_small['label']

print(f"Variables utilisées ({len(cols_to_keep)}) : {cols_to_keep}\n")




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




rf_model = RandomForestClassifier(
    n_estimators=200,       # nombre d'arbres
    max_depth=10,           # profondeur max (évite le surapprentissage)
    min_samples_split=5,    # taille minimale pour diviser un nœud
    min_samples_leaf=2,     # taille minimale d’une feuille
    random_state=42,
    n_jobs=-1               # utilise tous les cœurs CPU
)

rf_model.fit(X_train_scaled, y_train)



y_pred = rf_model.predict(X_test_scaled)
y_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("===== Modèle Random Forest =====")
print(f"Accuracy : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC AUC : {auc:.4f}\n")



cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.xlabel('Prédiction')
plt.ylabel('Valeur réelle')
plt.title('Matrice de confusion - Modèle Random Forest')
plt.show()



importances = pd.DataFrame({
    'Variable': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Variable', data=importances, color='mediumpurple')
plt.title("Importance des variables - Modèle Random Forest")
plt.show()

print(importances)
