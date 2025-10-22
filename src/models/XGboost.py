

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from xgboost import XGBClassifier


df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

# Échantillon stratifié (10 000 lignes)
from sklearn.model_selection import train_test_split
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



xgb_model = XGBClassifier(
    n_estimators=200,         # Nombre d'arbres
    learning_rate=0.1,        # Taux d'apprentissage
    max_depth=5,              # Profondeur max des arbres
    subsample=0.8,            # Fraction d’échantillons utilisés par arbre
    colsample_bytree=0.8,     # Fraction de features utilisées par arbre
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'     # Evite les warnings inutiles
)

xgb_model.fit(X_train_scaled, y_train)




y_pred = xgb_model.predict(X_test_scaled)
y_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("===== Modèle XGBoost =====")
print(f"Accuracy : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC AUC : {auc:.4f}\n")



cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Prédiction')
plt.ylabel('Valeur réelle')
plt.title('Matrice de confusion - Modèle XGBoost')
plt.show()




importances = pd.DataFrame({
    'Variable': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Variable', data=importances, color='forestgreen')
plt.title("Importance des variables - Modèle XGBoost")
plt.show()

print(importances)
