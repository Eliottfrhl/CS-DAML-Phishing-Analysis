import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt



df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")
print(f"Dataset initial : {df.shape[0]} lignes, {df.shape[1]} colonnes\n")



from sklearn.model_selection import train_test_split

# On découpe le dataset complet en un échantillon de 10 000 lignes, stratifié sur 'label'
_, df_small = train_test_split(
    df, test_size=10000, random_state=42, stratify=df['label']
)

print(f"Échantillon sélectionné : {df_small.shape[0]} lignes\n")




cols_to_keep = [
    'URLLength', 'DomainLength', 'NoOfSubDomain', 'IsDomainIP',
    'NoOfLettersInURL', 'NoOfDegitsInURL', 'NoOfEqualsInURL',
    'NoOfQMarkInURL', 'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL', 'TLDLength'
]

# On vérifie qu'elles existent dans le dataset
cols_to_keep = [c for c in cols_to_keep if c in df_small.columns]

X = df_small[cols_to_keep]
y = df_small['label']

print(f"Variables conservées ({len(cols_to_keep)} au total) :")
print(cols_to_keep, "\n")



def train_and_evaluate(X, y, title="Modèle"):
    # Split train/test (80/20 stratifié)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modèle
    logreg = LogisticRegression(max_iter=500, random_state=42)
    logreg.fit(X_train_scaled, y_train)

    # Prédictions
    y_pred = logreg.predict(X_test_scaled)
    y_proba = logreg.predict_proba(X_test_scaled)[:, 1]

    # Métriques
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"===== {title} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC : {auc:.4f}\n")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédiction')
    plt.ylabel('Valeur réelle')
    plt.title(f'Matrice de confusion - {title}')
    plt.show()

    return logreg, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled




model_url, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = train_and_evaluate(
    X, y, "Modèle basé sur 10 000 lignes et les caractéristiques de l'URL"
)




coeffs = pd.DataFrame({
    'Variable': X.columns,
    'Coefficient': model_url.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Coefficient', y='Variable', data=coeffs.head(10), color='steelblue')
plt.title("Top 10 variables influentes (poids positifs) - Modèle URL-based")
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x='Coefficient', y='Variable', data=coeffs.tail(10), color='tomato')
plt.title("Top 10 variables influentes (poids négatifs) - Modèle URL-based")
plt.show()
