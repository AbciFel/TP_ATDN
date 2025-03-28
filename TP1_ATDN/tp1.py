#ABCI Fella
#TP1 ATDN2
#28/03/2025 

#ABCI Fella
#TP1 ATDN2
#PARTIE1: exploration des données
import pandas as pd
import numpy as np

#generation des donnees
np.random.seed(0)  
data = pd.DataFrame({
    'poids': np.random.normal(2.5, 0.5, 100),
    'nourriture': np.random.normal(1.2, 0.3, 100),
    'température': np.random.normal(25, 2, 100)
})
print(data)

print("Moyenne :")
print(data.mean())
print("\nMédiane :")
print(data.median())

print("\nÉcart-type :")
print(data.std())

print("\nVariance :")
print(data.var())

print("\nQuartiles :")
print(data.quantile([0.25, 0.5, 0.75]))
import matplotlib.pyplot as plt
import seaborn as sns
#histogrammes
data.hist(bins=20, figsize=(12, 4))
plt.suptitle("Histogrammes des variables")
plt.show()
#boxplots
plt.figure(figsize=(12, 4))
for i, col in enumerate(data.columns):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=data[col])
    plt.title(f'Boxplot de {col}')
plt.tight_layout()
plt.show()

#exercice 2: detection de outliers 
#IQR
def detect_outliers_iqr(data, column):
    Q1= data[column].quantile(0.25)
    Q3 =data[column].quantile(0.75)
    IQR=Q3-Q1
    lower =Q1 - 1.5 * IQR
    upper= Q3 + 1.5 * IQR
    return data[(data[column] < lower) | (data[column] > upper)]
outliers_poids = detect_outliers_iqr(data, 'poids')
outliers_nourriture = detect_outliers_iqr(data, 'nourriture')
outliers_temp = detect_outliers_iqr(data,'température')
print("Outliers - Poids :",len(outliers_poids))
print("Outliers - Nourriture :", len(outliers_nourriture))
print("Outliers - Température :",len(outliers_temp))
from scipy.stats import zscore

#calcul des Z-scores
z_scores= data.apply(zscore)
outliers_z_poids =data[np.abs(z_scores['poids']) >3]
outliers_z_nourriture= data[np.abs(z_scores['nourriture']) > 3]
outliers_z_temp= data[np.abs(z_scores['température'])> 3]
print("Z-Score - Outliers poids :",len(outliers_z_poids))
print("Z-Score - Outliers nourriture :", len(outliers_z_nourriture))
print("Z-Score - Outliers température :",len(outliers_z_temp))

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 4))
colonnes = ['poids', 'nourriture', 'température']

for i, col in enumerate(colonnes):
    Q1 =data[col].quantile(0.25)
    Q3= data[col].quantile(0.75)
    IQR =Q3 - Q1
    lower= Q1 - 1.5 * IQR
    upper =Q3 +1.5 * IQR
    outliers =data[(data[col] < lower) | (data[col] > upper)]
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=data[col])
    plt.title(f'Boxplot de {col}')
    for val in outliers[col]:
        plt.scatter(0, val, color='red', zorder=10)

plt.tight_layout()
plt.show()

from scipy.stats import shapiro

#test de shapiro-wilk
for col in data.columns:
    stat, p_value = shapiro(data[col])
    print(f"{col} ➜ Statistique = {stat:.4f}, p-value = {p_value:.4f}")

#test t
from scipy.stats import ttest_ind
median_nourriture = data['nourriture'].median()
groupe1 = data[data['nourriture'] <= median_nourriture]['poids']
groupe2 = data[data['nourriture'] > median_nourriture]['poids']
stat, p_value = ttest_ind(groupe1, groupe2)
print(f"Test t ➜ Statistique = {stat:.4f}, p-value = {p_value:.4f}")
#ANOVA 
from scipy.stats import f_oneway
data['groupe_temp'] = pd.qcut(data['température'], 3, labels=["basse", "moyenne", "haute"])
g1 = data[data['groupe_temp'] == "basse"]['poids']
g2 = data[data['groupe_temp'] == "moyenne"]['poids']
g3 = data[data['groupe_temp'] == "haute"]['poids']
stat, p_value = f_oneway(g1, g2, g3)
print(f"ANOVA ➜ Statistique = {stat:.4f}, p-value = {p_value:.4f}")

#PARTIE 2: reduction de dimensionnalité 
#exercice 4: analyse de composantes principales ACP

from sklearn.preprocessing import StandardScaler
data =pd.read_csv("/Users/fella/Desktop/TP_ATDN/TP1_ATDN/donnees_elevage_poulet.csv")

X = data.copy()

#normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#matrice de covariance
cov_matrix = np.cov(X_scaled, rowvar=False)
#valeurs & vecteurs propres
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
#decroissant des composantes principales
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
#projection des donnees sur les 2 premieres composantes
X_pca_manual = X_scaled @ eigenvectors[:, :2]
#pourcentage de variance expliquée
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
print(f"Variance expliquée par PC1 : {explained_variance_ratio[0]*100:.2f}%")
print(f"Variance expliquée par PC2 : {explained_variance_ratio[1]*100:.2f}%\n")
# matrice de covariance
print("Matrice de covariance :\n")
print(pd.DataFrame(cov_matrix, index=X.columns, columns=X.columns))
#qffichage des valeurs propres
print("\nValeurs propres :\n")
for i, val in enumerate(eigenvalues):
    print(f"Composante {i+1} : {val:.4f}")
#affichage des vecteurs propres
print("\nVecteurs propres (composantes principales) :\n")
for i in range(len(eigenvectors)):
    print(f"Composante {i+1} :")
    for j, var in enumerate(X.columns):
        print(f"  {var} : {eigenvectors[j, i]:.4f}")
    print()



# Pourcentage de variance expliquée par chaque composante
explained_variance_ratio = eigenvalues / np.sum(eigenvalues) * 100
cumulative_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eigenvalues) + 1), explained_variance_ratio, marker='o', label="Variance individuelle")
plt.plot(range(1, len(eigenvalues) + 1), cumulative_variance, marker='s', label="Variance cumulée", linestyle='--')
plt.xlabel("Composante principale")
plt.ylabel("Variance expliquée (%)")
plt.title("Scree Plot – Analyse des composantes principales")
plt.axhline(y=70, color='red', linestyle=':', label="Seuil 70%")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#exercice 5 : ACP a noyau 
from sklearn.decomposition import KernelPCA

# On normalise les données
from sklearn.preprocessing import StandardScaler
X = data.copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Différents noyaux à tester
kernels = ['linear', 'rbf', 'poly']

for kernel in kernels:
    kpca = KernelPCA(n_components=2, kernel=kernel)
    X_kpca = kpca.fit_transform(X_scaled)

    # Affichage
    plt.figure(figsize=(6, 5))
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], alpha=0.6)
    plt.title(f"KernelPCA avec noyau = {kernel}")
    plt.xlabel("Composante 1")
    plt.ylabel("Composante 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#PARTIE 3: methode d'ensemble 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
#randomforest
df = data.copy()
df['Survie_binaire'] = (df['Taux_survie_%'] >= 90).astype(int)
X = df.drop(columns=['Taux_survie_%', 'Survie_binaire'])
y = df['Survie_binaire']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Accuracy :", accuracy_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("\nRapport complet :\n", classification_report(y_test, y_pred))
#importance des variables
importances = rf.feature_importances_
features = df.drop(columns=['Taux_survie_%', 'Survie_binaire']).columns

#affichage
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(8,5))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel("Importance")
plt.title("Importance des variables (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#exercice 7: Boosting 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
X = data.drop(columns=['Gain_poids_jour_g'])
y = data['Gain_poids_jour_g']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
# AdaBoost
ada = AdaBoostRegressor(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)

#evaluation
def eval_model(name, y_true, y_pred):
    print(f"\n{name}")
    print("MSE :", mean_squared_error(y_true, y_pred))
    print("R²  :", r2_score(y_true, y_pred))

eval_model("AdaBoost", y_test, y_pred_ada)
eval_model("Gradient Boosting", y_test, y_pred_gbr)
