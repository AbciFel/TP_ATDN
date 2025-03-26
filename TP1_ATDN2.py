#ABCI Fella
#TP1 ATDN2 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

#ETAPE 1 
df =pd.read_csv("/Users/fella/Desktop/TP_ATDN2/rendement_mais.csv")


# ETAPE 2.1 mesures de tendance centrale
moyenne= df['RENDEMENT_T_HA'].mean()
mediane =df['RENDEMENT_T_HA'].median()
mode =df['RENDEMENT_T_HA'].mode()[0]

print("=== Tendance centrale ===")
print(f"Moyenne: {moyenne:.2f}")
print(f"Médiane : {mediane}")
print(f"Mode : {mode}")

# ETAPE 2.2 mesures de dispersion
std= df['RENDEMENT_T_HA'].std()
var =df['RENDEMENT_T_HA'].var()
etendue= df['RENDEMENT_T_HA'].max() - df['RENDEMENT_T_HA'].min()

print("\n=== Dispersion ===")
print(f"Écart-type : {std:.2f}")
print(f"Variance : {var:.2f}")
print(f"Étendue : {etendue:.2f}")

# ETAPE 2.3 visualisation
for col in ['RENDEMENT_T_HA', 'PRECIPITATIONS_MM', 'TEMPERATURE_C']:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Histogramme de {col}")
    plt.show()

# Boxplot du rendement
plt.figure()
sns.boxplot(x=df['RENDEMENT_T_HA'])
plt.title("Boxplot du rendement")
plt.show()

#faut bien expliquer les valeurs de la matrice , 1 la donnée est toujours corrélée avec elle meme et expliquer la corelation , pourquoi c'est utile de savoir si il  y a deux variables sont corrélés 

# ETAPE 2.4 correlation
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation")
plt.show()

# ETAPE 3 : Test ANOVA 
groupes = df.groupby("TYPE_SOL")["RENDEMENT_T_HA"].apply(list)
anova = f_oneway(*groupes)
print("\n=== ANOVA ===")
print(f"p-value : {anova.pvalue:.4f}")
if anova.pvalue < 0.05:
    print("→ Le type de sol a une influence significative sur le rendement.")
else:
    print("→ Pas d'influence significative du type de sol.")

# =====================
# 🤖 5. Modélisation (Étape 4)
# =====================
# One-hot encoding
df_encoded = pd.get_dummies(df, columns=["TYPE_SOL"], drop_first=True)

# Features & target
X = df_encoded.drop(columns=["RENDEMENT_T_HA"])
y = df_encoded["RENDEMENT_T_HA"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Régression linéaire
lr = LinearRegression()
lr.fit(X_train, y_train)

# Arbre de décision

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

# Fonction d'évaluation
def eval_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2

# Résultats
mae_lr, rmse_lr, r2_lr = eval_model(lr, X_test, y_test)
mae_dt, rmse_dt, r2_dt = eval_model(dt, X_test, y_test)

print("\n=== Résultats - Régression linéaire ===")
print(f"MAE : {mae_lr:.2f}, RMSE : {rmse_lr:.2f}, R² : {r2_lr:.2f}")

print("\n=== Résultats - Arbre de décision ===")
print(f"MAE : {mae_dt:.2f}, RMSE : {rmse_dt:.2f}, R² : {r2_dt:.2f}")

# =====================
# 6. Importance des variables (etape 5)
# =====================
importances = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=importances.index)
plt.title("Importance des variables (arbre de décision)")
plt.xlabel("Score d'importance")
plt.ylabel("Variables")
plt.show()