#ABCI Fella
#TP2 ATDN2
#01/04/2025 

#Partie1: Optimisation bayesienne 
#=======Implementation et applications ===============
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import matplotlib.pyplot as plt

#on chharge les donnees
df =pd.read_csv("/Users/fella/Desktop/TP_ATDN/TP2_ATDN/tp2_atdn_donnees.csv")
#on extrait les colonnes utiles
X= df[['Humidité (%)', 'Température (°C)']]
y =df['Rendement agricole (t/ha)']

#definir lespace de recherche 
search_space = [
    Real(X['Humidité (%)'].min(), X['Humidité (%)'].max(), name='humidite'),
    Real(X['Température (°C)'].min(), X['Température (°C)'].max(), name='temperature'),
]
#fonction objectif -> on prend les x et on approxime le rendement par interpolation
@use_named_args(search_space)
def objective(humidite, temperature):
    #calcul de la distance entre ce point et les points du dataset
    distances= np.sqrt((X['Humidité (%)'] - humidite)**2 + (X['Température (°C)'] - temperature)**2)
    #moyenne ponderee des rendements les plus proches (on prend les 5 plus proches)
    idx =distances.nsmallest(5).index
    return -y[idx].mean()  #-> on met un "-" car gp_minimize cherche un minimum

#on lance l optimisation bayesienne
result = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=30,  # nombre d'iterations
    random_state=0
)
#resultat optimal
print("Meilleure humidité :",result.x[0])
print("Meilleure température :",result.x[1])
print("Rendement max estimé :", -result.fun)
#courbe de convergence
plt.plot(-np.array(result.func_vals))  # rendement a chaque etape
plt.title("Convergence de l'optimisation")
plt.xlabel("Itération")
plt.ylabel("Rendement estime (t/ha)")
plt.grid(True)
plt.show()

#============================================================================
#random forest + scoring 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
from skopt.space import Integer
import numpy as np
#donnees
X = df[['Humidité (%)', 'Température (°C)', 'pH du sol', 'Précipitations (mm)']]  # toutes les features
y = df['Rendement agricole (t/ha)']
# on separe les donnees en entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Grid search 
# grille hyperparametres
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 10]
}

grid = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, cv=3)
grid.fit(X_train, y_train)

y_pred_grid = grid.predict(X_test)
mse_grid = mean_squared_error(y_test, y_pred_grid)
print("Grid Search MSE :", mse_grid)

from scipy.stats import randint
param_dist = {
    'n_estimators': randint(50, 150),
    'max_depth': randint(3, 15)
}
random = RandomizedSearchCV(RandomForestRegressor(random_state=0), param_distributions=param_dist, n_iter=10, cv=3, random_state=0)
random.fit(X_train, y_train)
y_pred_random = random.predict(X_test)
mse_random = mean_squared_error(y_test, y_pred_random)
print("Random Search MSE :", mse_random)
#optimisation bayesienne 
search_spaces = {
    'n_estimators': Integer(50, 150),
    'max_depth': Integer(3, 15)
}

opt = BayesSearchCV(
    estimator=RandomForestRegressor(random_state=0),
    search_spaces=search_spaces,
    n_iter=10,
    cv=3,
    random_state=0
)

opt.fit(X_train, y_train)

y_pred_bayes = opt.predict(X_test)
mse_bayes = mean_squared_error(y_test, y_pred_bayes)
print("Bayesian Optimization MSE :", mse_bayes)
#comparaison finale
print("Comparaison des méthodes :")
print(f"Grid Search       → MSE = {mse_grid:.3f}")
print(f"Random Search     → MSE = {mse_random:.3f}")
print(f"Bayesian Opt.     → MSE = {mse_bayes:.3f}")


#==============Visualisation====================================================================
import matplotlib.pyplot as plt

#courbe de convergence
# on recupere les scores a chaque iteration
scores = -opt.cv_results_['mean_test_score']  # on prend le - car on veut l'erreur (pas le score R²)

plt.plot(range(1, len(scores)+1), scores, marker='o')
plt.title("Courbe de convergence (Bayesian Optimization)")
plt.xlabel("Itération")
plt.ylabel("MSE moyen (validation croisée)")
plt.grid(True)
plt.show()
#visu des points testes 
import seaborn as sns

#extraire les points teses
n_estimators = [params['n_estimators'] for params in opt.cv_results_['params']]
max_depths = [params['max_depth'] for params in opt.cv_results_['params']]
scores = -opt.cv_results_['mean_test_score']
# on crée un scatterplot avec colorbar
plt.figure(figsize=(8, 6))
sc = plt.scatter(n_estimators, max_depths, c=scores, cmap='viridis', s=100)
plt.title("Points testés par l'optimisation bayésienne")
plt.xlabel("n_estimators")
plt.ylabel("max_depth")
cbar = plt.colorbar(sc)
cbar.set_label("MSE (plus clair = meilleur)")
plt.grid(True)
plt.show()

#==========================================================================
#Partie 2: modeles bayesiens a noyau 
#imports et selections des variables 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
import numpy as np
#on choisit une feature simple (temperature)
X = df[['Température (°C)']].values
y = df['Rendement agricole (t/ha)'].values
#modele de regression bayesienne 
#definir un noyau-> constante * RBF (radial)
kernel = C(1.0, (1e-2, 1e2)) * RBF(5.0, (1e-2, 1e2))
#Creer le modele GP
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
#entrainer le modele
gp.fit(X, y)
#predictions +incertitude
#on genere des points pour la prediction (temperatures entre min et max)
X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
#faire la prediction
y_pred, sigma = gp.predict(X_pred, return_std=True)
#visualisation
plt.figure(figsize=(10, 6))
plt.plot(X, y, 'kx', label='Données réelles')
plt.plot(X_pred, y_pred, 'b-', label='Prédiction moyenne')
plt.fill_between(X_pred.flatten(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, 
                 color='lightblue', alpha=0.5, label='Intervalle de confiance à 95%')
plt.xlabel("Température (°C)")
plt.ylabel("Rendement agricole (t/ha)")
plt.title("Régression bayésienne à noyau")
plt.legend()
plt.grid(True)
plt.show()

#Q12 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#preparation des donnees 
#variables explicatives (climat)
X = df[['Température (°C)', 'Précipitations (mm)', 'Humidité (%)']]
#variable cible -> type de sol
y = df['Type de sol']
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Convertir en 0, 1, 2
# separation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
#modele bayesien 
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
kernel = 1.0 * RBF()
gpc = GaussianProcessClassifier(kernel=kernel, random_state=0)
gpc.fit(X_train, y_train)
#predictions
y_pred_gpc = gpc.predict(X_test)
acc_gpc = accuracy_score(y_test, y_pred_gpc)
print("Précision (Bayesian GP) :", acc_gpc)
#comparaison avec SVM classique 
from sklearn.svm import SVC
#SVM classique
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print("Précision (SVM classique) :", acc_svm)
 
#Q13
#visualisation des probabiltés 
#probabilités predites pour chaque classe
probas = gpc.predict_proba(X_test)
#confiance = max(probabilite) pour chaque prediction
confiance = np.max(probas, axis=1)
plt.figure(figsize=(8, 5))
plt.hist(confiance, bins=10, color='skyblue', edgecolor='black')
plt.title("Distribution de la confiance des prédictions (GPC)")
plt.xlabel("Confiance (max des probabilités)")
plt.ylabel("Nombre d’échantillons")
plt.grid(True)
plt.show()

#Q14 on teste different noyaux (lineaire, rbf, poly)
from sklearn.gaussian_process.kernels import RBF, DotProduct, ConstantKernel
#noyau polynomial approximé : (c + x·x′)^2
poly_kernel = ConstantKernel(1.0) * (DotProduct()) ** 2  # degré 2
#definir les noyaux
kernels = {
    'RBF': RBF(length_scale=1.0),
    'Linéaire (DotProduct)': DotProduct(),
    'Polynomial (approx)': poly_kernel
}

#boucle dentrainement et evaluation
from sklearn.gaussian_process import GaussianProcessClassifier
for name, kernel in kernels.items():
    model = GaussianProcessClassifier(kernel=kernel, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Noyau : {name} → Précision = {acc:.2f}")

