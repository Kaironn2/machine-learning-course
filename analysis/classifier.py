# %%

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / 'data'
file_path = DATA_DIR / 'dados_cerveja_nota.xlsx'
df = pd.read_excel(file_path)
df

# %%

df['aprovado'] = (df['nota'] > 5).astype(int)
df

# %%

import matplotlib.pyplot as plt

plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.grid(True)
plt.title('Cerveja vs Aprovação')
plt.xlabel('Cervejas')
plt.ylabel('Aprovado')

# %%

from sklearn import linear_model
from sklearn import tree
from sklearn import naive_bayes

X = df[['cerveja']]
y = df['aprovado']

reg = linear_model.LogisticRegression(penalty=None, fit_intercept=True)
reg.fit(X, y)

# %%

reg_predict = reg.predict(X.drop_duplicates())
reg_proba = reg.predict_proba(X.drop_duplicates())[:, 1]

tr_full = tree.DecisionTreeClassifier(random_state=42)
tr_full.fit(X, y)
tr_full_predict = tr_full.predict(X.drop_duplicates())
tr_full_predict_proba = tr_full.predict_proba(X.drop_duplicates())[:, 1]

tr_d2 = tree.DecisionTreeClassifier(random_state=42, max_depth=2)
tr_d2.fit(X, y)
tr_d2_predict = tr_d2.predict(X.drop_duplicates())
tr_d2_predict_proba = tr_d2.predict_proba(X.drop_duplicates())[:, 1]

nb = naive_bayes.GaussianNB()
nb.fit(X, y)
nb_predict = nb.predict(X.drop_duplicates())
nb_proba = nb.predict(X.drop_duplicates())

# %%

plt.plot(X, y, 'o', color='royalblue')
plt.grid(True)
plt.title('Cerveja vs Aprovação')
plt.xlabel('Cervejas')
plt.ylabel('Aprovado')
plt.plot(X.drop_duplicates(), reg_predict, color='red')
plt.plot(X.drop_duplicates(), reg_proba, color='magenta')

plt.plot(X.drop_duplicates(), tr_full_predict, color='green')
plt.plot(X.drop_duplicates(), tr_full_predict_proba, color='brown')

plt.plot(X.drop_duplicates(), tr_d2_predict, color='purple')
plt.plot(X.drop_duplicates(), tr_d2_predict_proba, color='yellow')

plt.plot(X.drop_duplicates(), nb_predict, color='cyan')
plt.plot(X.drop_duplicates(), nb_proba, color='orange')

plt.hlines(0.5, xmin=1, xmax=9, linestyles='--', colors='black')

plt.legend([
    'Observação',
    'Reg Predict',
    'Reg Proba',
    'Tr Full Predict',
    'Tr Full Proba',
    'Tr d2 Predict',
    'Tr d2 proba',
    'Nb predict',
    'Nb proba',
])

# %%
