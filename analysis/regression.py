# %%

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / 'data'
file_path = DATA_DIR / 'dados_cerveja_nota.xlsx'
df = pd.read_excel(file_path)
df

# %%

from sklearn import linear_model
from sklearn import tree

X = df[['cerveja']]
y = df['nota']

reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(X, y)

# %%

a, b = reg.intercept_, reg.coef_[0]
print(a, b)

# %%

predict_reg = reg.predict(X.drop_duplicates())

tr_full = tree.DecisionTreeRegressor(random_state=42)
tr_full.fit(X, y)
predict_tr_full = tr_full.predict(X.drop_duplicates())

tr_d2 = tree.DecisionTreeRegressor(random_state=42, max_depth=2)
tr_d2.fit(X, y)
predict_tr_d2 = tr_d2.predict(X.drop_duplicates())

# %%

import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title('Relação Cerveja vs Nota')
plt.xlabel('Cerveja')
plt.ylabel('Nota')

plt.plot(X.drop_duplicates()['cerveja'], predict_reg)
plt.plot(X.drop_duplicates()['cerveja'], predict_tr_full, color='black')
plt.plot(X.drop_duplicates()['cerveja'], predict_tr_d2, color='magenta')

plt.legend(['Observado', f'y = {a:.3f} + {b:.3f} x', 'Árvore Full', 'Árvore D2'])

# %%

plt.figure(dpi=400)

tree.plot_tree(tr_d2, feature_names=['cerveja'], filled=True)

# %%
