# %%

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / 'data'
file_path = DATA_DIR / 'dados_cerveja.xlsx'
df = pd.read_excel(file_path)
df

# %%

features = ['temperatura', 'copo', 'espuma', 'cor']
target = ['classe']

X = df[features]
y = df[target]

X = X.replace({'mud': 1, 'pint': 2, 'sim': 1, 'não': 0, 'clara': 0, 'escura': 1})

X
# %%

from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X, y)

# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model, feature_names=features, class_names=model.classes_, filled=True)
