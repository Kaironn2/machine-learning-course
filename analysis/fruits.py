# %%

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / 'data'
file_path = DATA_DIR / 'dados_frutas.xlsx'
df = pd.read_excel(file_path)
df

# %%

from sklearn import tree

model = tree.DecisionTreeClassifier(random_state=42)

# %%

y = df['Fruta']

features = ['Arredondada', 'Suculenta', 'Vermelha', 'Doce']
x = df[features]

# %%

model.fit(x, y)

# %%
model.predict([[0, 0, 0, 0]])

# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model, feature_names=features, class_names=model.classes_, filled=True)

# %%

proba = model.predict_proba([[1, 1, 1, 1]])[0]
pd.Series(proba, index=model.classes_)

# %%
