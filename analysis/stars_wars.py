# %%

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / 'data'
file_path = DATA_DIR / 'dados_clones.parquet'
df = pd.read_parquet(file_path, 'fastparquet')

# %%

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df: pd.DataFrame = df.applymap(
    lambda v: v.strip().lower().replace(' ', '_') if isinstance(v, str) else v
)

df.head()

# %%

df = df.drop('general_jedi_encarregado', axis=1)

# %%

features = [
    'p2o_master_id',
    'massa(em_kilos)',
    'estatura(cm)',
    'distância_ombro_a_ombro',
    'tamanho_do_crânio',
    'tamanho_dos_pés',
    'tempo_de_existência(em_meses)',
]
target = ['status']

X = df[features]
y = df[target]

general_replaces = {
    'yoda': 1,
    'shaak_ti': 2,
    'obi-wan_kenobi': 3,
    'aayla_secura': 4,
    'mace_windu': 5,
}
type_replaces = {'tipo_1': 1, 'tipo_2': 2, 'tipo_3': 3, 'tipo_4': 4, 'tipo_5': 5}
replaces = general_replaces | type_replaces
X = X.replace(replaces)

X

# %%

from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X, y)

# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model, feature_names=features, class_names=model.classes_, filled=True, max_depth=3)

# %%
