import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

clusters_range = np.arange(2, 16, 1, dtype=int)
df = pd.read_csv('voice_features.csv')
X = df.drop(columns=['id', 'symptoms_score'])

categorical = ['stimulus']
numeric_data = X.select_dtypes([np.number])
numeric_features = numeric_data.columns

column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical),
    ('scaling', StandardScaler(), numeric_features)
])

X_scaled = column_transformer.fit_transform(X)

gm_bic = []
for k in clusters_range:
    gm = GaussianMixture(n_components=k, random_state=42)
    gm.fit(X_scaled)
    gm_bic.append(gm.bic(X_scaled))

plt.plot(clusters_range, gm_bic, marker='s')
plt.xlabel('$k$')
plt.ylabel('$BIC$')
plt.show()

# 7-8?
