from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

scores = []
for k in clusters_range:
    model = KMeans(n_clusters=k, random_state=42)
    predictions = model.fit_predict(X_scaled)

    scores.append(silhouette_score(X_scaled, predictions))

plt.plot(clusters_range, scores, marker='s')
plt.xlabel('$k$')
plt.ylabel('Silhouette score')
plt.show()

# 6-7?
