import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
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

participants = pd.read_excel("PsychiatricDiscourse_participant_data.xlsx")
schizophrenia_only = participants.loc[
    (participants['depression.symptoms'] == 0.) &
    (participants['thought.disorder.symptoms'] != 0.)
]

inertia = []
for k in clusters_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(np.sqrt(kmeans.inertia_))

plt.plot(clusters_range, inertia, marker='s')
plt.xlabel('$k$')
plt.ylabel('$J(C_k)$')
plt.show()

# 6-8?
