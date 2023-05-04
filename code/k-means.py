import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score


def learn(dataframe, s, model):
    true_predictions = list(dataframe[dataframe['stimulus'] == s].symptoms_score - 1)
    X = dataframe[dataframe['stimulus'] == s].drop(columns=['id', 'symptoms_score', 'stimulus'])
    numeric_data = X.select_dtypes([np.number])
    numeric_features = numeric_data.columns
    column_transformer = ColumnTransformer([
        ('scaling', StandardScaler(), numeric_features)
    ])
    X_scaled = column_transformer.fit_transform(X)

    nmi_res = []
    ari_res = []
    if model == 'kmeans':
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=2, init='k-means++',)  # random_state=k
            kmeans.fit(X_scaled)
            current_predictions = list(kmeans.predict(X_scaled))
            nmi_res.append(normalized_mutual_info_score(true_predictions, current_predictions))
            ari_res.append(adjusted_rand_score(true_predictions, current_predictions))

        mean_nmi = np.mean(nmi_res)
        std_nmi = np.std(nmi_res)

        mean_ari = np.mean(ari_res)
        std_ari = np.std(ari_res)

        return X_scaled, kmeans, mean_nmi, mean_ari
    elif model == 'spectral':
        spectral = SpectralClustering(n_components=2)
        current_predictions = spectral.fit_predict(X_scaled)
        nmi = normalized_mutual_info_score(true_predictions, current_predictions)
        ari = adjusted_rand_score(true_predictions, current_predictions)

        return X_scaled, spectral, nmi, ari


def feature_importance(X, model, cluster_number):
    cluster = X[np.where(model.labels_ == cluster_number)]
    cluster_means = np.mean(cluster, axis=0)
    grand_mean = np.mean(X, axis=0)
    diff = np.subtract(cluster_means, grand_mean)
    rel_diff = np.divide(diff, grand_mean)
    return rel_diff


df = pd.read_csv('voice_features_red.csv')

X_kpic, model_kpic, nmi_kpic, ari_kpic = learn(df, 'pic', 'kmeans')
X_kins, model_kins, nmi_kins, ari_kins = learn(df, 'instr', 'kmeans')
X_kpers, model_kpers, nmi_kpers, ari_kpers = learn(df, 'pers', 'kmeans')
X_spic, model_spic, nmi_spic, ari_spic = learn(df, 'pic', 'spectral')
X_sins, model_sins, nmi_sins, ari_sins = learn(df, 'instr', 'spectral')
X_spers, model_spers, nmi_spers, ari_spers = learn(df, 'pers', 'spectral')

print("KMEANS:")
print('Mean NMI scores for "pic", "instr" and "pers" respectively:', nmi_kpic, ',', nmi_kins, ',', nmi_kpers)
print('Mean ARI scores for "pic", "instr" and "pers" respectively:', ari_kpic, ',', ari_kins, ',', ari_kpers)
print('------------------------------------')
print("Importance of features in 0 cluster (pic stimulus):")
print(feature_importance(X_kpic, model_kpic, 0))
print("Importance of features in 1 cluster (pic stimulus):")
print(feature_importance(X_kpic, model_kpic, 1))
print("Importance of features in 0 cluster (instr stimulus):")
print(feature_importance(X_kins, model_kins, 0))
print("Importance of features in 1 cluster (instr stimulus):")
print(feature_importance(X_kins, model_kins, 1))
print("Importance of features in 0 cluster (pers stimulus):")
print(feature_importance(X_kpers, model_kpers, 0))
print("Importance of features in 1 cluster (pers stimulus):")
print(feature_importance(X_kpers, model_kpers, 1))

print("SPECTRAL:")
print('Mean NMI scores for "pic", "instr" and "pers" respectively:', nmi_spic, ',', nmi_sins, ',', nmi_spers)
print('Mean ARI scores for "pic", "instr" and "pers" respectively:', ari_spic, ',', ari_sins, ',', ari_spers)
print('------------------------------------')
print("Importance of features in 0 cluster (pic stimulus):")
print(feature_importance(X_spic, model_spic, 0))
print("Importance of features in 1 cluster (pic stimulus):")
print(feature_importance(X_spic, model_spic, 1))
print("Importance of features in 0 cluster (instr stimulus):")
print(feature_importance(X_sins, model_sins, 0))
print("Importance of features in 1 cluster (instr stimulus):")
print(feature_importance(X_sins, model_sins, 1))
print("Importance of features in 0 cluster (pers stimulus):")
print(feature_importance(X_spers, model_spers, 0))
print("Importance of features in 1 cluster (pers stimulus):")
print(feature_importance(X_spers, model_spers, 1))

