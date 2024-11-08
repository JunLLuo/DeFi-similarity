import os
import json
import random

import numpy as np
import networkx as nx
import pandas as pd

from natsort import natsorted
from collections import Counter
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import confusion_matrix, silhouette_score, normalized_mutual_info_score, mutual_info_score, \
    homogeneity_completeness_v_measure, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder
from karateclub.graph_embedding import Graph2Vec


# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)


def read_jsons_to_graphs(path_to_json):
    """
    Reads JSON files from the specified directory, converts them into networkx graphs,
    and adds node features.

    Parameters:
    path_to_json (str): The path to the directory containing the JSON files.

    Returns:
    list: A list of networkx graphs with node features.
    """
    # List all JSON files in the directory and sort them naturally
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    json_files = natsorted(json_files)

    graphs = []
    for data_path in json_files:

        with open(os.path.join(path_to_json, data_path), 'r') as file:
            data = json.load(file)

        graph = nx.from_edgelist(data["edges"])

        for node, feature in data['features'].items():
            if int(node) in graph.nodes:
                graph.nodes[int(node)]['feature'] = feature

        graphs.append(graph)

    return graphs


def purity_score(y_true, y_pred):
    """
    Calculates the purity score for clustering.

    Parameters:
    y_true (array-like): Ground truth labels.
    y_pred (array-like): Predicted cluster labels.

    Returns:
    float: Purity score.
    """
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


def get_dbscan_label(embeddings, eps=0.1):
    """
    Performs DBSCAN clustering on the embeddings and returns the labels and a dictionary of clusters.

    Parameters:
    embeddings (array-like): Embeddings to cluster.
    eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.

    Returns:
    tuple: DBSCAN labels and a dictionary with cluster labels as keys and corresponding indices as values.
    """
    dbscan_embeddings_array = np.array(embeddings)
    dbscan = DBSCAN(eps=eps, min_samples=5, metric='cosine')
    dbscan.fit(dbscan_embeddings_array)

    label_dict = {}
    for i, label in enumerate(dbscan.labels_):
        label_dict.setdefault(label, []).append(i)

    print(f'In total: {len(set(dbscan.labels_))} clusters')
    return dbscan.labels_, label_dict


def get_agglomerative_labels(embeddings, n_clusters=None, distance_threshold=None):
    """
    Performs Agglomerative clustering on the embeddings and returns the labels and a dictionary of clusters.

    Parameters:
    embeddings (array-like): Embeddings to cluster.
    n_clusters (int): The number of clusters to find. If None, a distance threshold is used.
    distance_threshold (float): The linkage distance threshold for clustering.

    Returns:
    tuple: Agglomerative labels and a dictionary with cluster labels as keys and corresponding indices as values.
    """
    embeddings_array = np.array(embeddings)
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold,
                                            affinity='cosine', linkage='complete')
    agglomerative.fit(embeddings_array)

    label_dict = {}
    for i, label in enumerate(agglomerative.labels_):
        label_dict.setdefault(label, []).append(i)

    print(f'In total: {len(set(agglomerative.labels_))} clusters')
    return agglomerative.labels_, label_dict


def get_the_rows_from_cluster(df, dbscan_labels, label_dict, cluster_index, print_label_cts=False):
    """
    Retrieves rows from the dataframe that belong to a specific cluster.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    dbscan_labels (array-like): The labels from the DBSCAN clustering.
    label_dict (dict): A dictionary of cluster labels and their corresponding indices.
    cluster_index (int): The index of the cluster to retrieve.
    print_label_cts (bool): Whether to print the counts of each cluster.

    Returns:
    pd.DataFrame: A dataframe containing the rows for the specified cluster.
    """
    label_counts = Counter(dbscan_labels)
    if print_label_cts:
        print(label_counts)

    cluster_ids = [k for (k, _) in label_counts.most_common()]
    print(f'Cluster {cluster_ids[cluster_index]}: {len(label_dict[cluster_ids[cluster_index]])} members')

    return df.loc[label_dict[cluster_ids[cluster_index]]][
        ['to_protocol', 'MethodName', 'MethodId', 'subtraces', 'signatures_sub']]


def perform_clustering(embeddings, df, n_clusters=None, distance_threshold=None):
    """
    Performs clustering using Agglomerative clustering and adds the cluster labels to the dataframe.

    Parameters:
    embeddings (array-like): Embeddings to cluster.
    df (pd.DataFrame): The dataframe to which cluster labels will be added.
    n_clusters (int): The number of clusters to find. If None, a distance threshold is used.
    distance_threshold (float): The linkage distance threshold for clustering.

    Returns:
    tuple: The updated dataframe with cluster labels, the prediction labels, and a dictionary of clusters.
    """
    prediction_labels, label_dict = get_agglomerative_labels(embeddings, n_clusters=n_clusters,
                                                             distance_threshold=distance_threshold)
    df['cluster_'] = prediction_labels
    return df, prediction_labels, label_dict


def evaluation(df_test, embeddings, gt_label='refined_category'):
    """
    Evaluates the clustering results using various metrics.

    Parameters:
    df_test (pd.DataFrame): The dataframe containing test data with ground truth labels.
    embeddings (array-like): The embeddings used for clustering.
    gt_label (str): The column name of the ground truth labels in the dataframe.

    Returns:
    None: Prints evaluation metrics such as NMI, mutual information, homogeneity, completeness, V-measure, purity, silhouette score, and Calinski-Harabasz index.
    """
    encoder = LabelEncoder()
    true_labels = encoder.fit_transform(df_test[gt_label])

    nmi_score = normalized_mutual_info_score(true_labels, df_test['cluster_'])
    print(f"Normalized Mutual Information: {nmi_score}")

    unnorm_nmi_score = mutual_info_score(true_labels, df_test['cluster_'])
    print(f"Mutual Information: {unnorm_nmi_score}")

    h, c, v = homogeneity_completeness_v_measure(true_labels, df_test['cluster_'])
    print(f"Homogeneity: {h}, Completeness: {c}, V-measure: {v}")

    purity = purity_score(true_labels, df_test['cluster_'])
    print(f"Purity: {purity}")

    filtered_embeddings = embeddings[df_test.index]
    silhouette = silhouette_score(filtered_embeddings, df_test['cluster_'], metric='cosine')
    print(f"Silhouette Coefficient: {silhouette}")

    calinski_harabasz = calinski_harabasz_score(filtered_embeddings, df_test['cluster_'])
    print(f"Calinski-Harabasz Index: {calinski_harabasz}")

