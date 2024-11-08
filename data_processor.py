import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.sparse import lil_matrix
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform


def read_in(file_path):
    """
    Reads in a file and returns a DataFrame.
    Supports reading from pickle or CSV based on the file extension.

    Parameters:
    file_path (str): The path to the file to be read.

    Returns:
    pd.DataFrame: The data from the file as a DataFrame.
    """
    if file_path.endswith('.pkl'):
        return pd.read_pickle(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .pkl or .csv file.")


def parse_node_signatures(signature_str):
    """
    Parses a string of node signatures, splitting them by underscores and
    converting each component into a Python list using `ast.literal_eval`.

    Parameters:
    signature_str (str): The string containing node signatures separated by underscores.

    Returns:
    list: A list of node signatures, each represented as a Python list.
    """
    node_signatures = signature_str.split('_')
    return [ast.literal_eval(node or '[]') for node in node_signatures]


def cal_similarity(set1, set2, type_='jaccard'):
    """
    Calculates the similarity between two sets using the specified metric.

    Parameters:
    set1 (set): The first set.
    set2 (set): The second set.
    type_ (str): The type of similarity metric to use. Options are 'jaccard', 'dice', or 'overlap'.

    Returns:
    float: The calculated similarity between the two sets.
    """
    if type_ == 'jaccard':
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0
    elif type_ == 'dice':
        intersection = len(set1.intersection(set2))
        return (2. * intersection) / (len(set1) + len(set2)) if set1 or set2 else 1.0
    elif type_ == 'overlap':
        intersection = len(set1.intersection(set2))
        return intersection / min(len(set1), len(set2)) if set1 and set2 else 1.0
    else:
        raise ValueError('Unknown similarity type')


def get_signatures_similarity_matrix(df):
    """
    Generates a similarity matrix for unique node signatures.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the node signatures.

    Returns:
    tuple: A dictionary representing the similarity matrix and a list of unique node sets.
    """
    df['node_signatures'] = df['signatures_sub'].apply(parse_node_signatures)
    unique_nodes = set()

    for index, row in df.iterrows():
        for node_str in row['signatures_sub'].split('_'):
            node_set = set(eval(node_str))
            unique_nodes.add(frozenset(node_set))

    unique_nodes = list(unique_nodes)
    num_unique_nodes = len(unique_nodes)
    similarity_matrix = {}

    for i in range(num_unique_nodes):
        for j in range(i + 1, num_unique_nodes):
            similarity = cal_similarity(unique_nodes[i], unique_nodes[j])
            similarity_matrix[(unique_nodes[i], unique_nodes[j])] = similarity

    return similarity_matrix, unique_nodes


def get_signatures_dense_matrix(similarity_matrix, unique_nodes):
    """
    Converts the similarity matrix to a dense format.

    Parameters:
    similarity_matrix (dict): The dictionary containing the similarity values between node pairs.
    unique_nodes (list): A list of unique node sets.

    Returns:
    np.ndarray: A dense similarity matrix.
    """
    num_unique_nodes = len(unique_nodes)
    sparse_matrix = lil_matrix((num_unique_nodes, num_unique_nodes))

    for (node1, node2), similarity in similarity_matrix.items():
        i = unique_nodes.index(node1)
        j = unique_nodes.index(node2)
        sparse_matrix[i, j] = similarity
        sparse_matrix[j, i] = similarity

    dense_matrix = sparse_matrix.todense()
    return dense_matrix


def visualize_dense_to_distance_matrix(dense_matrix):
    """
    Converts the dense similarity matrix to a distance matrix and visualizes it.

    Parameters:
    dense_matrix (np.ndarray): A dense similarity matrix.

    Returns:
    None: Displays the heatmap of the similarity matrix.
    """
    distance_matrix = 1 - dense_matrix
    np.fill_diagonal(distance_matrix, 0)
    condensed_distance_matrix = squareform(distance_matrix)

    Z = linkage(condensed_distance_matrix, method='ward')
    dendro_order = leaves_list(Z)
    ordered_matrix = distance_matrix[dendro_order, :][:, dendro_order]

    plt.figure(figsize=(10, 8))
    sns.heatmap(1 - ordered_matrix, cmap='viridis')  # Display as similarity heatmap
    plt.title('Ordered Jaccard Similarity Heatmap between Unique Nodes')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')

    plt.show()


def signatures_to_cluster_labels(signatures_sub, node_to_cluster_map):
    """
    Maps node signatures to their respective cluster labels.

    Parameters:
    signatures_sub (str): The string containing node signatures.
    node_to_cluster_map (dict): A mapping of node sets to cluster labels.

    Returns:
    str: A string of cluster labels separated by underscores.
    """
    cluster_labels = []
    for node_str in signatures_sub.split('_'):
        if node_str:
            node_set = frozenset(eval(node_str))
            cluster_label = node_to_cluster_map.get(node_set, None)
            cluster_labels.append(str(cluster_label))
    return '_'.join(cluster_labels)


def make_node_group_labels(df, dense_matrix, unique_nodes, threshold=1.5):
    """
    Groups node signatures into clusters and assigns group labels.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the node signatures.
    dense_matrix (np.ndarray): A dense similarity matrix.
    unique_nodes (list): A list of unique node sets.
    threshold (float): The clustering threshold for the linkage method.

    Returns:
    pd.DataFrame: The DataFrame with the added 'node_group_labels' column.
    """
    distance_matrix = 1 - dense_matrix
    Z = linkage(distance_matrix, 'ward')

    # Generate the clusters
    clusters = fcluster(Z, threshold, criterion='distance')

    print(f'{len(set(clusters))} grouped labels of {len(unique_nodes)} node labels')

    # Map from unique node to its cluster label
    node_to_cluster = {node: cluster for node, cluster in zip(unique_nodes, clusters)}

    # Apply cluster labels to the DataFrame
    df['node_group_labels'] = df['signatures_sub'].apply(lambda x: signatures_to_cluster_labels(x, node_to_cluster))

    return df


def categorize_transaction(method):
    """
    Categorizes a transaction method string based on patterns related to capital locking, borrowing,
    repaying, swapping, liquidating, claiming rewards, or governance actions.

    Parameters:
    method (str): The transaction method string.

    Returns:
    str: The category of the transaction ('lock_capital', 'borrow', 'repay', 'redeem_withdraw',
         'swap', 'liquidate', 'interest_rewards', 'governance', or 'other').
    """
    if not isinstance(method, str):
        return 'other'

    # Extract the base method name before any parentheses and convert to lowercase
    method = method.split('(')[0].lower()

    if re.search(
            r'deposit|add.*(liquidity)|(?<!un)stake(?!.*unstake)|staking|(?<!b)lock(?!.*unlock)|lend|collateralize',
            method):
        return 'lock_capital'
    elif re.search(r'borrow', method):
        return 'borrow'
    elif re.search(r'repay', method):
        return 'repay'
    elif re.search(r'withdraw|remove.*(liquidity)|unstake|unlock', method):
        return 'redeem_withdraw'
    elif re.search(r'swap|exchange', method):
        return 'swap'
    elif re.search(r'liquidate|liquidation', method):
        return 'liquidate'
    elif re.search(r'(get|claim).*(reward|fee)|harvest|earn', method):
        return 'interest_rewards'
    elif re.search(r'vote', method):
        return 'governance'
    else:
        return 'other'


def calculate_depth(subtrace_pattern):
    """
    Calculates the maximum depth of a subtrace pattern based on the number of children
    nodes in the trace, represented as underscore-separated integers.

    Parameters:
    subtrace_pattern (str): A string representing the trace pattern, with numbers indicating
                            the number of children nodes and underscores separating them.

    Returns:
    int: The maximum depth of the subtrace pattern.
    """
    nodes = subtrace_pattern.split('_')

    current_depth = 0
    max_depth = 0
    children_stack = []

    for node in nodes:
        children = int(node) if node.isdigit() else 0

        if children > 0:
            current_depth += 1
            children_stack.append(children - 1)
            max_depth = max(max_depth, current_depth)
        else:
            while children_stack and children_stack[-1] == 0:
                children_stack.pop()
                current_depth -= 1
            if children_stack:
                children_stack[-1] -= 1

    return max_depth


if __name__ == '__main__':
    df = read_in(f'data/building_blocks_10k.csv')

    similarity_matrix, unique_nodes = get_signatures_similarity_matrix(df)
    dense_matrix = get_signatures_dense_matrix(similarity_matrix, unique_nodes)

    df = make_node_group_labels(df, dense_matrix, unique_nodes, threshold=1.5)

    df['refined_category'] = df.apply(
        lambda x: categorize_transaction(x['MethodName']),
        axis=1
    )

    df.to_pickle('./temp_df.pkl')

    print(Counter(df['refined_category']))

