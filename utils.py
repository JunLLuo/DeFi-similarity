import json
from copy import deepcopy
import pandas as pd


class NodeEdgeExtractor:
    def __init__(self):
        self.method_index = None
        self.address_index = None

    def get_edges_(self, row_subtraces, print_procedure=False):
        """
        Generates edges for the nodes in a subtrace pattern.

        Parameters:
        row_subtraces (str): A string representing the subtrace of a row.
        print_procedure (bool): If True, print the edge generation procedure.

        Returns:
        list: A list of edges where each edge is a pair [node1, node2].
        """
        edges = []
        row_subtraces_list = str(row_subtraces).split('_')
        row_subtraces_list = [eval(i) for i in row_subtraces_list]
        previous_skip = {}

        def find_edges(i, edges, degree, height, previous_skip, print_procedure=print_procedure):
            current_edges = 0
            max_edges = deepcopy(degree)
            j = 0
            skip = 0

            while j < len(row_subtraces_list[i + 1:]):
                edges.append([i, i + 1 + j])
                current_edges += 1
                if row_subtraces_list[i + 1 + j] > 0:
                    skip = find_edges(i + 1 + j, edges, row_subtraces_list[i + 1 + j], height + 1, previous_skip)
                if current_edges >= max_edges:
                    return skip + current_edges

                extra = 0
                for k, skips in previous_skip.items():
                    if k > height:
                        extra += sum(skips)
                        previous_skip[k] = []
                j = j + 1 + skip + extra

                if skip > 0:
                    previous_skip.setdefault(height, []).append(skip)
                    skip = 0

        find_edges(0, edges, row_subtraces_list[0], 0, previous_skip)
        return edges

    def get_nodes_(self, row, all_addresses=None, label=False, label_mode='3class', all_method_ids=None,
                   all_signatures=None, only_edge=False):
        """
        Generates features for the nodes in a graph.

        Parameters:
        row (pd.Series): A row from the dataframe.
        all_addresses (list): A list of all addresses (optional).
        label (bool): If True, categorize nodes into specific classes.
        label_mode (str): Specifies the labeling mode ('3class', 'node_group_labels').
        all_method_ids (list): A list of all method IDs (optional).
        all_signatures (list): A list of all signatures (optional).
        only_edge (bool): If True, label nodes with their edge attribute (method ID).

        Returns:
        dict: A dictionary of node features.
        """
        row_addresses_sub = row['addresses_sub']
        row_method_ids = row['MethodIds']
        row_signatures_sub = row['signatures_sub']
        row_node_group_labels = row['node_group_labels']

        # Initialize method index if not done yet
        if self.method_index is None and all_method_ids is not None:
            self.method_index = {method: idx for idx, method in enumerate(unique(all_method_ids))}

        # Label the nodes
        if label:
            address_set = unique(row_addresses_sub.split('_'))
            self.address_index = {}
            if label_mode == '3class':
                for i, address in enumerate(address_set):
                    if address.endswith('DEPLOYED'):
                        self.address_index[address] = 0
                    elif address == 'ASSET':
                        self.address_index[address] = 1
                    else:
                        self.address_index[address] = 2
            elif label_mode == 'node_group_labels':
                address_list = row_addresses_sub.split('_')
                node_group_labels_list = row_node_group_labels.split('_')
                self.address_index = {address: label for address, label in zip(address_list, node_group_labels_list)}
            else:
                raise ValueError('Unknown label_mode')
        else:
            # Assign unique labels to addresses or signatures
            if all_addresses is not None:
                self.address_index = {address: idx for idx, address in enumerate(unique(all_addresses))}
            elif all_signatures is not None:
                self.address_index = {sig: idx for idx, sig in enumerate(unique(all_signatures))}

        # Generate node features
        features = {}
        for i, (address, method, sig) in enumerate(
                zip(row_addresses_sub.split('_'), row_method_ids.split('_'), row_signatures_sub.split('_'))):
            if self.method_index and method in self.method_index:
                features[i] = self.method_index[
                    method] if only_edge else f"{self.method_index[method]}_{self.address_index[address]}"
            else:
                features[i] = self.address_index.get(sig, self.address_index[address])
        return features


def unique(sequence):
    """
    Returns a list of unique elements in a sequence while preserving order.

    Parameters:
    sequence (list): A list of elements.

    Returns:
    list: A list of unique elements.
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def generate_json_files(df, label_mode='node_group_labels', print_procedure=False):
    """
    Generates JSON files for each row in the dataframe with node features and edges.

    Parameters:
    df (pd.DataFrame): The dataframe containing the subtraces and node information.
    label_mode (str): The labeling mode to use for node features ('3class', 'node_group_labels').
    print_procedure (bool): If True, print the edge generation procedure.

    Returns:
    None: Creates JSON files in the current directory.
    """
    ner = NodeEdgeExtractor()

    for i, row in df.iterrows():
        edges = ner.get_edges_(row['subtraces'], print_procedure=print_procedure)

        # Uncomment one of the following lines depending on the desired labeling method

        # Option 1: Generate node 3 classes if label_mode is '3class'
        # features = ner.get_nodes_(row, all_addresses=None, all_method_ids=None, label=True, only_edge=False, label_mode='3class')

        # Option 2: Generate node features with node group labels
        features = ner.get_nodes_(row, all_addresses=None, all_method_ids=None, label=True, only_edge=False,
                                  label_mode=label_mode)

        # Option 3: Assign unique labels for each individual address
        # features = ner.get_nodes_(row, all_addresses=all_addresses, all_method_ids=None, label=False, only_edge=False)

        # Option 4: Label nodes based on the edge method (method ID)
        # features = ner.get_nodes_(row, all_addresses=all_addresses, all_method_ids=all_method_ids, label=False, only_edge=True)

        # Option 5: Label nodes based on signature
        # features = ner.get_nodes_(row, all_addresses=None, all_method_ids=None, label=False, all_signatures=all_signatures, only_edge=False)

        # Create dictionary for output
        dict_out = {'edges': edges, 'features': features}

        # Write to JSON file
        with open(f'{i}.json', 'w') as f:
            json.dump(dict_out, f)

        print(f'Generated JSON for row {i}')


if __name__ == '__main__':
    df = pd.read_pickle('./temp_df.pkl')
    generate_json_files(df, label_mode='node_group_labels')
