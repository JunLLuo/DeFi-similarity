import warnings
import pandas as pd

from data_processor import calculate_depth
from model import read_jsons_to_graphs, Graph2Vec, perform_clustering, evaluation
from visualizer import visualize_embedding

warnings.simplefilter("ignore")

path_to_processed_bbs = 'processed_data/signatures_group/'


if __name__ == '__main__':

    print(f"loading the builing blocks jsons with node feature: {path_to_processed_bbs.split('/')[-2]}")
    graphs = read_jsons_to_graphs(path_to_processed_bbs)
    print(f"{len(graphs)} building blocks loaded")
    model = Graph2Vec(dimensions=128, epochs=100, attributed=True, learning_rate=0.05, min_count=0, down_sampling=0,
                      seed=6,
                      workers=1)
    print(f"graph representation model fitting")
    model.fit(graphs)

    embeddings = model.get_embedding()
    print(f'{len(embeddings)} embedding vector generated')

    # run data_professor to generate this processed data frame
    df = pd.read_pickle('./temp_df.pkl')

    # perform clustering
    df, prediction_labels, label_dict = perform_clustering(embeddings, df, n_clusters=None, distance_threshold=0.6)

    # evaluate and plot the metrics
    filtered_df = df[df['subtraces'].apply(calculate_depth) != 0]
    evaluation(filtered_df, embeddings, gt_label='refined_category')

    # plot the 2d visualization
    visualize_embedding(embeddings, df, true_label='to_protocol')