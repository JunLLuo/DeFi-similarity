import matplotlib
matplotlib.use('Agg')

import sklearn.manifold

from math import log
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import pandas as pd

sns.set_theme(style="white", font_scale=2)


def visualize_embedding(embeddings, df, true_label='to_protocol'):
    """
    Visualizes embeddings using t-SNE and plots them with custom labels and markers.

    Parameters:
    embeddings (array-like): Embeddings to reduce and visualize.
    df (pd.DataFrame): DataFrame containing cluster, label, and metadata information.
    true_label (str): The label to use for coloring the visualization ('to_protocol' or 'refined_category').

    Returns:
    None: Displays the t-SNE plot.
    """
    # Perform t-SNE dimensionality reduction
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
    vectors_matrix_2d = tsne.fit_transform(embeddings)

    # Prepare data points for plotting
    points = pd.DataFrame(
        [
            (coords[0], coords[1], i, pred_cluster, true_lbl, log(ct))
            for coords, pred_cluster, i, true_lbl, ct, subtraces in [
                (vectors_matrix_2d[i], df['cluster_'].iloc[i], i, df[true_label].iloc[i], df['count'].iloc[i], df['subtraces'].iloc[i])
                for i, _ in df.iterrows()
            ]
            if len(subtraces) != 1 and true_lbl != 'other'  # Exclude 'other' and certain building blocks
        ],
        columns=["x", "y", "b_index", "pred_cluster", "true_label", "ct"]
    )

    # Mapping for relabeling true labels
    new_labels = {
        'uniswap': 'Uniswap', '0x': '0x', 'sushiswap': 'Sushiswap', 'aave': 'Aave', 'synthetix': 'Synthetix',
        '1inch': '1inch', 'badger': 'Badger', 'balancer': 'Balancer', 'maker': 'Maker', 'compound': 'Compound',
        'dydx': 'dYdX', 'curvefinance': 'Curve Finance', 'convex': 'Convex', 'renvm': 'RenVM', 'fei': 'Fei',
        'hegic': 'Hegic', 'instadapp': 'Instadapp', 'nexus': 'Nexus', 'vesper': 'Vesper', 'harvestfinance': 'Harvest Finance',
        'barnbridge': 'Barnbridge', 'futureswap': 'Futureswap', 'yearn': 'Yearn'
    }
    points['label_display'] = points['true_label'].map(new_labels)

    # Create a unique set of labels for color mapping
    labels_unique = points['label_display'].unique()
    base_palette = sns.color_palette("colorblind", n_colors=len(labels_unique))

    # Swap certain palette colors to align better with the desired mapping
    base_palette[2], base_palette[8] = base_palette[8], base_palette[2]
    base_palette[10], base_palette[16] = base_palette[16], base_palette[10]

    # Create marker styles for each set of labels
    label_to_marker = {label: 'o' for label in labels_unique[:8]}
    label_to_marker.update({label: 's' for label in labels_unique[8:16]})
    label_to_marker.update({label: 'X' for label in labels_unique[16:]})
    points['marker_style'] = points['label_display'].map(label_to_marker)

    # Create the plot
    g = sns.relplot(
        data=points, x='x', y='y', hue='label_display', style='marker_style',
        aspect=1.0, height=15, palette=base_palette, sizes=(140, 800), legend=False
    )

    # Create a custom legend using different markers
    marker_mapping = {
        'Uniswap': 'o', '0x': 'o', 'Sushiswap': 'o', 'Aave': 'o',
        'Synthetix': 'o', '1inch': 'o', 'Badger': 'o', 'Balancer': 'o',
        'Maker': 's', 'Compound': 's', 'dYdX': 's', 'Curve Finance': 's',
        'Convex': 's', 'RenVM': 's', 'Fei': 's', 'Hegic': 's',
        'Instadapp': 'X', 'Nexus': 'X', 'Vesper': 'X', 'Harvest Finance': 'X',
        'Barnbridge': 'X', 'Futureswap': 'X', 'Yearn': 'X'
    }
    legend_handles = [
        mlines.Line2D([], [], color=base_palette[i], marker=marker, linestyle='None', markersize=16, label=label)
        for i, (label, marker) in enumerate(marker_mapping.items())
    ]
    g.fig.legend(handles=legend_handles, title="Label Display", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set axis labels
    g.set_axis_labels('t-SNE 1', 't-SNE 2')

    # Annotate points with their index (building block)
    for ax in g.axes.ravel():
        for i, txt in enumerate(points['b_index']):
            ax.annotate(txt, (points['x'][i], points['y'][i]), fontsize=1.6)

    plt.gcf().set_dpi(300)
    plt.savefig('embedding_space_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

