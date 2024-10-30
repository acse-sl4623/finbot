import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from finbot.rag_chain_route import RAGChain
from finbot.vectorstore import QdrantVectorStore


# Perform t-SNE
def plot_embeddings(embedding_vectors_np, metadata):
    tsne = TSNE(n_components=2, random_state=0)
    vectors_2d = tsne.fit_transform(embedding_vectors_np)

    # Define color mapping for specific asset classes
    predefined_colors = {
        'Equity': 'blue',
        'Fixed Interest': 'red',
        'Mixed Asset': 'green'
    }

    # Find all unique classes in metadata
    unique_classes = list(set(metadata))

    # Assign random colors to any class not in predefined_colors
    color_map = plt.get_cmap('tab10', len(unique_classes))  # Using a colormap with enough colors
    class_to_color = {cls: predefined_colors.get(cls, color_map(i)) for i, cls in enumerate(unique_classes)}

    # Create an array of colors for each point
    colors = [class_to_color[cls] for cls in metadata]

    # Plot t-SNE result with color coding
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=colors, alpha=0.5)
    plt.title('t-SNE Visualization of Vector Similarity')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Create a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=class_to_color[cls], markersize=10) 
            for cls in unique_classes]
    plt.legend(handles, unique_classes, title='Asset Class')

    plt.show()

def get_vectors_from_collection(collection_name, section):
    vs = QdrantVectorStore(collection_name)
    response = vs.client.get_collection(collection_name)

    scroll_response = vs.client.scroll(
        collection_name=collection_name,
        limit=5000,
        with_vectors=True,
        with_payload=True
    )

    points = scroll_response[0]
    print(f"Number of points in collection {collection_name}: {len(points)}", '\n')

    asset_classes = [point.payload['metadata']['asset_class'] for point in points]
    if 'docs' in collection_name or 'obj' in collection_name:
        embedding_vectors_np = np.array([point.vector for point in points])
    else:
        embedding_vectors = []
        for point in points:
            if 'section_name' in point.payload['metadata'].keys():
                if point.payload['metadata']['section_name'] == f"{section.capitalize} Allocation":
                    embedding_vectors.append(point.vector)
        embedding_vectors_np = np.array(embedding_vectors)
        print(f"Number of points in collection {collection_name} for {section} allocation: {len(embedding_vectors)}", '\n')
    plot_embeddings(embedding_vectors_np, asset_classes)

def get_Fidelity_cluster(collection_name):
    vs = QdrantVectorStore(collection_name)
    response = vs.client.get_collection(collection_name)

    scroll_response = vs.client.scroll(
        collection_name=collection_name,
        limit=5000,
        with_vectors=True,
        with_payload=True
    )

    points = scroll_response[0]
    embedding_vectors = []
    labels = []
    asset_classes = []
    if 'docs' in collection_name or 'obj' in collection_name:
        for point in points:
            if 'Fidelity' in point.payload['metadata']['fund_name']:
                embedding_vectors.append(point.vector)
                labels.append(point.payload['metadata']['fund_name'])
                asset_classes.append(point.payload['metadata']['asset_class'])
        print(f"Number of points in collection {collection_name} for Fidelity funds: {len(embedding_vectors)}", '\n')
        embedding_vectors_np = np.array(embedding_vectors)
        unique_asset_classes = list(set(asset_classes))
        print("Asset classes:", unique_asset_classes)
     # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    vectors_2d = tsne.fit_transform(embedding_vectors_np)

    # Define color mapping for asset classes
    predefined_colors = {
        'Equity': 'blue',
        'Fixed Interest': 'red',
        'Mixed Asset': 'green'
    }

    # Assign random colors to any class not in predefined_colors
    color_map = plt.get_cmap('tab10', len(unique_asset_classes))  # Using a colormap with enough colors
    class_to_color = {cls: predefined_colors.get(cls, color_map(i)) for i, cls in enumerate(unique_asset_classes)}

    # Create an array of colors for each point
    colors = [class_to_color[cls] for cls in asset_classes]

    # Plot with labels and color coding
    plt.figure(figsize=(12, 10))

    for i in range(len(vectors_2d)):
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1], color=colors[i], alpha=0.7, edgecolors='w', s=100)
        # Label only a subset of points for better visibility
        if i % 3 == 0:  # Adjust the condition to place labels as needed
            plt.text(vectors_2d[i, 0], vectors_2d[i, 1], labels[i], fontsize=7, ha='right')

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=class_to_color[cls], markersize=10) 
            for cls in unique_asset_classes]
    plt.legend(handles, unique_asset_classes, title='Asset Class')
    plt.title('t-SNE Visualization of Fidelity Funds')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.show()

def get_cluster(collection_name, cluster: str):
    vs = QdrantVectorStore(collection_name)
    response = vs.client.get_collection(collection_name)

    scroll_response = vs.client.scroll(
        collection_name=collection_name,
        limit=5000,
        with_vectors=True,
        with_payload=True
    )

    points = scroll_response[0]
    embedding_vectors = []
    labels = []
    asset_classes = []
    if 'docs' in collection_name or 'obj' in collection_name:
        for point in points:
            if cluster in point.payload['metadata']['fund_name']:
                embedding_vectors.append(point.vector)
                labels.append(point.payload['metadata']['fund_name'])
                asset_classes.append(point.payload['metadata']['asset_class'])
        print(f"Number of points in collection {collection_name} for {cluster} funds: {len(embedding_vectors)}", '\n')
        embedding_vectors_np = np.array(embedding_vectors)
        unique_asset_classes = list(set(asset_classes))
        print("Asset classes:", unique_asset_classes)
    # Perform t-SNE
    perplexity = min(5, len(embedding_vectors) - 1)  # Set a lower perplexity value
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    vectors_2d = tsne.fit_transform(embedding_vectors_np)

    # Define color mapping for asset classes
    predefined_colors = {
        'Equity': 'blue',
        'Fixed Interest': 'red',
        'Mixed Asset': 'green'
    }

    # Assign random colors to any class not in predefined_colors
    color_map = plt.get_cmap('tab10', len(unique_asset_classes))  # Using a colormap with enough colors
    class_to_color = {cls: predefined_colors.get(cls, color_map(i)) for i, cls in enumerate(unique_asset_classes)}

    # Create an array of colors for each point
    colors = [class_to_color[cls] for cls in asset_classes]

    # Plot with labels and color coding
    plt.figure(figsize=(12, 10))

    for i in range(len(vectors_2d)):
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1], color=colors[i], alpha=0.7, edgecolors='w', s=100)
        # Label only a subset of points for better visibility
        if i % 3 == 0:  # Adjust the condition to place labels as needed
            plt.text(vectors_2d[i, 0], vectors_2d[i, 1], labels[i], fontsize=7, ha='right')

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=class_to_color[cls], markersize=10) 
            for cls in unique_asset_classes]
    plt.legend(handles, unique_asset_classes, title='Asset Class')
    plt.title(f"t-SNE Embeddings Visualization of {cluster} Funds")
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.show()

collection_name = "all_obj_kb_clean_inv"
cluster = "Income"
get_cluster(collection_name, cluster)