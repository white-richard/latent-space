import matplotlib.pyplot as plt
import pacmap


def visualize_embedding(X, y, save_path="./tmp/pacmap_embeddings.png"):
    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)

    # fit the data (The index of transformed data corresponds to the index of the original data)
    X_transformed = embedding.fit_transform(X, init="pca")

    # visualize the embedding
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap="Spectral", c=y, s=0.6)
    fig.savefig(save_path)
