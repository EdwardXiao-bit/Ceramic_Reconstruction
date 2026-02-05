import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_geo_embeddings(
    embeddings,
    ids=None,
    method="pca",
    figsize=(6, 6)
):
    """
    几何 embedding 可视化（PCA / t-SNE）

    :param embeddings: List[np.ndarray]  每个 shape = (D,)
    :param ids: List[int] or None         碎片ID，用于标注
    :param method: "pca" | "tsne"
    """
    X = np.stack(embeddings, axis=0)

    if method.lower() == "pca":
        reducer = PCA(n_components=2)
        X2 = reducer.fit_transform(X)
        title = "Geometry Embedding (PCA)"
    elif method.lower() == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=min(30, len(X) - 1),
            init="pca",
            learning_rate="auto"
        )
        X2 = reducer.fit_transform(X)
        title = "Geometry Embedding (t-SNE)"
    else:
        raise ValueError(f"Unknown method: {method}")

    plt.figure(figsize=figsize)
    plt.scatter(X2[:, 0], X2[:, 1], c="blue")

    if ids is not None:
        for i, fid in enumerate(ids):
            plt.text(
                X2[i, 0],
                X2[i, 1],
                str(fid),
                fontsize=9
            )

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
