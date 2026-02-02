import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from tqdm import tqdm

from hypercore.manifolds import Lorentz

def extract_features(model, dataloader, device, is_hyperbolic=False):
    if hasattr(model, "heads"):
        model.heads = nn.Identity()
    elif hasattr(model, "fc"):
        model.fc = nn.Identity()
    elif hasattr(model, "head"):
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        model.classifier = nn.Identity()

    model.to(device)
    model.eval()

    if is_hyperbolic:
        manifold:Lorentz = model.manifold_out
    else:
        manifold = None

    features_list = []
    labels_list = []

    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(dataloader)):
            images = images.to(device)

            embeddings = model(images)
            if is_hyperbolic:
                if idx == 0:
                    check = manifold.check_point_on_manifold(embeddings)
                    print(f"Lorentz manifold check: {check}")
                embeddings = manifold.projx(embeddings)
            embeddings = embeddings.cpu().numpy()

            features_list.append(embeddings)
            labels_list.append(targets.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return features, labels


def plot_umap(
    features,
    labels,
    save_path="umap_vit.png",
    n_neighbors=15,
    min_dist=0.1,
    is_hyperbolic=False,
    manifold=None,
    device=None,
):
    """
    Applies UMAP to features and saves a 2D scatter plot.
    """
    print(f"Fitting UMAP on {features.shape} matrix...")
    device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    orig_dim = features.shape[1]  # keep for title

    is_poincare = False

    if is_poincare and is_hyperbolic:
        raise NotImplementedError("Poincare UMAP not implemented yet.")
    
    elif is_hyperbolic:

        def dist_func(A_np, B_np):
            A = torch.from_numpy(A_np).to(device)
            B = torch.from_numpy(B_np).to(device)
            with torch.no_grad():
                D = manifold.pairwise_distance(A, B, keepdim=False, distance="geodesic")
                # if does not return N, M, raise
                assert D.ndim == 2 and D.shape[0] == A.shape[0] and D.shape[1] == B.shape[0]

            return D.detach().cpu().numpy()

        knn_indices, knn_dists = build_precomputed_knn(
            features, k=n_neighbors, dist_func=dist_func
        )
        X = np.asarray(features)
        if X.ndim != 2:
            raise ValueError(f"Expected features to be 2D (N,D). Got shape {X.shape}")

        N, D = X.shape

        features = (knn_indices, knn_dists)
        reducer = UMAP(
            # output_metric="precomputed",
            n_components=2,
            random_state=42,
            min_dist=min_dist,
            n_neighbors=n_neighbors,
            # metric="precomputed", 
            precomputed_knn=(knn_indices, knn_dists), 
        )
        X_dummy = np.zeros((N, 1), dtype=np.float32)
        proj_2d = reducer.fit_transform(X_dummy)
    else:
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        proj_2d = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        proj_2d[:, 0],
        proj_2d[:, 1],
        c=labels,
        cmap=(
            "tab10" if not is_hyperbolic else "Spectral"
        ),  # Good colormap for categorical data
        s=5,
        alpha=0.7,
    )
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)

    tag = "Hyp" if is_hyperbolic else "Euc"
    plt.title(
        f"UMAP Projection of {tag} ViT Features\n(Input dim: {orig_dim}, Neighbors: {n_neighbors})"
    )
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


import numpy as np

def build_precomputed_knn(features, k, dist_func, block=2048, dtype=np.float32):
    """
    Build (knn_indices, knn_dists) for UMAP from a custom distance function, without storing NxN.

    dist_func(A, B) must return a 2D array of distances with shape (A.shape[0], B.shape[0]).
    """
    X = np.asarray(features, dtype=dtype)
    N = X.shape[0]
    k = int(k)
    assert k >= 1 and k <= N

    knn_indices = np.empty((N, k), dtype=np.int64)
    knn_dists = np.empty((N, k), dtype=dtype)

    all_idx = np.arange(N, dtype=np.int64)

    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        A = X[i0:i1]  # (b, d)
        # maintain top-k for each row in this block
        best_d = np.full((i1 - i0, k), np.inf, dtype=dtype)
        best_j = np.full((i1 - i0, k), -1, dtype=np.int64)

        for j0 in range(0, N, block):
            j1 = min(N, j0 + block)
            B = X[j0:j1]  # (c, d)

            D = dist_func(A, B).astype(dtype, copy=False)  # (b, c)

            # Merge candidates: concatenate current best with this chunk
            cand_d = np.concatenate([best_d, D], axis=1)  # (b, k+c)
            cand_j = np.concatenate(
                [best_j, all_idx[j0:j1][None, :].repeat(i1 - i0, axis=0)], axis=1
            )

            # Take k smallest per row using argpartition, then sort those k
            part = np.argpartition(cand_d, kth=k - 1, axis=1)[:, :k]
            new_d = np.take_along_axis(cand_d, part, axis=1)
            new_j = np.take_along_axis(cand_j, part, axis=1)

            order = np.argsort(new_d, axis=1)
            best_d = np.take_along_axis(new_d, order, axis=1)
            best_j = np.take_along_axis(new_j, order, axis=1)

        # Ensure self is included as first neighbor with dist 0
        rows = np.arange(i0, i1)
        self_pos = best_j == rows[:, None]
        has_self = self_pos.any(axis=1)

        # If self missing, force replace last neighbor with self
        missing = ~has_self
        if np.any(missing):
            best_j[missing, -1] = rows[missing]
            best_d[missing, -1] = 0.0
            # re-sort
            order = np.argsort(best_d, axis=1)
            best_d = np.take_along_axis(best_d, order, axis=1)
            best_j = np.take_along_axis(best_j, order, axis=1)

        # If self present but not first, swap into first position
        for r in range(i1 - i0):
            pos = np.where(best_j[r] == (i0 + r))[0]
            if pos.size and pos[0] != 0:
                p = pos[0]
                best_j[r, 0], best_j[r, p] = best_j[r, p], best_j[r, 0]
                best_d[r, 0], best_d[r, p] = best_d[r, p], best_d[r, 0]
                best_d[r, 0] = 0.0

        knn_indices[i0:i1] = best_j
        knn_dists[i0:i1] = best_d

    return knn_indices, knn_dists
