import os
import numpy as np
import torch

from openTSNE.affinity import PerplexityBasedNN
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, confusion_matrix


def seed_everything(seed=0):
    """
    Set random seeds for reproducibility.
    """
    print(f"Global seed set to {seed}")

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


def load_embedding(dataset="CIFAR-10", data_root="./data_HFCLIP", is_shuffle=True, is_torch=False, seed=None):
    """
    Load pre-extracted image embeddings and labels.

    Expected files:
        {data_root}/{dataset}_image_embedding_train.npy
        {data_root}/{dataset}_labels_train.txt

    Optional files:
        {data_root}/{dataset}_image_embedding_test.npy
        {data_root}/{dataset}_labels_test.txt
    """
    print(f"\nLoading dataset: {dataset}")

    if seed is not None:
        np.random.seed(seed)

    train_feature_path = os.path.join(data_root, f"{dataset}_image_embedding_train.npy")
    train_label_path = os.path.join(data_root, f"{dataset}_labels_train.txt")

    x = np.load(train_feature_path)
    y = np.loadtxt(train_label_path).astype(np.int32)

    test_feature_path = os.path.join(data_root, f"{dataset}_image_embedding_test.npy")
    test_label_path = os.path.join(data_root, f"{dataset}_labels_test.txt")

    # If test embeddings are available, merge train and test splits.
    if os.path.exists(test_feature_path) and os.path.exists(test_label_path):
        x_test = np.load(test_feature_path)
        y_test = np.loadtxt(test_label_path).astype(np.int32)
        x = np.concatenate([x, x_test], axis=0)
        y = np.concatenate([y, y_test], axis=0)

    # Normalize features onto the unit sphere.
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    if is_shuffle:
        perm = np.random.permutation(len(x))
        x, y = x[perm], y[perm]

    if is_torch:
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()

    return x, y


def cluster_metric(label, pred):
    """
    Compute clustering ACC, NMI, and ARI.

    ACC is computed by the Hungarian matching between predicted clusters and ground-truth labels.
    """
    if torch.is_tensor(label):
        label = label.cpu().numpy()

    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()

    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)

    cm = confusion_matrix(label, pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    acc = cm[row_ind, col_ind].sum() / cm.sum()

    return acc, nmi, ari


def knn(X, perplexity):
    """
    Build the affinity matrix from input features using openTSNE's perplexity-based neighbors.
    """
    print("Computing affinity matrix ...")
    # Build a sparse affinity matrix using perplexity-based nearest neighbors.
    affinity = PerplexityBasedNN(X.astype(np.float64), perplexity, method="exact", n_jobs=-1, verbose=True).P.tocsr()
    # P is a probability matrix, so we multiply it by the number of samples
    # to convert it to the same affinity scale used in spectral clustering.
    affinity = affinity * X.shape[0]

    return affinity


def clustering_score(pred, affinity_matrix):
    """
    Compute the graph clustering score used to select the best assignment-construction trial.

    For each cluster, this score measures the ratio between its within-cluster affinity
    and its total affinity.
    """
    score = 0.0

    for k in np.unique(pred):
        idx = pred == k
        denominator = affinity_matrix[idx].sum()

        if denominator == 0:
            continue

        score += affinity_matrix[idx][:, idx].sum() / denominator

    return score


def sinkhorn(K, reg=0.2, n_iter=4):
    # Convert scores into positive values with entropic regularization.
    # A smaller reg makes the assignment sharper; a larger reg makes it smoother.
    P = np.exp(K / reg)

    # Alternately normalize rows and columns.
    # Row normalization makes each sample have comparable total assignment mass.
    # Column normalization encourages balanced cluster assignment mass.
    for _ in range(n_iter):
        P /= P.sum(axis=1, keepdims=True)
        P /= P.sum(axis=0, keepdims=True)

    # Rescale so that each row has approximately unit mass after column normalization.
    return P * len(P)
