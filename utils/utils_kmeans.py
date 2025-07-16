import torch
import numpy as np
from sklearn.cluster import KMeans as sklearn_KMeans

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None, device='cuda:0'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.device = device

    def fit(self, X:torch.Tensor):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
            X = X.to(self.device) if torch.cuda.is_available() else X

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Initialize centroids
        idx = torch.randperm(X.shape[0])[:self.n_clusters]
        self.cluster_centers_ = X[idx]

        for _ in range(self.max_iter):
            # Assign each point to the nearest centroid
            distances = torch.cdist(X, self.cluster_centers_)
            labels = torch.argmin(distances, dim=1)

            # Update centroids
            new_centroids = torch.stack([X[labels == k].mean(dim=0) for k in range(self.n_clusters)])

            # Check for convergence
            if torch.sum(torch.abs(new_centroids - self.cluster_centers_)) < self.tol:
                break

            self.cluster_centers_ = new_centroids

        self.labels_ = labels
        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.cuda() if torch.cuda.is_available() else X

        distances = torch.cdist(X, self.centroids)
        return torch.argmin(distances, dim=1)

def fit_kmeans(features, clusters, length, random_state=3407):
    kmeans = KMeans(n_clusters=clusters, random_state=random_state)
    kmeans.fit(features)
    prompt_kmeans_inits = []
    if length > 1:
        for i in range(clusters):
            prompt_features = features[kmeans.labels_ == i]
            if len(prompt_features) > length:
                pool_kmeans = KMeans(n_clusters=length, random_state=random_state)
                pool_kmeans.fit(prompt_features)

                prompt_kmeans = pool_kmeans.cluster_centers_
                prompt_kmeans_inits.append(prompt_kmeans)
            else:
                prompt_kmeans = torch.tile(kmeans.cluster_centers_[i], (length, 1))
                prompt_kmeans_inits.append(prompt_kmeans)
        out = torch.stack(prompt_kmeans_inits)
    else:
        if clusters == 1:
            out = kmeans.cluster_centers_
        else:
            out = kmeans.cluster_centers_.squeeze()
    return out
# 示例用法
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # 生成数据
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # 使用sklearn的KMeans
    sklearn_kmeans = sklearn_KMeans(n_clusters=4)
    sklearn_labels = sklearn_kmeans.fit_predict(X)
    sklearn_centers = sklearn_kmeans.cluster_centers_

    # 使用自定义的KMeans
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    custom_labels = kmeans.labels_.cpu().numpy()
    custom_centers = kmeans.centroids.cpu().numpy()

    # 可视化结果
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=sklearn_labels, cmap='viridis')
    plt.scatter(sklearn_centers[:, 0], sklearn_centers[:, 1], s=200, c='red', marker='X')
    plt.title("Sklearn KMeans")


    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=custom_labels, cmap='viridis')
    plt.scatter(custom_centers[:, 0], custom_centers[:, 1], s=200, c='red', marker='X')
    plt.title("Custom KMeans")

    plt.show()
