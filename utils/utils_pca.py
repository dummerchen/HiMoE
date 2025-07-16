import torch
def PCA(data: torch.Tensor, n_components, random_state=None):
    if random_state is not None:
        torch.manual_seed(random_state)
    mean = torch.mean(data, dim=0)
    centered_data = data - mean
    U, S, V = torch.svd(centered_data)
    return torch.matmul(centered_data, V[:, :n_components])