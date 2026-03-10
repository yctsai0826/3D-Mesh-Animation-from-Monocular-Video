import torch

def build_knn_graph(p0: torch.Tensor, k: int = 6):
    """
    p0: (M,3) torch float tensor (same device as model)
    returns edge_index: (2,E) long tensor with undirected edges
    """
    M = p0.shape[0]
    # pairwise distance matrix
    d = torch.cdist(p0, p0) + torch.eye(M, device=p0.device) * 1e9
    knn = torch.topk(d, k=k, largest=False).indices  # (M,k)
    src = torch.arange(M, device=p0.device).unsqueeze(1).repeat(1, k)
    edge_index = torch.stack([src.reshape(-1), knn.reshape(-1)], dim=0)
    # make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index.long()