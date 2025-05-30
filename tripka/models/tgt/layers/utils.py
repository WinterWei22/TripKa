import torch

def get_probMat(edge_matrix, num_nodes,default_value=0.15):
    prob_matrix = torch.zeros(edge_matrix.shape)
    for i in range(edge_matrix.shape[0]):
        submatrix = edge_matrix[i, :num_nodes[i], :num_nodes[i]].clone()
        # 0->default value, others->
        temp_matrix = torch.where(submatrix == 0, 
                                  torch.tensor(default_value), 
                                  torch.min(submatrix * 0.2, torch.tensor(1.0)))
        
        prob_matrix[i, :num_nodes[i], :num_nodes[i]] = temp_matrix

    return prob_matrix

def trimat_to_mask(triplet_matrix, num_nodes):
    """
    triplet_matrix: [b,i,j,k]
    num_nodes: [b]
    return: prob_matrix
    """
    edge_matrix = triplet_matrix.sum(-1)
    prob_matrix = get_probMat(edge_matrix, num_nodes)
    
    return prob_matrix

def condense(edges_mat):
    """
    edges_mat: (N, i, j, dim)
    x: (N, i * j, dim)
    """
    N, i, j, dim = edges_mat.shape
    x = edges_mat.view(N, i * j, dim)

    return x, (N, i, j, dim)

def uncondense(x, mat_shape):
    """
    x: (N, i * j, dim)
    edges_mat: (N, i, j, dim)
    """
    N, i, j, dim = mat_shape
    edges_mat = x.view(N, i, j, dim)

    return edges_mat

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_remove = ids_shuffle[:, len_keep:]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    x_removed = torch.gather(x, dim=1, index=ids_remove.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore, x_removed

def prob_masking(x, prob_mat, mask_ratio):
    """
        x: [N, i*j, dim]
        prob_mat: [N, i, j]
    """
    N, L, D = x.shape  # batch, length, dim

    N, i, j = prob_mat.shape
    prob = prob_mat.view(N, i * j)
    len_keep = min(int(L * (1 - mask_ratio)), torch.count_nonzero(prob, dim=1).min().item())

    ids_shuffle = torch.argsort(prob, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_remove = ids_shuffle[:, len_keep:]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    x_removed = torch.gather(x, dim=1, index=ids_remove.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore, x_removed

def recover_masking(x_masked, x_removed, ids_restore):
    x_ = torch.cat([x_masked, x_removed], dim=1)
    x_recovered = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[-1]))

    return x_recovered