import os, random, torch, numpy as np

# ---- Repro ----
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---- Padding collate for variable-length sequences ----
# Pads to the max N in batch and returns a seq_mask (B,N,1,1) marking valid timesteps.

def pad_collate(batch):
    B = len(batch)
    maxN = max(item['inp'].shape[0] for item in batch)
    M = batch[0]['inp'].shape[1]

    inp = torch.zeros(B, maxN, M, 3, dtype=batch[0]['inp'].dtype)
    tgt = torch.zeros(B, maxN, M, 3, dtype=batch[0]['tgt'].dtype)
    target_x = torch.zeros(B, maxN, M, 1, dtype=batch[0]['target_x'].dtype)
    seq_mask = torch.zeros(B, maxN, 1, 1, dtype=torch.bool)

    meta_list = []
    for i, s in enumerate(batch):
        n = s['inp'].shape[0]
        inp[i, :n] = s['inp']
        tgt[i, :n] = s['tgt']
        target_x[i, :n] = s['target_x']
        seq_mask[i, :n] = True
        meta_list.append(s.get('meta', {}))

    return {
        'inp': inp,
        'tgt': tgt,
        'target_x': target_x,
        'seq_mask': seq_mask,
        'meta_list': meta_list,
        'num_nodes': batch[0]['num_nodes']
    }

# ---- Masking for pretrain ----
# Only sample masked positions from valid timesteps defined by seq_mask.

def make_masks(X, seq_mask=None, ratio=0.5, mode='mixed', time_block=0):
    """
    X: (B,N,M,3) tensor; returns mask boolean (B,N,M,3).
    seq_mask: (B,N,1,1) bool; if provided, we only mask where seq_mask==True.
    """
    B, N, M, C = X.shape
    mask = torch.zeros(B, N, M, C, dtype=torch.bool, device=X.device)
    for b in range(B):
        if seq_mask is None:
            valid_flat = torch.arange(N*M, device=X.device)
        else:
            valid_t = torch.nonzero(seq_mask[b, :, 0, 0], as_tuple=False).squeeze(1)
            if valid_t.numel() == 0:
                continue
            t_grid = valid_t.repeat_interleave(M)
            m_grid = torch.arange(M, device=X.device).repeat(valid_t.numel())
            valid_flat = t_grid * M + m_grid
        num = int(valid_flat.numel() * ratio)
        sel = valid_flat[torch.randperm(valid_flat.numel(), device=X.device)[:max(1, num)]]
        tn = torch.stack([sel // M, sel % M], dim=-1)
        for t, m in tn:
            if mode == 'coord':
                cc = torch.randint(0, C, (1,), device=X.device).item()
                mask[b, t, m, cc] = True
            else:
                mask[b, t, m, :] = True
    return mask


def apply_masks(X, mask, fill='zero'):
    X_masked = X.clone()
    if fill == 'zero':
        X_masked[mask] = 0.0
    elif fill == 'noise':
        X_masked[mask] = torch.randn_like(X_masked[mask]) * 0.01
    return X_masked

# ---- Kinematics helpers ----
def diff1(z):
    return z[:, 1:, ...] - z[:, :-1, ...]

def diff2(z):
    return diff1(diff1(z))

# ---- Checkpoint ----
def save_ckpt(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)

def load_ckpt(path, map_location=None):
    return torch.load(path, map_location=map_location)