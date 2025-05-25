import torch
import numpy as np


def sample_triplets_hard_negative(features, labels):
    """
    features: Tensor shape [B, d]
    labels:   Tensor shape [B]

    return: (anchor, positive, negative) => shape [B, d] 
    """
    device = features.device
    B = features.size(0)
    
    anchor_idx = []
    pos_idx = []
    neg_idx = []
    
    # dist_matrix[i, j] = ||features[i] - features[j]|| (L2)
    with torch.no_grad():
        # (B, d) => (B,1,d) => broadcast => (B,B,d)
        diff = features.unsqueeze(1) - features.unsqueeze(0)  # [B,B,d]
        dist_matrix = diff.pow(2).sum(dim=2).sqrt()           # [B,B]
    
    for i in range(B):
        anchor_label = labels[i].item()
        
        same_label_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
        diff_label_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]
        
        if len(same_label_indices) > 1 and len(diff_label_indices) > 0:
            # 1) Positive (anchor!=pos)
            pos_candidates = same_label_indices[same_label_indices != i]
            pos_i = pos_candidates[torch.randint(len(pos_candidates), size=())]
            
            # 2) Negative(Hard) 
            neg_dists = dist_matrix[i, diff_label_indices]  # shape [#diff]
            min_dist_idx = torch.argmin(neg_dists)          # scalar
            neg_i = diff_label_indices[min_dist_idx]
            
            anchor_idx.append(i)
            pos_idx.append(pos_i.item())
            neg_idx.append(neg_i.item())
        else:
            anchor_idx.append(i)
            pos_idx.append(i)
            neg_idx.append(i)
    
    anchor_idx = torch.tensor(anchor_idx, dtype=torch.long, device=device)
    pos_idx    = torch.tensor(pos_idx,    dtype=torch.long, device=device)
    neg_idx    = torch.tensor(neg_idx,    dtype=torch.long, device=device)
    
    anchors   = features[anchor_idx]
    positives = features[pos_idx]
    negatives = features[neg_idx]
    
    return anchors, positives, negatives