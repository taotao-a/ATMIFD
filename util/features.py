"""Feature transformations shared by cache generation and runtime normalization."""

import numpy as np


def trace_node_features(trace):
    """Build six propagation descriptors from a (time,node,node,channel) tensor."""
    edge = np.asarray(trace, dtype=np.float32).sum(axis=-1)
    eps = 1e-8
    node_count = edge.shape[1]
    out_sum, in_sum = edge.sum(axis=2), edge.sum(axis=1)
    out_degree = (edge > 0).sum(axis=2).astype(np.float32)
    in_degree = (edge > 0).sum(axis=1).astype(np.float32)
    out_ratio = edge.max(axis=2) / (out_sum + eps)
    in_ratio = edge.max(axis=1) / (in_sum + eps)
    out_sum, in_sum = np.log1p(out_sum), np.log1p(in_sum)
    if node_count > 1:
        out_degree /= node_count - 1.0
        in_degree /= node_count - 1.0
    features = np.stack([out_sum, in_sum, out_degree, in_degree,
                         out_ratio, in_ratio], axis=-1).astype(np.float32)
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True)
    return (features - mean) / (std + 1e-6)
