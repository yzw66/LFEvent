"""Preserved graph construction, Ward hierarchy and minimum structural entropy cut."""
# 聚类过程不读取事件真值：先构图，再在原始特征的 Ward 层次树上选择结构熵最小的切分。
from itertools import combinations
import re
import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage

def cosine_knn_graph(features, k=3):
    # 兼容的余弦 kNN 图；当前训练入口实际使用文件末尾的 reference_graph。
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) < 2:
        raise ValueError("At least two feature vectors are required")
    if not np.isfinite(features).all():
        raise ValueError("Non-finite features")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norms, np.finfo(float).tiny)
    similarities = normalized @ normalized.T
    # 将对角线设为负无穷，避免把样本自身选为最近邻。
    np.fill_diagonal(similarities, -np.inf)
    k = min(k, len(features) - 1)
    if k < 1:
        raise ValueError("k must be positive")
    neighbors = np.argpartition(similarities, len(features) - k, axis=1)[:, -k:]
    sources = np.repeat(np.arange(len(features)), k)
    destinations = neighbors.ravel()
    # 将有向近邻关系合并为无向边：(a,b) 和 (b,a) 只保留一次。
    edges = np.unique(np.sort(np.column_stack((sources, destinations)), axis=1), axis=0)
    weights = similarities[edges[:, 0], edges[:, 1]]
    # 结构熵需要非负度和体积，因此沿用当前实现，只保留正余弦边权。
    positive = weights > 0
    return edges[positive], weights[positive]



def structural_entropy_curve(tree, edges, weights, n_samples):
    """Evaluate all N Ward partitions, including singletons, in merge order.

    The result is indexed by (number_of_clusters - 1). Isolates contribute zero.
    Updating community volumes and cuts is algebraically equivalent to Eq. (1).
    No event labels or candidate range are used.
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    weights = np.asarray(weights, dtype=np.float64)
    # degrees[v] 是节点加权度；无向边对两个端点都贡献一次，total 为全图加权度之和。
    degrees = np.bincount(edges.ravel(), weights=np.repeat(weights, 2), minlength=n_samples)
    total = degrees.sum()
    if total <= 0:
        raise ValueError("The positive-weight kNN graph has zero volume")
    # 叶节点编号为 0..N-1，Ward 每次合并生成一个新编号，总计 2N-1 个节点。
    # volumes 为社区内节点度之和；cuts 为该社区通往外部的边权之和。
    volumes = np.zeros(2 * n_samples - 1)
    cuts = np.zeros_like(volumes)
    volumes[:n_samples] = cuts[:n_samples] = degrees
    adjacency = {i: {} for i in range(n_samples)}
    for (a, b), weight in zip(edges, weights):
        adjacency[int(a)][int(b)] = float(weight)
        adjacency[int(b)][int(a)] = float(weight)
    log_total = np.log2(total)

    def contribution(volume, cut):
        # 固定全图节点度后，社区对二维结构熵的可变部分；孤立节点的贡献记为 0。
        if volume <= 0:
            return 0.0
        return ((volume - cut) * np.log2(volume) + cut * log_total) / total

    positive = degrees > 0
    # 初始时每个样本自成一簇；随后按 Ward 合并次序，逐步计算 N、N-1、...、1 簇的熵。
    entropy = -np.sum((degrees[positive] / total) * np.log2(degrees[positive] / total))
    scores = np.empty(n_samples)
    scores[-1] = entropy
    for i, merge in enumerate(tree):
        a, b = map(int, merge[:2])
        merged = n_samples + i
        left, right = adjacency.pop(a), adjacency.pop(b)
        crossing = left.pop(b, 0.0)
        right.pop(a, None)
        volumes[merged] = volumes[a] + volumes[b]
        # 两社区之间的边合并后成为内部边，因此外部割边权要减去两倍 crossing。
        cuts[merged] = max(0.0, cuts[a] + cuts[b] - 2 * crossing)
        # 只更新参与合并的社区贡献，无需对每个候选簇数重新遍历整张图。
        entropy += contribution(volumes[merged], cuts[merged])
        entropy -= contribution(volumes[a], cuts[a]) + contribution(volumes[b], cuts[b])
        if len(left) < len(right):
            left, right = right, left
        for neighbor, weight in right.items():
            left[neighbor] = left.get(neighbor, 0.0) + weight
        for neighbor, weight in left.items():
            adjacency[neighbor].pop(a, None)
            adjacency[neighbor].pop(b, None)
            adjacency[neighbor][merged] = weight
        adjacency[merged] = left
        # 数组索引为“簇数减一”：scores[K-1] 对应 K 个簇时的结构熵。
        scores[n_samples - i - 2] = entropy
    return scores



def cluster_features(features, k=3):
    # 余弦图配合 Ward 的便捷入口；当前 engine.evaluate 使用 reference_graph 后执行相同选簇步骤。
    features = np.asarray(features, dtype=np.float64)
    edges, weights = cosine_knn_graph(features, k)
    tree = linkage(features, method="ward")
    scores = structural_entropy_curve(tree, edges, weights, len(features))
    # 从 N 到 1 搜索全部候选；熵完全相同时按搜索顺序选择较大的簇数。
    best_count = len(features) - int(np.argmin(scores[::-1]))
    labels = cut_tree(tree, n_clusters=[best_count]).ravel()
    return labels, scores, tree, len(edges)



def reference_graph(features, texts, k=1, include_attributes=True, neighbor_metric="pearson"):
    """Reference Pearson neighbors + shared tags; consistent zero-based indexing."""
    pairs = set()
    if neighbor_metric == "cosine":
        knn, _ = cosine_knn_graph(features, k)
        pairs.update(map(tuple, knn.tolist()))
    else:
        # 当前默认：用特征向量之间的 Pearson 相关系数选 1 个近邻，仅保留正相关关系。
        corr = np.corrcoef(features)
        np.fill_diagonal(corr, 0)
        neighbors = np.argsort(corr, axis=1)[:, -k:]
        for a, row in enumerate(neighbors):
            for b in row:
                if a != b and corr[a, b] > 0:
                    pairs.add((min(a, int(b)), max(a, int(b))))
    if include_attributes:
        # 原文中共享相同 hashtag 或 @mention 的样本补连边；转小写后按标签分别连接。
        tags = {}
        for i, text in enumerate(texts):
            for tag in set(re.findall(r"[#@]\w+", str(text).lower())):
                tags.setdefault(tag, []).append(i)
        for members in tags.values():
            pairs.update(combinations(members, 2))
    edges = np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    # Pearson 只决定近邻边是否存在；实际用于结构熵的边权统一取正余弦相似度。
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), np.finfo(float).tiny)
    weights = np.einsum("ij,ij->i", normalized[edges[:, 0]], normalized[edges[:, 1]])
    positive = weights > 0
    return edges[positive], weights[positive]
