"""Independent checks against the definition of two-dimensional SE."""

import unittest

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage

from lfevent.clustering import cluster_features, cosine_knn_graph, structural_entropy_curve


def reference_entropy(labels, edges, weights):
    degrees = np.zeros(len(labels))
    for (a, b), weight in zip(edges, weights):
        degrees[a] += weight
        degrees[b] += weight
    total = degrees.sum()
    result = 0.0
    for label in np.unique(labels):
        members = set(np.flatnonzero(labels == label))
        volume = sum(degrees[i] for i in members)
        if volume == 0:
            continue
        cut = sum(weight for (a, b), weight in zip(edges, weights) if (a in members) != (b in members))
        result -= cut / total * np.log2(volume / total)
        for i in members:
            if degrees[i] > 0:
                result -= degrees[i] / total * np.log2(degrees[i] / volume)
    return result


class EntropyTests(unittest.TestCase):
    def test_all_ward_cuts_match_definition(self):
        # Also checks merges between disconnected components and repeated distances.
        for seed in range(8):
            features = np.random.default_rng(seed).normal(size=(15, 6))
            edges, weights = cosine_knn_graph(features, 3)
            tree = linkage(features, method="ward")
            curve = structural_entropy_curve(tree, edges, weights, len(features))
            for count in range(1, len(features) + 1):
                labels = cut_tree(tree, n_clusters=[count]).ravel()
                self.assertAlmostEqual(curve[count - 1], reference_entropy(labels, edges, weights), places=11)

    def test_isolates_have_zero_contribution(self):
        features = np.array([[1., 0], [1., .1], [-1., 0], [-1., -.1], [0., 0]])
        edges, weights = cosine_knn_graph(features, 1)
        self.assertNotIn(4, edges)
        tree = linkage(features, method="ward")
        curve = structural_entropy_curve(tree, edges, weights, len(features))
        self.assertTrue(np.isfinite(curve).all())
        for count in range(1, len(features) + 1):
            labels = cut_tree(tree, n_clusters=[count]).ravel()
            self.assertAlmostEqual(curve[count - 1], reference_entropy(labels, edges, weights), places=11)

    def test_graph_uses_cosine_and_excludes_self(self):
        features = np.array([[1., 0], [2., 0], [0., 1]])
        edges, weights = cosine_knn_graph(features, 1)
        np.testing.assert_array_equal(edges, [[0, 1]])
        np.testing.assert_allclose(weights, [1.])

    def test_selected_partition_is_global_minimum(self):
        features = np.random.default_rng(97).normal(size=(25, 8))
        labels, curve, tree, _ = cluster_features(features, 3)
        chosen = np.unique(labels).size
        self.assertAlmostEqual(curve[chosen - 1], curve.min(), places=11)
        edges, weights = cosine_knn_graph(features, 3)
        self.assertAlmostEqual(curve[chosen - 1], reference_entropy(labels, edges, weights), places=11)

    def test_zero_volume_graph_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "zero volume"):
            cluster_features(np.zeros((4, 3)))


if __name__ == "__main__":
    unittest.main()
