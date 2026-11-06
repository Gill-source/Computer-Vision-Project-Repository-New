"""Small, independent correctness checks for the hand-written KNN pipeline."""
import unittest

import numpy as np

from python import (KNNClassifier, classification_scores, make_features,
                    stratified_folds, stratified_sample)


class KNNTests(unittest.TestCase):
    def test_distances_match_definitions(self):
        train = np.array([[1, 2, 4], [3, 1, 0], [0, 0, 0]], dtype=np.float32)
        query = np.array([[2, 3, 1], [0, 0, 0]], dtype=np.float32)
        labels = np.array([0, 1, 2])
        for metric in ("euclidean", "manhattan", "cosine"):
            actual = KNNClassifier(metric, train_batch_size=2).fit(train, labels)._distances(query)
            expected = np.zeros_like(actual)
            for i, a in enumerate(query):
                for j, b in enumerate(train):
                    if metric == "euclidean":
                        expected[i, j] = sum((float(u) - float(v)) ** 2 for u, v in zip(a, b))
                    elif metric == "manhattan":
                        expected[i, j] = sum(abs(float(u) - float(v)) for u, v in zip(a, b))
                    else:
                        denominator = np.linalg.norm(a) * np.linalg.norm(b)
                        expected[i, j] = (1 - np.dot(a, b) / denominator if denominator
                                          else (0 if not np.any(a) and not np.any(b) else 1))
            np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_voting_ties_and_batch_consistency(self):
        x = np.array([[0], [2], [4], [6]], dtype=np.float32)
        y = np.array([1, 0, 0, 1])
        query = np.array([[1], [3], [5]], dtype=np.float32)
        first = KNNClassifier(query_batch_size=1).fit(x, y).predict_many_k(query, [1, 2, 3])
        second = KNNClassifier(query_batch_size=8).fit(x, y).predict_many_k(query, [1, 2, 3])
        np.testing.assert_array_equal(first[1], [1, 0, 0])
        np.testing.assert_array_equal(first[2], [0, 0, 0])
        np.testing.assert_array_equal(first[3], [0, 0, 0])
        for k in first:
            np.testing.assert_array_equal(first[k], second[k])

    def test_metrics_from_hand_counted_confusion(self):
        scores, matrix, per_class = classification_scores(
            np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), n_classes=2)
        np.testing.assert_array_equal(matrix, [[1, 1], [0, 2]])
        self.assertAlmostEqual(scores["accuracy"], 0.75)
        self.assertAlmostEqual(scores["precision"], (1 + 2 / 3) / 2)
        self.assertAlmostEqual(scores["recall"], 0.75)
        self.assertAlmostEqual(scores["f1"], (2 / 3 + 0.8) / 2)
        self.assertEqual([row["support"] for row in per_class], [2, 2])
        zero_scores, _, _ = classification_scores(np.array([0]), np.array([0]), n_classes=2)
        self.assertAlmostEqual(zero_scores["precision"], 0.5)

    def test_stratified_sampling_and_disjoint_folds(self):
        labels = np.repeat(np.arange(10), 23)
        chosen = stratified_sample(labels, 107, seed=42)
        self.assertEqual(len(chosen), 107)
        self.assertEqual(len(np.unique(chosen)), 107)
        np.testing.assert_array_equal(chosen, stratified_sample(labels, 107, seed=42))
        counts = np.bincount(labels[chosen])
        self.assertLessEqual(int(counts.max() - counts.min()), 1)
        folds = stratified_folds(labels[chosen])
        validation_indices = []
        for train, validation in folds:
            self.assertEqual(len(np.intersect1d(train, validation)), 0)
            self.assertEqual(len(train) + len(validation), 107)
            self.assertEqual(len(np.unique(labels[chosen][validation])), 10)
            validation_indices.extend(validation)
        np.testing.assert_array_equal(np.sort(validation_indices), np.arange(107))

    def test_pooling_channel_order_and_original_pixels(self):
        pixels = np.concatenate([np.zeros(1024), np.full(1024, 128), np.full(1024, 255)]).astype(np.uint8)[None, :]
        result = make_features(pixels, image_size=8)
        self.assertEqual(result.shape, (1, 192))
        np.testing.assert_allclose(result[0, :64], 0)
        np.testing.assert_allclose(result[0, 64:128], 128 / 255)
        np.testing.assert_allclose(result[0, 128:], 1)
        np.testing.assert_allclose(make_features(pixels, 32), pixels / 255, atol=1e-7)

    def test_k_cannot_exceed_training_size(self):
        model = KNNClassifier().fit(np.array([[0], [1]]), np.array([0, 1]))
        with self.assertRaises(ValueError):
            model.predict_many_k(np.array([[0]]), [3])


if __name__ == "__main__":
    unittest.main()
