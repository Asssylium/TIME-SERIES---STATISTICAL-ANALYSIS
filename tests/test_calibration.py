"""Analytic calibration, constant fallback, and input-contract regression tests."""

import unittest

import numpy as np

from src.calibration import ProbabilityShrinkageCalibrator


class ProbabilityShrinkageTests(unittest.TestCase):
    def test_analytic_brier_fit_and_rank_preservation(self):
        probabilities = np.array([[0.0, 0.5, 1.0]])
        labels = np.array([[1, 0, 1]])
        model = ProbabilityShrinkageCalibrator(anchor=0.25).fit(probabilities, labels)
        # numerator=5/16, denominator=11/16, so the optimal weight is 5/11.
        self.assertAlmostEqual(model.weight_, 5 / 11)
        adjusted = model.transform(probabilities)
        np.testing.assert_allclose(adjusted, [[3 / 22, 4 / 11, 13 / 22]])
        np.testing.assert_array_equal(np.argsort(adjusted), np.argsort(probabilities))
        self.assertEqual(adjusted.shape, probabilities.shape)
        self.assertLess(np.mean((adjusted - labels) ** 2), np.mean((probabilities - labels) ** 2))

    def test_negative_association_becomes_constant_not_reversed(self):
        model = ProbabilityShrinkageCalibrator(0.5).fit([0.1, 0.9], [1, 0])
        self.assertEqual(model.weight_, 0.0)
        np.testing.assert_array_equal(model.transform([0, 0.5, 1]), [0.5, 0.5, 0.5])

    def test_weight_cannot_amplify_confidence_or_leave_bounds(self):
        model = ProbabilityShrinkageCalibrator(0.5).fit([0.2, 0.8], [0, 1])
        self.assertEqual(model.weight_, 1.0)
        np.testing.assert_array_equal(model.transform([0, 0.2, 0.8, 1]), [0, 0.2, 0.8, 1])
        for anchor in (0.0, 1.0):
            boundary = ProbabilityShrinkageCalibrator(anchor).fit([0, 1], [0, 1])
            transformed = boundary.transform([0, 0.2, 0.8, 1])
            self.assertTrue(((transformed >= 0) & (transformed <= 1)).all())

    def test_constant_predictions_at_anchor_have_defined_fallback(self):
        model = ProbabilityShrinkageCalibrator(0.25).fit([0.25, 0.25], [0, 1])
        self.assertEqual(model.weight_, 0.0)
        np.testing.assert_array_equal(model.transform([0.1, 0.9]), [0.25, 0.25])

    def test_fit_and_transform_do_not_mutate_inputs(self):
        probabilities = np.array([0.0, 0.5, 1.0])
        labels = np.array([1, 0, 1])
        original_p, original_y = probabilities.copy(), labels.copy()
        model = ProbabilityShrinkageCalibrator(0.25)
        self.assertIs(model.fit(probabilities, labels), model)
        transformed = model.transform(probabilities)
        np.testing.assert_array_equal(probabilities, original_p)
        np.testing.assert_array_equal(labels, original_y)
        self.assertFalse(np.shares_memory(transformed, probabilities))
        previous_weight = model.weight_
        model.transform([0.8, 0.9])
        self.assertEqual(model.weight_, previous_weight)

    def test_invalid_anchors_and_unfitted_transform(self):
        for anchor in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(anchor=anchor), self.assertRaises(ValueError):
                ProbabilityShrinkageCalibrator(anchor)
        with self.assertRaises(RuntimeError):
            ProbabilityShrinkageCalibrator(0.25).transform([0.2])

    def test_invalid_shapes_values_and_labels(self):
        model = ProbabilityShrinkageCalibrator(0.25)
        for probabilities, labels in (
            ([], []),
            (0.2, 1),
            ([0.1, 0.2], [[0, 1]]),
            ([float("nan")], [0]),
            ([float("inf")], [1]),
            ([-0.1], [1]),
            ([1.1], [0]),
            ([0.1], [0.5]),
            ([0.1], [float("nan")]),
            (["0.2"], [0]),
            ([0.1 + 0j], [0]),
        ):
            with self.subTest(probabilities=probabilities, labels=labels):
                with self.assertRaises(ValueError):
                    model.fit(probabilities, labels)
        model.fit([0.1, 0.9], [0, 1])
        for values in ([], 0.5, [float("inf")], [-0.1], [1.1], ["0.5"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                model.transform(values)


if __name__ == "__main__":
    unittest.main()
