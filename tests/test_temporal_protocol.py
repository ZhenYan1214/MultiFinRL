import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from module_a_data.labeling import compute_quantile_thresholds
from module_c_fusion.fusion.train import train_with_validation
from module_c_fusion.validation.event_validation_head import evaluate_category
from shared.temporal_split import build_temporal_manifest, split_name_by_date, split_rows


class TemporalProtocolTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2024-01-01", periods=100).strftime("%Y-%m-%d").tolist()
        self.manifest = build_temporal_manifest(
            "TEST", self.dates, train_ratio=0.7, validation_ratio=0.15,
            gap_trading_days=5,
        )

    def test_split_is_disjoint_and_keeps_two_five_day_gaps(self):
        groups = self.manifest["dates"]
        self.assertEqual(len(groups["train"]), 70)
        self.assertEqual(len(groups["gap_train_validation"]), 5)
        self.assertEqual(len(groups["validation"]), 10)
        self.assertEqual(len(groups["gap_validation_test"]), 5)
        self.assertEqual(len(groups["test"]), 10)
        flattened = [date for values in groups.values() for date in values]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(sorted(flattened), self.dates)

    def test_split_rows_rejects_missing_upstream_dates(self):
        rows = [(date, index) for index, date in enumerate(self.dates[:-1])]
        with self.assertRaisesRegex(ValueError, "資料缺少 manifest"):
            split_rows(rows, self.manifest)

    def test_split_rows_rejects_duplicate_dates(self):
        rows = [(date, index) for index, date in enumerate(self.dates)]
        rows.append((self.dates[0], 999))
        with self.assertRaisesRegex(ValueError, "重複日期"):
            split_rows(rows, self.manifest)

    def test_split_lookup_marks_gap_dates_explicitly(self):
        lookup = split_name_by_date(self.manifest)
        self.assertEqual(lookup[self.dates[70]], "gap_train_validation")
        self.assertEqual(lookup[self.dates[85]], "gap_validation_test")
        self.assertEqual(lookup[self.dates[-1]], "test")

    def test_quantile_thresholds_use_only_requested_train_dates(self):
        index = pd.bdate_range("2024-01-01", periods=12)
        closes = np.asarray([100, 101, 102, 103, 104, 105, 106, 130, 160, 200, 250, 320])
        frame = pd.DataFrame({"Close": closes}, index=index)
        train_dates = index[:6].strftime("%Y-%m-%d").tolist()
        train_returns = frame["Close"].shift(-1).div(frame["Close"]).sub(1).loc[index[:6]]
        expected = np.quantile(train_returns.to_numpy(), [1 / 3, 2 / 3])

        bearish, bullish = compute_quantile_thresholds(
            frame, horizon=1, low_q=1 / 3, high_q=2 / 3, dates=train_dates,
        )

        self.assertAlmostEqual(bearish, expected[0])
        self.assertAlmostEqual(bullish, expected[1])
        full_bearish, full_bullish = compute_quantile_thresholds(frame, horizon=1)
        self.assertNotAlmostEqual(bullish, full_bullish)
        self.assertNotAlmostEqual(bearish, full_bearish)

    def test_validation_labels_do_not_update_fusion_parameters(self):
        class TinyFusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(1, 2)
                self.head = nn.Linear(2, 2)

            def forward(self, h_v, _h_t, _h_r):
                return self.head(torch.tanh(self.projection(h_v)))

        h_v = np.asarray([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        zeros = np.zeros_like(h_v)
        train_batch = [(h_v, zeros, zeros, np.asarray([0, 0, 1, 1, 2, 2]))]
        validation_a = [(h_v, zeros, zeros, np.asarray([0, 0, 1, 1, 2, 2]))]
        validation_b = [(h_v, zeros, zeros, np.asarray([2, 2, 1, 1, 0, 0]))]

        torch.manual_seed(7)
        model_a = TinyFusion()
        train_with_validation(model_a, train_batch, validation_a, 1, 1e-2, "cpu")
        state_a = {name: value.detach().clone() for name, value in model_a.state_dict().items()}

        torch.manual_seed(7)
        model_b = TinyFusion()
        train_with_validation(model_b, train_batch, validation_b, 1, 1e-2, "cpu")

        for name, expected in state_a.items():
            self.assertTrue(torch.equal(expected, model_b.state_dict()[name]), name)

    def test_event_probe_uses_validation_for_selection_and_test_for_report(self):
        def rows(values, labels, prefix):
            return [
                (f"{prefix}-{index}", np.asarray([value], dtype=np.float32), [label])
                for index, (value, label) in enumerate(zip(values, labels))
            ]

        train = rows(range(10), [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], "train")
        validation = rows([0, 1, 8, 9], [0, 0, 1, 1], "validation")
        test = rows([0, 2, 7, 9], [0, 0, 1, 1], "test")

        result = evaluate_category(train, validation, test, 0, [0.1, 1.0])

        self.assertTrue(result["evaluated"])
        self.assertIn(result["selected_C"], [0.1, 1.0])
        self.assertEqual(result["positive_days"], {"train": 5, "validation": 2, "test": 2})
        self.assertIn("validation_candidates", result)


if __name__ == "__main__":
    unittest.main()
