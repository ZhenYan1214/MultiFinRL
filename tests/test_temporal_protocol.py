import unittest

import numpy as np
import pandas as pd

from module_a_data.labeling import compute_quantile_thresholds
from module_c_fusion.rl.train_ppo import curriculum_stages
from shared.temporal_split import assign_temporal_splits, partition_by_split


class TemporalProtocolTest(unittest.TestCase):
    def test_future_label_crossing_boundary_is_purged(self):
        dates = pd.bdate_range("2024-01-01", periods=20).strftime("%Y-%m-%d").tolist()
        targets = [dates[min(i + 2, len(dates) - 1)] for i in range(len(dates))]

        splits = assign_temporal_splits(dates, targets, train_ratio=0.60, validation_ratio=0.20)

        self.assertEqual(splits[9:12], ["train", "purged", "purged"])
        self.assertEqual(splits[12:16], ["validation", "validation", "purged", "purged"])
        self.assertTrue(all(split == "test" for split in splits[16:]))

    def test_partition_uses_persisted_split_instead_of_recutting_intersection(self):
        rows = [
            ("2024-01-01", "a", "train"),
            ("2024-01-08", "b", "purged"),
            ("2024-02-01", "c", "validation"),
            ("2024-03-01", "d", "test"),
        ]
        train, validation, test = partition_by_split(rows)
        self.assertEqual([row[0] for row in train], ["2024-01-01"])
        self.assertEqual([row[0] for row in validation], ["2024-02-01"])
        self.assertEqual([row[0] for row in test], ["2024-03-01"])

    def test_quantile_thresholds_fit_only_requested_dates(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        # 後半段刻意放入巨大跳升；若誤用全期間，門檻會明顯改變。
        close = [100, 101, 102, 103, 104, 105, 200, 400, 800, 1600]
        df = pd.DataFrame({"Close": close}, index=dates)
        fit_dates = dates[:5].strftime("%Y-%m-%d").tolist()

        low, high = compute_quantile_thresholds(
            df, horizon=1, low_q=1 / 3, high_q=2 / 3, fit_dates=fit_dates
        )
        expected = (df["Close"].shift(-1) / df["Close"] - 1).loc[dates[:5]]
        self.assertAlmostEqual(low, float(expected.quantile(1 / 3)))
        self.assertAlmostEqual(high, float(expected.quantile(2 / 3)))

    def test_curriculum_stages_never_stitch_non_contiguous_days(self):
        # 第一欄保留原始時間索引，方便驗證每個 stage 是否真的是連續切片。
        z = np.arange(60, dtype=np.float32).reshape(-1, 1)
        returns = np.array(([0.001] * 10 + [0.08] * 10) * 3, dtype=np.float32)

        stages = curriculum_stages(z, returns, n_stages=3, min_stage_days=10)

        for z_stage, _ in stages:
            self.assertTrue(np.all(np.diff(z_stage[:, 0]) == 1))
        self.assertTrue(np.array_equal(stages[-1][0], z))


if __name__ == "__main__":
    unittest.main()
