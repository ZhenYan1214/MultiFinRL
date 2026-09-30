"""回測：累積報酬、Sharpe Ratio、最大回撤（MDD），比較加入 RL 前後的績效。

- 嚴禁使用未來資訊；t 日決定的權重用 t+1 日報酬結算。
- 內建三種策略：
    buy_and_hold   全程滿倉（baseline）
    rule_based     分類器預測 BULLISH 滿倉 / BEARISH 空倉 / NEUTRAL 半倉（無 RL 對照組）
    ppo            用訓練好的 PPO agent 決定權重（RL 組）
- 報告輸出到 data/outputs/backtest/report_{run_id}.json。

用法：
    python -m module_c_fusion.backtest.backtest --ticker AAPL --strategy buy_and_hold
"""
import argparse
import datetime as dt

import numpy as np

from shared import paths
from shared.temporal_split import load_temporal_manifest, split_rows
from shared.utils import load_config, read_json, write_json
from module_c_fusion.fusion.consolidate import load_index

TRADING_DAYS_PER_YEAR = 252


# --- 績效指標 ---

def cumulative_return(nav: np.ndarray) -> float:
    return float(nav[-1] / nav[0] - 1)


def sharpe_ratio(daily_returns: np.ndarray, risk_free: float = 0.0) -> float:
    ex = daily_returns - risk_free / TRADING_DAYS_PER_YEAR
    if ex.std() == 0:
        return 0.0
    return float(ex.mean() / ex.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(nav: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav)
    return float(((peak - nav) / peak).max())


def run_backtest(weights: np.ndarray, asset_returns: np.ndarray, cost: float = 0.001) -> dict:
    """weights[t] 是 t 日收盤後決定的持股權重，用 asset_returns[t]（t->t+1 報酬）結算。"""
    turnover = np.abs(np.diff(weights, prepend=0.0))
    port_returns = weights * asset_returns - turnover * cost
    nav = np.cumprod(1.0 + port_returns)
    return {
        "cumulative_return": cumulative_return(np.concatenate([[1.0], nav])),
        "sharpe_ratio": sharpe_ratio(port_returns),
        "max_drawdown": max_drawdown(np.concatenate([[1.0], nav])),
        "n_days": len(weights),
        "final_nav": float(nav[-1]),
    }


# --- 資料與策略 ---

def load_series(ticker: str):
    """回傳 (dates, z_seq, next_day_returns)。

    優先讀彙整索引；索引不存在時 fallback 成逐日掃描。
    注意：這裡回傳整段期間的序列，不自動切測試期。main() 會用共用
    manifest 只取 Test；rule_based 的模型選擇則只使用 Train/Validation。
    """
    idx = load_index(ticker)
    if idx is not None:
        return idx["dates"].tolist(), idx["z"], idx["return_next"]

    z_dir = paths.OUTPUTS / "z_fused" / ticker
    dates, z_list, r_list = [], [], []
    for f in sorted(z_dir.glob("*.npy")):
        record_file = paths.daily_json(ticker, f.stem)
        if not record_file.exists():
            continue
        prices = read_json(record_file)["prices"]
        dates.append(f.stem)
        z_list.append(np.load(f))
        r_list.append(prices["future_closes"][0] / prices["close_t0"] - 1)
    return dates, np.stack(z_list), np.array(r_list)


def weights_buy_and_hold(n: int, **_) -> np.ndarray:
    return np.ones(n)


def weights_rule_based(z_seq: np.ndarray, ticker: str, weighted: bool = False, **_) -> np.ndarray:
    """沿用 manifest：Train 選模型、Train+Validation refit、只對 Test 出訊號。"""
    from sklearn.linear_model import LogisticRegression
    from module_c_fusion.validation.classifier import arrays, load_z_and_labels, metrics
    rows = load_z_and_labels(ticker)
    groups = split_rows(rows, load_temporal_manifest(ticker), require_all_manifest_dates=True)
    Xtr, ytr = arrays(groups["train"])
    Xval, yval = arrays(groups["validation"])
    class_weight = "balanced" if weighted else None
    best = None
    for c_value in (0.01, 0.1, 1.0, 10.0):
        candidate = LogisticRegression(
            max_iter=1000, class_weight=class_weight, C=c_value
        ).fit(Xtr, ytr)
        result = metrics(yval, candidate.predict(Xval))
        key = (result["macro_f1"], result["accuracy"], -c_value)
        if best is None or key > best[0]:
            best = (key, c_value)
    Xfit, yfit = arrays(groups["train"] + groups["validation"])
    clf = LogisticRegression(
        max_iter=1000, class_weight=class_weight, C=best[1]
    ).fit(Xfit, yfit)
    pred = clf.predict(z_seq)  # 0=BEARISH,1=NEUTRAL,2=BULLISH
    return np.select([pred == 2, pred == 1], [1.0, 0.5], default=0.0)


def weights_ppo(z_seq: np.ndarray, **_) -> np.ndarray:
    from stable_baselines3 import PPO
    agent = PPO.load(paths.OUTPUTS / "checkpoints" / "ppo_agent.zip")
    weights, w_prev = [], 0.0
    for z in z_seq:
        obs = np.concatenate([z, [w_prev]]).astype(np.float32)
        action, _ = agent.predict(obs, deterministic=True)
        w_prev = float(np.clip(action[0], 0, 1))
        weights.append(w_prev)
    return np.array(weights)

STRATEGIES = {
    "buy_and_hold": weights_buy_and_hold,
    "rule_based": weights_rule_based,
    "ppo": weights_ppo,
}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--strategy", choices=STRATEGIES, default="buy_and_hold")
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--weighted", action="store_true",
                    help="rule_based 分類器使用 class_weight=balanced")
    args = ap.parse_args()

    dates, z_seq, returns = load_series(args.ticker)
    if len(dates) < 10:
        raise SystemExit("Z_fused 不足，先跑 module_c_fusion.fusion.train")

    manifest = load_temporal_manifest(args.ticker)
    rows = list(zip(dates, list(z_seq), returns.tolist()))
    groups = split_rows(rows, manifest, require_all_manifest_dates=True)
    test_rows = groups["test"]
    dates = [row[0] for row in test_rows]
    z_seq = np.stack([row[1] for row in test_rows])
    returns = np.asarray([row[2] for row in test_rows])

    fn = STRATEGIES[args.strategy]
    weights = fn(n=len(dates), z_seq=z_seq, ticker=args.ticker, weighted=args.weighted)
    result = run_backtest(np.asarray(weights, dtype=float), returns, args.cost)

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "run_id": run_id, "ticker": args.ticker, "strategy": args.strategy,
        "protocol": manifest["protocol"],
        "period": [dates[0], dates[-1]], "cost": args.cost,
        "split": "test", "split_manifest": str(paths.temporal_split_path(args.ticker)),
        "weighted_classifier": args.weighted if args.strategy == "rule_based" else None,
        **result,
    }
    out = paths.OUTPUTS / "backtest" / f"report_{run_id}.json"
    write_json(report, out)
    print(f"[backtest] {args.strategy}: cum={result['cumulative_return']:.2%} "
          f"sharpe={result['sharpe_ratio']:.2f} mdd={result['max_drawdown']:.2%} -> {out}")


if __name__ == "__main__":
    main()
