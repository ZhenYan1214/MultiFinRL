"""用 PPO（stable-baselines3）訓練投資組合配置 agent。

第一年是否需完整 PPO 或 rule-based 回測即可，尚未定案
（docs/decisions.md 待確認事項）；環境與訓練入口已架好，隨時可跑。

用法：
    python -m module_c_fusion.rl.train_ppo --fake                # 假 Z_fused 測通
    python -m module_c_fusion.rl.train_ppo --ticker AAPL         # 真實 Z_fused
"""
import argparse

import numpy as np

from shared import paths
from shared.utils import load_config, read_json
from module_c_fusion.rl.env import PortfolioEnv


def load_real(ticker: str):
    """讀 Z_fused 序列與對應次日報酬（用 A 的 future_closes[0] 對 close_t0）。"""
    z_dir = paths.OUTPUTS / "z_fused" / ticker
    z_list, r_list = [], []
    for f in sorted(z_dir.glob("*.npy")):
        record_file = paths.daily_json(ticker, f.stem)
        if not record_file.exists():
            continue
        prices = read_json(record_file)["prices"]
        z_list.append(np.load(f))
        r_list.append(prices["future_closes"][0] / prices["close_t0"] - 1)
    return np.stack(z_list), np.array(r_list)


def make_fake(n: int = 200, z_dim: int = 768, seed: int = 42):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, z_dim)).astype(np.float32),
            rng.normal(0.0005, 0.02, n).astype(np.float32))


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--timesteps", type=int, default=10_000)
    args = ap.parse_args()

    if args.fake:
        z_seq, returns = make_fake(z_dim=cfg["fusion"]["z_dim"], seed=cfg["seed"])
    else:
        z_seq, returns = load_real(args.ticker)
        if len(z_seq) < 30:
            raise SystemExit("Z_fused 不足，先跑 module_c_fusion.fusion.train")

    env = PortfolioEnv(z_seq, returns)

    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", env, verbose=1, seed=cfg["seed"])
    model.learn(total_timesteps=args.timesteps)

    out = paths.OUTPUTS / "checkpoints" / "ppo_agent.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"[train_ppo] agent -> {out}")


if __name__ == "__main__":
    main()
