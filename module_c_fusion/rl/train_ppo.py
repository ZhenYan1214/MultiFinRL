"""用 PPO（stable-baselines3）訓練投資組合配置 agent。

第一年是否需完整 PPO 或 rule-based 回測即可，尚未定案
（docs/decisions.md 待確認事項）；環境與訓練入口已架好，隨時可跑。

curriculum learning（計畫書 3.6 節 anticipated challenge (2)：「先在低波動的簡化市場環境訓練，
再逐步導入高波動／崩盤等複雜情境，以提升模型在極端行情下的穩健性」）預設關閉，用 --curriculum
開啟，方便直接跟不開的版本做 A/B 對照（decisions.md #52）。

用法：
    python -m module_c_fusion.rl.train_ppo --fake                       # 假 Z_fused 測通
    python -m module_c_fusion.rl.train_ppo --ticker AAPL                # 只用 train split，一次性訓練
    python -m module_c_fusion.rl.train_ppo --ticker AAPL --curriculum   # 真實 Z_fused，curriculum learning
"""
import argparse
from pathlib import Path

import numpy as np

from shared import paths
from shared.utils import load_config, read_json, write_json
from module_c_fusion.rl.env import PortfolioEnv
from module_c_fusion.fusion.consolidate import load_index


def validate_agent_metadata(agent_path, ticker: str, index: dict) -> None:
    """拒絕把舊資料或別支股票訓練的 PPO agent 套到目前的 Z_fused。"""
    agent_path = Path(agent_path)
    meta_path = agent_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise ValueError(f"{agent_path} 缺少新版訓練 metadata；請重訓 PPO")
    meta = read_json(meta_path)
    expected_hash = str(index.get("z_fused_sha256", ""))
    if (meta.get("ticker") != ticker or meta.get("training_split") != "train"
            or not expected_hash or meta.get("z_fused_sha256") != expected_hash):
        raise ValueError(
            f"{agent_path} 與 {ticker} 目前的 train-only Z_fused 不相容；請重訓 PPO"
        )


def load_real(ticker: str):
    """讀 Z_fused 序列與對應次日報酬（用 A 的 future_closes[0] 對 close_t0）。

    只讀帶資料指紋的彙整索引，避免混用殘留的逐日 Z_fused。
    """
    idx = load_index(ticker)
    if idx is None or "split" not in idx:
        raise ValueError("缺少新版 Z_fused index；請重跑 fusion.train/consolidate")
    return idx["dates"], idx["z"], idx["return_next"], idx["split"]


def make_fake(n: int = 200, z_dim: int = 768, seed: int = 42):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, z_dim)).astype(np.float32),
            rng.normal(0.0005, 0.02, n).astype(np.float32))


def compute_trailing_vol(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """逐日估計「截至當天為止」的滾動波動度（trailing std），只看得到過去，不會用到未來報酬，
    符合專案一貫的 no-look-ahead 原則。天數不足一個 window 時，用目前累積到的天數計算
    （不會產生 NaN，也不需要丟掉最前面幾天）。只依賴 numpy，方便獨立單元測試。
    """
    n = len(returns)
    vol = np.zeros(n, dtype=np.float32)
    for i in range(n):
        window_slice = returns[max(0, i - window + 1):i + 1]
        vol[i] = float(np.std(window_slice)) if len(window_slice) > 1 else 0.0
    return vol


def curriculum_stages(z_seq: np.ndarray, returns: np.ndarray, n_stages: int = 3,
                      vol_window: int = 20, min_stage_days: int = 30):
    """依「截至當天為止的滾動波動度」把資料切成 n_stages 個由簡入繁的訓練階段（計畫書 3.6 節
    anticipated challenge (2)）。

    每一階段都是一段「連續」交易日：先找固定長度內平均 trailing volatility 最低的連續區間，
    後續階段再由左右相鄰日期中較平穩的一側逐日向外擴張，最後一階段強制涵蓋完整 train split。
    這樣仍符合由低波動到完整市場的 curriculum，也不會把相隔很遠的低波動日期硬接成連續 episode，
    造成 agent 在中間缺失期間持倉卻沒有計入報酬的失真。難度排序只使用 train split，不接觸
    validation/test；單日波動度本身則只用截至當日的歷史報酬計算。

    回傳：[(z_stage1, returns_stage1), ..., (z_seq, returns)]，長度 n_stages，最後一個元素
    一定是完整資料集。
    """
    n = len(returns)
    if len(z_seq) != n:
        raise ValueError("z_seq 與 returns 長度必須一致")
    if n == 0:
        raise ValueError("curriculum 至少需要 1 天資料")
    if n_stages < 1:
        raise ValueError("n_stages 必須 >= 1")

    vol = compute_trailing_vol(returns, vol_window)
    first_days = min(n, max(1, min_stage_days, int(np.ceil(n / n_stages))))

    # 找平均 trailing volatility 最低的第一段連續區間。
    window_sums = np.convolve(vol.astype(np.float64), np.ones(first_days), mode="valid")
    start = int(np.argmin(window_sums))
    end = start + first_days

    stages = []
    for i in range(1, n_stages + 1):
        target_days = min(n, max(first_days, int(np.ceil(n * i / n_stages))))
        while end - start < target_days:
            left_vol = vol[start - 1] if start > 0 else np.inf
            right_vol = vol[end] if end < n else np.inf
            if left_vol <= right_vol:
                start -= 1
            else:
                end += 1
        stages.append((z_seq[start:end], returns[start:end]))
    stages[-1] = (z_seq, returns)  # 最後一階段強制用完整資料，避免因為 cutoff 邊界少算幾天
    return stages


def curriculum_stage_timesteps(total_timesteps: int, n_stages: int) -> list[int]:
    """依難度遞增分配每階段的訓練步數：越後面（越接近完整/複雜資料）的階段分到越多步數。

    decisions.md #53：均分步數（每階段 total/n_stages）實測會訓出「永遠空手」的退化 policy——
    推測是簡單階段（低波動子集）步數足夠讓 policy 收斂到「不持倉最安全」這個局部最優，
    PPO 的 clip 機制讓後面階段的更新很難把它拉回來。改成線性遞增權重（1,2,...,n_stages，
    正規化到總和等於 total_timesteps），在維持 A/B 對照總計算預算相同的前提下，讓完整資料
    階段取得最多步數，前面簡單階段步數壓低、減少它收斂到局部最優的機會。
    """
    weights = np.arange(1, n_stages + 1, dtype=np.float64)
    raw = total_timesteps * weights / weights.sum()
    steps = np.floor(raw).astype(int)
    steps[-1] += total_timesteps - int(steps.sum())  # 補足取整數損失的步數，全部補到最後一階段
    return steps.tolist()


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--timesteps", type=int, default=10_000)
    ap.add_argument("--curriculum", action="store_true",
                    help="開啟 curriculum learning：先在連續低波動區間訓練，逐步擴大到含高波動/崩盤"
                         "的完整資料（計畫書 3.6 節 anticipated challenge (2)，見 decisions.md #52）。"
                         "預設關閉，行為與加入這個選項之前完全一樣，方便直接跟不開的版本做對照。")
    ap.add_argument("--curriculum_stages", type=int, default=3, help="curriculum learning 的階段數")
    ap.add_argument("--ent_coef", type=float, default=0.0,
                    help="PPO 的 entropy 係數，越大越鼓勵探索（stable-baselines3 預設 0.0，"
                         "這裡沿用同樣的預設值，不動預設行為）。如果 --curriculum 又收斂到「永遠"
                         "空手」的退化 policy，可以試著調大（例如 0.01）避免 action 分布的標準差"
                         "太快收斂到接近 0、探索停滯（decisions.md #53）")
    args = ap.parse_args()

    if args.fake:
        z_seq, returns = make_fake(z_dim=cfg["fusion"]["z_dim"], seed=cfg["seed"])
        train_dates = np.array([], dtype=str)
        z_fused_sha256 = "synthetic"
    else:
        dates, z_all, returns_all, splits = load_real(args.ticker)
        current_index = load_index(args.ticker)
        z_fused_sha256 = str(current_index["z_fused_sha256"])
        train_mask = splits == "train"
        train_dates = dates[train_mask]
        z_seq, returns = z_all[train_mask], returns_all[train_mask]
        if len(z_seq) < 30:
            raise SystemExit("train split 的 Z_fused 不足，先重跑 build_dataset 與 fusion.train")
        print(f"[train_ppo] 僅使用 train split：{len(z_seq)} 天 "
              f"({train_dates[0]} ~ {train_dates[-1]})")

    from stable_baselines3 import PPO

    if args.curriculum:
        stages = curriculum_stages(z_seq, returns, n_stages=args.curriculum_stages)
        stage_steps = curriculum_stage_timesteps(args.timesteps, len(stages))

        env = PortfolioEnv(*stages[0])
        model = PPO("MlpPolicy", env, verbose=1, seed=cfg["seed"], ent_coef=args.ent_coef)
        for i, (z_stage, r_stage) in enumerate(stages):
            if i > 0:
                model.set_env(PortfolioEnv(z_stage, r_stage))
            steps = stage_steps[i]
            print(f"[train_ppo] curriculum stage {i + 1}/{len(stages)}："
                 f"{len(z_stage)} 個連續交易日（逐步擴大到完整 train），timesteps={steps}")
            model.learn(total_timesteps=steps, reset_num_timesteps=False)
        if hasattr(model.policy, "log_std"):
            print(f"[train_ppo] 訓練後 policy log_std = {model.policy.log_std.detach().cpu().numpy()}"
                 "（數字越負代表 action 分布標準差越小、探索越少，是判斷有沒有收斂到「不動」"
                 "policy 的診斷線索，見 decisions.md #53）")
    else:
        env = PortfolioEnv(z_seq, returns)
        model = PPO("MlpPolicy", env, verbose=1, seed=cfg["seed"], ent_coef=args.ent_coef)
        model.learn(total_timesteps=args.timesteps)

    suffix = "fake" if args.fake else args.ticker
    out = paths.OUTPUTS / "checkpoints" / f"ppo_agent_{suffix}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    write_json({
        "ticker": suffix,
        "training_split": "synthetic" if args.fake else "train",
        "n_days": int(len(z_seq)),
        "period": [str(train_dates[0]), str(train_dates[-1])] if len(train_dates) else [],
        "timesteps": args.timesteps,
        "curriculum": args.curriculum,
        "curriculum_stages": args.curriculum_stages if args.curriculum else None,
        "seed": cfg["seed"],
        "z_fused_sha256": z_fused_sha256,
    }, out.with_suffix(".meta.json"))
    print(f"[train_ppo] agent -> {out}")


if __name__ == "__main__":
    main()
