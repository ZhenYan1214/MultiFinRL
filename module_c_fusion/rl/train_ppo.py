"""用 PPO（stable-baselines3）訓練投資組合配置 agent。

第一年是否需完整 PPO 或 rule-based 回測即可，尚未定案
（docs/decisions.md 待確認事項）；環境與訓練入口已架好，隨時可跑。

curriculum learning（計畫書 3.6 節 anticipated challenge (2)：「先在低波動的簡化市場環境訓練，
再逐步導入高波動／崩盤等複雜情境，以提升模型在極端行情下的穩健性」）預設關閉，用 --curriculum
開啟，方便直接跟不開的版本做 A/B 對照（decisions.md #52）。

用法：
    python -m module_c_fusion.rl.train_ppo --fake                       # 假 Z_fused 測通
    python -m module_c_fusion.rl.train_ppo --ticker AAPL                # 真實 Z_fused，一次性訓練
    python -m module_c_fusion.rl.train_ppo --ticker AAPL --curriculum   # 真實 Z_fused，curriculum learning
"""
import argparse

import numpy as np

from shared import paths
from shared.utils import load_config, read_json
from module_c_fusion.rl.env import PortfolioEnv
from module_c_fusion.fusion.consolidate import load_index


def load_real(ticker: str):
    """讀 Z_fused 序列與對應次日報酬（用 A 的 future_closes[0] 對 close_t0）。

    優先讀彙整索引；索引不存在時 fallback 成逐日掃描。
    """
    idx = load_index(ticker)
    if idx is not None:
        return idx["z"], idx["return_next"]

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

    每一階段是「目前難度以內」的天數累積集合（不是互斥分段）——階段 1 只挑全部資料裡波動度
    排名最平穩的一批，階段 2 放寬到更多天，以此類推，最後一階段強制用完整資料（含最劇烈的
    崩盤/高波動時期），保證不會因為排名邊界漏掉任何一天。挑出來的天數依然維持原本的時間序
    （不是按波動度排序餵進 env），因為 env 的前期持倉狀態是逐步累積的，要維持「照時間往前走」
    的意義；篩選天數本身不影響 no-look-ahead，因為每天的 next-day return 是 A/C 事先算好、
    跟著那天走的固定值，不是在 env 裡臨時往未來看。

    回傳：[(z_stage1, returns_stage1), ..., (z_seq, returns)]，長度 n_stages，最後一個元素
    一定是完整資料集。
    """
    n = len(returns)
    vol = compute_trailing_vol(returns, vol_window)
    order = np.argsort(vol, kind="stable")  # 由平穩到劇烈的日期排名，只是用來決定「這階段收哪些天」

    stages = []
    for i in range(1, n_stages + 1):
        cutoff = min(n, max(min_stage_days, int(np.ceil(n * i / n_stages))))
        keep_idx = np.sort(order[:cutoff])  # 還原成時間序，不是波動度排序
        stages.append((z_seq[keep_idx], returns[keep_idx]))
    stages[-1] = (z_seq, returns)  # 最後一階段強制用完整資料，避免因為 cutoff 邊界少算幾天
    return stages


def curriculum_stage_timesteps(total_timesteps: int, n_stages: int) -> list[int]:
    """依難度遞增分配每階段的訓練步數：越後面（越接近完整/複雜資料）的階段分到越多步數。

    decisions.md #53：均分步數（每階段 total/n_stages）實測會訓出「永遠空手」的退化 policy——
    推測是簡單階段（低波動子集）步數足夠讓 policy 收斂到「不持倉最安全」這個局部最優，
    PPO 的 clip 機制讓後面階段的更新很難把它拉回來。改成線性遞增權重（1,2,...,n_stages，
    正規化到總和等於 total_timesteps），讓最後（完整資料）階段拿到的步數不少於原本一次性
    訓練的量，前面階段步數壓低、減少它收斂到局部最優的機會。
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
                    help="開啟 curriculum learning：先在低波動天數訓練，逐步擴大到含高波動/崩盤"
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
    else:
        z_seq, returns = load_real(args.ticker)
        if len(z_seq) < 30:
            raise SystemExit("Z_fused 不足，先跑 module_c_fusion.fusion.train")

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
                 f"{len(z_stage)} 天（波動度門檻遞增），timesteps={steps}")
            model.learn(total_timesteps=steps, reset_num_timesteps=False)
        if hasattr(model.policy, "log_std"):
            print(f"[train_ppo] 訓練後 policy log_std = {model.policy.log_std.detach().cpu().numpy()}"
                 "（數字越負代表 action 分布標準差越小、探索越少，是判斷有沒有收斂到「不動」"
                 "policy 的診斷線索，見 decisions.md #53）")
    else:
        env = PortfolioEnv(z_seq, returns)
        model = PPO("MlpPolicy", env, verbose=1, seed=cfg["seed"], ent_coef=args.ent_coef)
        model.learn(total_timesteps=args.timesteps)

    out = paths.OUTPUTS / "checkpoints" / "ppo_agent.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"[train_ppo] agent -> {out}")


if __name__ == "__main__":
    main()
