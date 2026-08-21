"""跨模態可解釋性：Integrated Gradients 套在訓練好的 PPO policy 上（計畫書 3.6 節 (4)）。

公式：
    Attribution_i(x) = (x_i - x_i') * integral[alpha=0->1] d(pi(a|x' + alpha*(x-x'))) / dx_i  dalpha

x 是實際的 observation（Z_fused z_dim 維 + 前期持倉 1 維，共 z_dim+1 維），x' 是 baseline
（計畫書原文建議用零向量）。用 captum 的 IntegratedGradients 實作這個積分（黎曼和近似，
n_steps 控制切割數）。

pi(a|x) 這裡取 policy 動作分布的 mean（確定性動作），不是隨機取樣的動作——取樣動作對輸入
不是平滑可微分的，mean 才是有意義、可以算梯度的攻擊目標，也是 backtest.py 的 weights_ppo()
在 deterministic=True 時實際採用的動作。

輸出：
    data/outputs/explainability/{TICKER}/attributions.npy   # [N, z_dim+1]，逐日逐維歸因值
    data/outputs/explainability/{TICKER}/summary.json        # 人類可讀摘要

用法：
    python -m module_c_fusion.explainability.integrated_gradients --ticker AAPL

前置：先跑過 module_c_fusion.fusion.train（產出 Z_fused）跟 module_c_fusion.rl.train_ppo
（產出 data/outputs/checkpoints/ppo_agent.zip）。
"""
import argparse

import numpy as np

from shared import paths
from shared.utils import write_json
from module_c_fusion.fusion.consolidate import load_index


class PolicyMeanWrapper:
    """把 stable-baselines3 PPO 的 policy 包成 captum 看得懂的可微分函式：
    輸入 observation [B, z_dim+1]，輸出動作分布的 mean [B, action_dim]。

    寫成一般 class（不是 nn.Module）也可以給 captum 用，只要 __call__ 是可微分的
    torch 運算即可；這裡刻意不繼承 nn.Module，避免 captum 誤把 wrapper 自己的（不存在的）
    參數也納入梯度追蹤，只追蹤 obs 本身。
    """

    def __init__(self, sb3_policy):
        self.policy = sb3_policy

    def __call__(self, obs):
        dist = self.policy.get_distribution(obs)
        return dist.distribution.mean


def load_agent(path):
    """讀取訓練好的 PPO agent，強制用 CPU（IG 只需要跑 forward + 算梯度，不需要 GPU）。"""
    from stable_baselines3 import PPO
    model = PPO.load(path, device="cpu")
    model.policy.eval()
    return model


def rebuild_holding_trajectory(model, z_seq: np.ndarray) -> np.ndarray:
    """依序跑一遍 agent，重建出跟 backtest.py 的 weights_ppo() 一致的前期持倉軌跡。

    不能隨便給零：env.py 的 observation 定義是「Z_fused + 前一天的持倉權重」，如果歸因分析
    時的前期持倉跟 agent 實際運作時看到的不一樣，算出來的歸因就跟真實決策情境脫節，失去
    「解釋 agent 實際做的決策」這個目的。
    """
    weights = np.zeros(len(z_seq), dtype=np.float32)
    w = 0.0
    for i, z in enumerate(z_seq):
        weights[i] = w
        obs = np.concatenate([z, [w]]).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        w = float(np.clip(action[0], 0.0, 1.0))
    return weights


def compute_attributions(model, z_seq: np.ndarray, prev_weights: np.ndarray,
                         n_steps: int = 50) -> np.ndarray:
    """對每一天的 observation 算 Integrated Gradients，回傳 [N, z_dim+1] 的歸因矩陣。

    baseline 用零向量（計畫書原文建議）。逐筆算而不是整批一次算，避免 batch 維度跟
    captum 對 target/attribute 的語意混淆；動作維度只有 1，target 固定填 0。
    """
    import torch
    from captum.attr import IntegratedGradients

    wrapped = PolicyMeanWrapper(model.policy)
    ig = IntegratedGradients(wrapped)

    obs = np.concatenate([z_seq, prev_weights[:, None]], axis=1).astype(np.float32)
    obs_t = torch.from_numpy(obs)
    baseline = torch.zeros_like(obs_t)

    attributions = []
    for i in range(obs_t.shape[0]):
        a = ig.attribute(obs_t[i:i + 1], baseline[i:i + 1], n_steps=n_steps, target=0)
        attributions.append(a.detach().cpu().numpy()[0])
    return np.stack(attributions)


def summarize(attributions: np.ndarray, z_dim: int) -> dict:
    """人類可讀摘要：Z_fused 各維度平均絕對歸因值排序（列前後各 10 維），
    以及「前期持倉」這一維（最後一維，不屬於 Z_fused）的歸因，方便對照。

    只依賴 numpy，不需要 torch，方便獨立單元測試。
    """
    z_attr = attributions[:, :z_dim]
    holding_attr = attributions[:, z_dim]
    mean_abs = np.abs(z_attr).mean(axis=0)
    order = np.argsort(-mean_abs)
    top = [{"dim": int(d), "mean_abs_attribution": float(mean_abs[d])} for d in order[:10]]
    bottom = [{"dim": int(d), "mean_abs_attribution": float(mean_abs[d])} for d in order[-10:]]
    return {
        "n_days": int(attributions.shape[0]),
        "z_dim": int(z_dim),
        "top10_dims_by_mean_abs_attribution": top,
        "bottom10_dims_by_mean_abs_attribution": bottom,
        "holding_dim_mean_abs_attribution": float(np.abs(holding_attr).mean()),
        "z_total_mean_abs_attribution": float(np.abs(z_attr).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--n_steps", type=int, default=50, help="Integrated Gradients 積分黎曼和切割數")
    ap.add_argument("--agent", default=None, help="不指定就用 data/outputs/checkpoints/ppo_agent.zip")
    args = ap.parse_args()

    idx = load_index(args.ticker)
    if idx is None:
        raise SystemExit(f"找不到 {args.ticker} 的 Z_fused 索引，先跑 module_c_fusion.fusion.train")
    z_seq = idx["z"]
    z_dim = z_seq.shape[1]

    agent_path = args.agent or (paths.OUTPUTS / "checkpoints" / "ppo_agent.zip")
    model = load_agent(agent_path)

    prev_weights = rebuild_holding_trajectory(model, z_seq)
    attributions = compute_attributions(model, z_seq, prev_weights, n_steps=args.n_steps)
    summary = summarize(attributions, z_dim)

    out_dir = paths.OUTPUTS / "explainability" / args.ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "attributions.npy", attributions)
    write_json(summary, out_dir / "summary.json")
    print(f"[integrated_gradients] {args.ticker}: {summary['n_days']} 天，"
         f"Z_fused 平均絕對歸因={summary['z_total_mean_abs_attribution']:.6f}，"
         f"最高貢獻維度={summary['top10_dims_by_mean_abs_attribution'][0]} -> {out_dir}")


if __name__ == "__main__":
    main()
