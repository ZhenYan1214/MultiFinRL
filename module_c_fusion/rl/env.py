"""PPO 強化學習環境與獎勵函數（gymnasium API）。

- State：當日 Z_fused + 前期持倉權重。
- Action：投資組合權重（單股版：[現金, 股票] 兩維，softmax 正規化）。
- Reward：風險敏感型 = 期望報酬 - λ_vol * 波動 - λ_mdd * 回撤 - 交易成本。
- 嚴禁 look-ahead：t 日的 action 用 t+1 日的報酬結算。
"""
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # gymnasium 未安裝時仍可 import 本檔（僅供閱讀）
    gym = None
    spaces = None


class PortfolioEnv(gym.Env if gym else object):
    """單股 + 現金的最簡投組環境；多標的版本擴充 action 維度即可。

    z_seq:      [T, z_dim] 每日 Z_fused（依日期排序）
    returns:    [T] 每日次日報酬（close_{t+1}/close_t - 1），與 z_seq 對齊
    """

    def __init__(self, z_seq: np.ndarray, returns: np.ndarray,
                 cost: float = 0.001, lambda_vol: float = 0.1,
                 lambda_mdd: float = 0.1, vol_window: int = 20):
        super().__init__()
        assert len(z_seq) == len(returns)
        self.z_seq = z_seq.astype(np.float32)
        self.returns = returns.astype(np.float32)
        self.cost = cost
        self.lambda_vol = lambda_vol
        self.lambda_mdd = lambda_mdd
        self.vol_window = vol_window

        z_dim = z_seq.shape[1]
        # state = Z_fused + 前期股票權重
        self.observation_space = spaces.Box(-np.inf, np.inf, (z_dim + 1,), np.float32)
        # action = 股票權重 [0, 1]（其餘為現金）
        self.action_space = spaces.Box(0.0, 1.0, (1,), np.float32)
        self._reset_state()

    def _reset_state(self):
        self.t = 0
        self.weight = 0.0
        self.nav = 1.0
        self.peak = 1.0
        self.recent_rewards: list[float] = []

    def _obs(self):
        return np.concatenate([self.z_seq[self.t], [self.weight]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action):
        w = float(np.clip(action[0], 0.0, 1.0))
        turnover = abs(w - self.weight)
        r_portfolio = w * self.returns[self.t] - turnover * self.cost

        self.nav *= (1.0 + r_portfolio)
        self.peak = max(self.peak, self.nav)
        drawdown = 1.0 - self.nav / self.peak

        self.recent_rewards.append(r_portfolio)
        vol = float(np.std(self.recent_rewards[-self.vol_window:]))

        reward = r_portfolio - self.lambda_vol * vol - self.lambda_mdd * drawdown

        self.weight = w
        self.t += 1
        terminated = self.t >= len(self.z_seq) - 1
        info = {"nav": self.nav, "drawdown": drawdown, "weight": w}
        return (self._obs() if not terminated else np.zeros_like(self._obs()),
                reward, terminated, False, info)
