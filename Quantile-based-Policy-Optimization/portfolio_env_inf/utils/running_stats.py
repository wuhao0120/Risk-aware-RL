import math


class RunningMeanStd:
    """
    EMA（指数移动平均）在线估计均值和方差

    用于对 return 做归一化, 使策略梯度权重与约束项量级匹配。
    等效窗宽约为 1/decay 条轨迹（默认 decay=0.01 → 约 100 条）。

    使用方法:
        rms = RunningMeanStd(decay=0.01)
        rms.update(10.5)   # 每个 episode 更新一次
        rms.update(11.2)
        normalized = (x - rms.mean) / rms.std
    """

    def __init__(self, decay=0.01):
        """
        Args:
            decay: EMA 衰减系数 α, 等效窗宽 ≈ 1/α
                   - 0.01 → 约最近 100 条轨迹
                   - 0.005 → 约最近 200 条轨迹
                   - 0.02 → 约最近 50 条轨迹
        """
        self.decay = decay         # EMA 衰减系数 α
        self.mean = 0.0            # EMA 均值
        self.var = 1.0             # EMA 方差 (初始为1, 避免训练早期除零)
        self._initialized = False  # 首次更新前用第一个样本直接初始化

    def update(self, x):
        """
        EMA 更新均值和方差

        Args:
            x: 标量, 新的观测值 (e.g., 一条轨迹的折扣回报 U(τ))

        更新公式:
            mean_new = (1-α) · mean_old + α · x
            var_new  = (1-α) · var_old  + α · (x - mean_new)²

        等价于对最近样本的指数衰减加权平均:
            mean = α·x + α(1-α)·x_{-1} + α(1-α)²·x_{-2} + ...
        """
        if not self._initialized:
            # 首次更新: 直接用第一个样本初始化, 避免 mean=0 引起的偏差
            self.mean = x
            self.var = 1.0
            self._initialized = True
            return

        # EMA 均值更新: mean ← (1-α)·mean + α·x
        self.mean = (1.0 - self.decay) * self.mean + self.decay * x

        # EMA 方差更新: var ← (1-α)·var + α·(x - mean_new)²
        # 用更新后的 mean 计算残差, 保持对称性
        self.var = (1.0 - self.decay) * self.var + self.decay * (x - self.mean) ** 2

    @property
    def std(self):
        """返回标准差, 下限 sqrt(1e-8) 避免除零"""
        return math.sqrt(max(self.var, 1e-8))
