"""双轮支撑时的有限时域 MPC，使用固定惩罚系数的 ADMM。"""
import json

import numpy as np
from scipy.linalg import solve_discrete_are
try:
    from numba import njit
except ImportError:
    njit = None


def _riccati_factors(a, b, q, terminal, regularized_r, horizon):
    """由终端代价倒推每步反馈增益和输入代价的逆。"""
    gains = np.empty((horizon, 4, 10))
    input_inverse = np.empty((horizon, 4, 4))
    closed_loop_t = np.empty((horizon, 10, 10))
    p = terminal
    identity = np.eye(4)
    for k in range(horizon - 1, -1, -1):
        bt_p = b.T @ p
        h = regularized_r + bt_p @ b
        input_inverse[k] = np.linalg.solve(h, identity)
        gains[k] = input_inverse[k] @ (bt_p @ a)
        closed_loop = a - b @ gains[k]
        closed_loop_t[k] = closed_loop.T
        p = q + a.T @ p @ closed_loop
        p = (p + p.T) * 0.5
    return gains, input_inverse, closed_loop_t


def _riccati_input_compiled(measured, state_cost, input_cost, a, b, gains,
                   input_inverse, closed_loop_t, output):
    """先倒推仿射项，再从实测状态前推整段输入。"""
    p = state_cost[-1].copy()
    next_p = np.empty_like(p)
    gradient = np.empty(4)
    for k in range(output.shape[0] - 1, -1, -1):
        for i in range(4):
            value = input_cost[k, i]
            for j in range(10):
                value += b[j, i] * p[j]
            gradient[i] = value
        for i in range(4):
            value = 0.
            for j in range(4):
                value -= input_inverse[k, i, j] * gradient[j]
            output[k, i] = value
        for i in range(10):
            value = state_cost[k, i]
            for j in range(10):
                value += closed_loop_t[k, i, j] * p[j]
            for j in range(4):
                value -= gains[k, j, i] * input_cost[k, j]
            next_p[i] = value
        p, next_p = next_p, p

    state = measured.copy()
    next_state = np.empty_like(state)
    for k in range(output.shape[0]):
        for i in range(4):
            value = output[k, i]
            for j in range(10):
                value -= gains[k, i, j] * state[j]
            output[k, i] = value
        for i in range(10):
            value = 0.
            for j in range(10):
                value += a[i, j] * state[j]
            for j in range(4):
                value += b[i, j] * output[k, j]
            next_state[i] = value
        state, next_state = next_state, state


def _riccati_input(measured, state_cost, input_cost, a, b, gains,
                   input_inverse, closed_loop_t, output):
    p = state_cost[-1].copy()
    for k in range(output.shape[0] - 1, -1, -1):
        output[k] = -input_inverse[k] @ (b.T @ p + input_cost[k])
        p = state_cost[k] + closed_loop_t[k] @ p - gains[k].T @ input_cost[k]
    state = measured
    for k in range(output.shape[0]):
        output[k] -= gains[k] @ state
        state = a @ state + b @ output[k]


if njit is not None:
    _riccati_factors = njit(cache=False)(_riccati_factors)
    _riccati_input = njit(cache=False)(_riccati_input_compiled)


class BalancePrediction:
    def __init__(self, data_dir):
        with (data_dir / "linearization_metadata.json").open(encoding="utf-8") as file:
            weights = json.load(file)
        self.Q = np.diag(weights["Q_diagonal"])
        self.R = np.asarray(weights["R_matrix"])
        with np.load(data_dir / "AB_sampling_points.npz", allow_pickle=False) as samples:
            left, right = samples["L_l"], samples["L_r"]
            self.left_grid = np.unique(left)
            self.right_grid = np.unique(right)
            order = np.lexsort((right, left))
            shape = (self.left_grid.size, self.right_grid.size)
            self.A = samples["A_d"][order].reshape(*shape, 10, 10)
            self.B = samples["B_d"][order].reshape(*shape, 10, 4)
            equilibrium = np.column_stack((samples["e2"], samples["e3_l"], samples["e3_r"]))
            self.equilibrium = equilibrium[order].reshape(*shape, 3)
            self.feedforward = samples["u_eq"][order].reshape(*shape, 4)
        # 初始化时求完网格终端权重，升降跨网格时不再触发 DARE。
        self.terminal_by_node = np.empty_like(self.A)
        for node in np.ndindex(shape):
            self.terminal_by_node[node] = solve_discrete_are(
                self.A[node], self.B[node], self.Q, self.R)
        self.cached_lengths = None

    def interpolate(self, lengths):
        left = float(np.clip(lengths[0], self.left_grid[0], self.left_grid[-1]))
        right = float(np.clip(lengths[1], self.right_grid[0], self.right_grid[-1]))
        if (left, right) == self.cached_lengths:
            return self.cached_model

        i = int(np.clip(np.searchsorted(self.left_grid, left) - 1, 0, self.left_grid.size - 2))
        j = int(np.clip(np.searchsorted(self.right_grid, right) - 1, 0, self.right_grid.size - 2))
        l = (left - self.left_grid[i]) / (self.left_grid[i+1] - self.left_grid[i])
        r = (right - self.right_grid[j]) / (self.right_grid[j+1] - self.right_grid[j])
        nodes = ((i, j), (i+1, j), (i, j+1), (i+1, j+1))
        weights = ((1-l)*(1-r), l*(1-r), (1-l)*r, l*r)
        self.cached_model = tuple(
            sum(weight * values[node] for node, weight in zip(nodes, weights))
            for values in (self.A, self.B, self.terminal_by_node, self.equilibrium, self.feedforward))
        self.cached_lengths = (left, right)
        return self.cached_model


class BalanceMPC(BalancePrediction):
    def __init__(self, data_dir, dt, horizon=24, rho=2.0):
        super().__init__(data_dir)
        self.dt = dt
        self.horizon = horizon
        self.rho = rho
        self.cached_factor_lengths = None
        self.reset()
        self.prediction_times = dt * np.arange(1, horizon + 1)
        self.regularized_input_cost = self.R + rho * np.eye(4)
        if njit is not None:
            middle = self.left_grid.size // 2, self.right_grid.size // 2
            a, b = self.A[middle], self.B[middle]
            factors = _riccati_factors(a, b, self.Q, self.Q,
                                       self.regularized_input_cost, horizon)
            _riccati_input(np.zeros(10), np.zeros((horizon + 1, 10)),
                           np.zeros((horizon, 4)), a, b, *factors, np.empty((horizon, 4)))

    def reset(self):
        self.plan = None
        self.dual = None
        self.previous_feedforward = None
        self.iterations = 0
        self.residual = np.inf

    def _prediction_model(self, lengths):
        a, b, terminal, equilibrium, feedforward = self.interpolate(lengths)
        if self.cached_lengths == self.cached_factor_lengths:
            return self.cached_prediction
        factors = _riccati_factors(a, b, self.Q, terminal, self.regularized_input_cost,
                                   self.horizon)
        self.cached_prediction = (a, b, *factors), terminal, equilibrium, feedforward
        self.cached_factor_lengths = self.cached_lengths
        return self.cached_prediction

    def solve(self, lengths, state, target, bounds):
        measured = np.asarray(state).reshape(10).copy()
        target = np.asarray(target).reshape(10)
        if not (np.all(np.isfinite(lengths)) and np.all(np.isfinite(measured))
                and np.all(np.isfinite(target))):
            self.reset()
            return None
        model, terminal, equilibrium, feedforward = self._prediction_model(lengths)
        reference = np.tile(target, (self.horizon, 1))
        # 预测使用静态平衡偏置，跟踪目标由当前指令单独给出。
        measured[2:5] -= equilibrium
        reference[:, 2:5] -= equilibrium
        reference[:, 0] += self.prediction_times * reference[:, 5]
        reference[:, 1] += self.prediction_times * reference[:, 6]
        state_cost = np.zeros((self.horizon + 1, 10))
        state_cost[1:-1] = -reference[:-1] @ self.Q
        state_cost[-1] = -terminal @ reference[-1]
        primal = np.empty((self.horizon, 4))
        # 调用方已用本拍的雅可比与轴向力将电机限幅换算为虚拟摆矩区间。
        # 整个预测域使用同一组区间。
        bounds = np.asarray(bounds)
        lower = bounds[:, 0] - feedforward
        upper = bounds[:, 1] - feedforward
        if self.plan is None:
            input_cost = np.zeros_like(primal)
            _riccati_input(measured, state_cost, input_cost, *model, primal)
            projected = np.clip(primal, lower, upper)
            dual = np.zeros_like(projected)
        else:
            # 前馈随腿长变化，先将上一拍的实际力矩转回本拍的增量坐标。
            shifted = np.concatenate((self.plan[1:], self.plan[-1:]))
            shifted += self.previous_feedforward - feedforward
            projected = np.clip(shifted, lower, upper)
            dual = np.concatenate((self.dual[1:], self.dual[-1:]))
        for iteration in range(1, 61):
            input_cost = self.rho * (dual - projected)
            _riccati_input(measured, state_cost, input_cost, *model, primal)
            relaxed = 1.6 * primal - .6 * projected
            previous = projected
            projected = np.clip(relaxed + dual, lower, upper)
            dual += relaxed - projected
            self.residual = max(np.max(np.abs(primal - projected)),
                                self.rho * np.max(np.abs(projected - previous)))
            if self.residual < .002:
                break
        if not np.isfinite(self.residual) or not np.all(np.isfinite(projected)):
            self.reset()
            self.iterations = iteration
            return None
        # 达到迭代上限时仍可执行已投影的可行力矩。
        self.iterations = iteration
        self.plan, self.dual = projected, dual
        self.previous_feedforward = feedforward
        return (feedforward + projected[0]).reshape(4, 1)
