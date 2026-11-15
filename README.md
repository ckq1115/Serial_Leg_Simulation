# 串联轮腿底盘MuJoCo仿真

基于 MuJoCo 的串联轮腿底盘仿真。包含闭链URDF模型，RMUC2027场地（能钻狗洞 能打飞镖 能飞小飞机）。

## 安装

在项目根目录运行：

```bash
python -m pip install mujoco numpy scipy pyyaml pynput
```

如需重新生成动力学和控制增益表，另装 `sympy`。

## 快速开始

```bash
# 默认场景：RMUC2027，打开查看器
python sim/main.py

# 平地自主平衡，无界面运行 5 秒
python sim/main.py --scene mjcf/bigeye_flat.xml --control-mode lqr --headless --duration 5

# 启用 MPC
python sim/main.py --balance-controller mpc

# 运行 LESO 版本
python sim/main_leso.py
```

查看器中，方向键控制行驶和转向，左、右 Shift 分别升高和降低车身，J 触发单次跳跃，Space 暂停或继续仿真。在主入口中，M 可切换 LQR 和 MPC。更多参数见 `python sim/main.py --help`。

## 控制方式

- **LQR + VMC**：十状态 LQR 控制纵向平衡、偏航和腿摆角，按左右腿长调度增益。独立支撑控制调节高度和横滚，VMC 将虚拟力和摆矩映射为髋电机力矩。
- **MPC + VMC**：十状态 MPC 计算左右腿摆矩和轮矩。腿长（双轮支撑时为车身高度）和横滚由与 LQR 共用的独立支撑反馈调节。
- **LESO**：通过独立入口运行线性扩张状态观测器版本。

**LESO 和 MPC 的参数目前均未细调，控制效果可能不是很好。**

## 项目结构

| 路径 | 内容 |
| --- | --- |
| [`sim/`](sim/) | 仿真入口、控制器和腿部运动学 |
| [`mjcf/`](mjcf/) | 机器人模型和场景入口 |
| [`RMUC2027/`](RMUC2027/) | RMUC2027 场地及资源 |
| [`config/robot_params.yaml`](config/robot_params.yaml) | 机器人参数和控制配置 |
| [`tools/linearize.py`](tools/linearize.py) | 离线动力学线性化与控制增益生成 |
| [`data/`](data/) | 线性化数据、增益表和导出的 C 头文件 |

运行 `python tools/linearize.py` 可重新生成控制数据。若只需用已有线性化数据生成 LESO 表，运行 `python tools/linearize.py --leso-only --leso-check-grid 81`。
