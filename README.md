# 串联五连杆轮腿机器人仿真

基于 MuJoCo 物理引擎的串联五连杆轮腿机器人仿真平台。核心控制器采用 **10 维状态空间 LQR + VMC（虚拟模型控制）** 方案，支持键盘遥控与 LQR 自主平衡两种运行模式。

## 目录

- [系统架构](#系统架构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [运行模式](#运行模式)
  - [键盘遥控模式](#键盘遥控模式)
  - [LQR 自主平衡模式](#lqr-自主平衡模式)
- [命令行参数](#命令行参数)
- [配置文件说明](#配置文件说明)
- [项目结构](#项目结构)
- [工具脚本](#工具脚本)
  - [LQR 增益表生成](#lqr-增益表生成)
- [控制理论背景](#控制理论背景)
- [实机部署参考](#实机部署参考)

---

## 系统架构

```
┌──────────────────────────────────────────────────┐
│                   main.py                         │
│          (参数解析 / 仿真循环 / 日志)               │
├──────────────────────────────────────────────────┤
│                StandController                    │
│  ┌─────────────┬──────────────┬────────────────┐ │
│  │ KalmanOdometry│  LQR Governor │ FiveLinkLeg   │ │
│  │  (状态估计)   │ (键盘→目标值) │  (五连杆运动学) │ │
│  ├─────────────┼──────────────┼────────────────┤ │
│  │  VmcMapper   │   PID ×5     │ Chebyshev 2D  │ │
│  │  (力矩映射)   │ (腿长/横滚)   │  (增益调度)    │ │
│  └─────────────┴──────────────┴────────────────┘ │
├──────────────────────────────────────────────────┤
│              MuJoCo 物理引擎                       │
│         mjcf/serial_leg.xml (机器人模型)           │
└──────────────────────────────────────────────────┘
```

控制流程简述（对应手册 §13 chassis_task 管线）：

1. **感知（§2）**：从 MuJoCo 读取传感器 + 五连杆运动学正解 → 虚拟腿长 L 和摆角 θ
2. **估计（§7）**：Kalman 滤波器融合 IMU 加速度与轮式里程计 → 水平位移 x 与速度 ẋ
3. **规划（§4-5, §10）**：Chebyshev 增益调度 K(L_l, L_r) + 键盘参考调速器 + 跳跃状态机
4. **控制（§4, §8）**：LQR 最优反馈 u=-Kx + PID 腿长/Roll + 重力前馈
5. **映射（§6, §9）**：VMC 力矩映射 τ=JᵀF + 离地检测 K 矩阵抑制
6. **执行**：力矩限幅 → MuJoCo `data.ctrl` → `mj_step()`
7. **遥测**：UDP 发送 24 维状态数据（供 VOFA+ 可视化）

---

## 环境要求

| 依赖 | 说明 |
|------|------|
| Python 3.9+ | 建议使用 conda/venv 虚拟环境 |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | 物理引擎（`pip install mujoco`） |
| NumPy | 数值计算 |
| SciPy | 线性化工具需要（`fsolve`, `expm`, `least_squares` 等） |
| PyYAML | 读取配置文件 |
| SymPy | 符号动力学推导（仅 `tools/linearize.py` 需要） |
| pynput | 键盘读取（仅键盘遥控模式需要） |

安装依赖：

```bash
pip install mujoco numpy scipy pyyaml sympy pynput
```

---

## 快速开始

### 1. 键盘遥控模式（默认）

```bash
cd Serial_Leg_Simulation
python sim/main.py
```

启动后会弹出 MuJoCo 可视化窗口，机器人自动站立。使用键盘控制：

| 按键 | 功能 |
|------|------|
| `↑` / `↓` | 前进 / 后退 |
| `←` / `→` | 左转 / 右转 |
| `Shift`（左/右） | 升高 / 降低车身 |
| `Space` | 跳跃（现在会疯） |

### 2. LQR 自主平衡模式

```bash
python sim/main.py --control-mode lqr
```

机器人站在原点自主平衡，不做移动。

### 3. 无界面模式（Headless）

```bash
python sim/main.py --headless --duration 10
```

适合批量测试、自动化脚本。运行 10 秒后自动退出。

### 4. 施加外部扰动

```bash
python sim/main.py --push-time 2.0 --push-duration 0.5 --push-force-x 50
```

在 t=2.0s 时对车身施加 x 方向 50N 的冲量扰动，持续 0.5s。常用于测试抗扰能力。

```bash
python sim/main.py --initial-pitch 0.1 --initial-x-velocity 1.0
```

设置初始俯仰角 0.1 rad 和初始 x 速度 1.0 m/s，测试从非零初始状态的恢复能力。

---

## 运行模式

### 键盘遥控模式

默认模式（`--control-mode keyboard`）。键盘输入经过 **LqrReferenceGovernor** 处理：

- 速度指令以配置的加速度平滑斜坡变化（`speed_accel_mps2` / `speed_release_accel_mps2`）
- 偏航指令积分为累积 yaw 角度目标，并保留 yaw 角速度目标
- 松开速度键时以释放减速度（release accel）快速停止
- 支持 `instant_stop_on_release` 让速度瞬时刹车

键盘指令转换为 LQR 的位置/yaw 角目标，并叠加位置预览、速度误差和轮端前馈。

### LQR 自主平衡模式

（`--control-mode lqr`）机器人维持在初始位置和姿态。可通过命令行参数设置目标偏置：

```bash
python sim/main.py --control-mode lqr --target-x 1.0
```

让机器人向前移动 1 米后停下。

---

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--duration` | float | `0.0` | 运行时长（秒），0 表示无限运行 |
| `--viewer` / `--headless` | flag | `--viewer` | 是否显示 MuJoCo 可视化窗口 |
| `--control-mode` | str | `keyboard` | 控制模式：`keyboard` 或 `lqr` |
| `--target-leg-length` | float | 自动 | 目标腿长（m），默认从初始姿态读取 |
| `--support-force` | float | `0.5*m*g` | 单腿支撑力（N） |
| `--target-x` | float | `0.0` | LQR 目标 x 位置（m） |
| `--target-yaw` | float | `0.0` | LQR 目标偏航角（rad） |
| `--target-pitch` | float | `0.0` | LQR 目标俯仰角（rad） |
| `--target-x-dot` | float | `0.0` | LQR 目标 x 速度（m/s） |
| `--target-yaw-dot` | float | `0.0` | LQR 目标偏航角速度（rad/s） |
| `--max-keyboard-speed` | float | `1.50` | 键盘最大速度（m/s） |
| `--max-keyboard-yaw-rate` | float | `5.00` | 键盘最大偏航角速度（rad/s） |
| `--keyboard-speed-accel` | float | `3.00` | 键盘速度加速度（m/s²） |
| `--keyboard-speed-release-accel` | float | `4.00` | 键盘速度释放减速度（m/s²） |
| `--keyboard-yaw-rate-accel` | float | `1.50` | 键盘偏航加速度（rad/s²） |
| `--keyboard-yaw-rate-release-accel` | float | `2.50` | 键盘偏航释放减速度（rad/s²） |
| `--initial-pitch` | float | `0.0` | 初始俯仰角（rad） |
| `--initial-x-velocity` | float | `0.0` | 初始 x 速度（m/s） |
| `--push-body` | str | `base` | 推动目标刚体名称 |
| `--push-time` | float | `1.0` | 推力开始时间（s） |
| `--push-duration` | float | `0.0` | 推力持续时间（s） |
| `--push-force-x/y/z` | float | `0.0` | 推力分量（N） |
| `--push-torque-x/y/z` | float | `0.0` | 推力矩分量（Nm） |
| `--joint-torque-limit` | float | `40.0` | 关节力矩限制（Nm） |
| `--wheel-torque-limit` | float | `5.0` | 轮端力矩限制（Nm） |
| `--length-position-force-limit` | float | `1500.0` | 腿长位置环力限制（N） |
| `--length-velocity-force-limit` | float | `1000.0` | 腿长速度环力限制（N） |
| `--roll-force-limit` | float | `1000.0` | 横滚力限制（N） |
| `--fall-angle` | float | `1.0` | 摔倒判定角（rad），超出此角度停止仿真 |
| `--log-every` | float | `0.5` | 日志打印间隔（s） |

---

## 配置文件说明

所有机器人物理参数和初始控制参数集中在 `config/robot_params.yaml` 中，MATLAB 建模、MuJoCo 仿真、Python 控制共享同一套参数。

### 主要配置段

| 段 | 说明 |
|----|------|
| `simulation` | 仿真步长（500Hz）、积分器（RK4）、重力加速度 |
| `geometry` | 轮径、轮距、车身尺寸、五连杆各段杆长 |
| `mass_properties` | 总质量、各部分质量与转动惯量 |
| `mujoco` | MuJoCo 模型文件路径、关节/传感器/执行器命名 |
| `actuators` | 电机力矩限幅、轮毂电机功率模型参数 |
| `contact` | 轮地接触摩擦参数、闭环运动学约束求解参数 |
| `control_initial` | PID 参数、LQR 目标偏置、键盘指令参数、Kalman 噪声参数 |

### 增益调度

增益矩阵使用 2D Chebyshev 多项式 `K(L_l, L_r)`，系数存储在 `data/lqr_gain_fit.py`。
仿真直接调用 `get_K(L_l, L_r)` 实时计算 4×10 增益矩阵，无需查表插值。

---

## 项目结构

```
Serial_Leg_Simulation/
├── config/
│   └── robot_params.yaml          # 统一参数配置文件
├── mjcf/
│   ├── serial_leg.xml             # MuJoCo 五连杆并联腿模型
│   └── ramp_wedge.stl             # 17° 楔形斜坡 STL
├── sim/                           # 在线仿真模块（每控制步调用）
│   ├── main.py                    # 主入口，参数解析与仿真循环
│   ├── controller.py              # StandController: 6 阶段控制管线编排器（§8, §13）
│   ├── kinematics.py              # FiveLinkLeg: FK + IK + Jacobian（§2）
│   ├── vmc.py                     # VmcMapper: VMC 力矩映射 τ=JᵀF（§6）
│   ├── pid.py                     # PID 控制器（§8）
│   ├── kalman.py                  # KalmanOdometry: 2-D Kalman 滤波器（§7）
│   ├── lqr_governor.py            # LqrReferenceGovernor: 参考调速器（§4-5）
│   ├── ground_contact.py          # GroundContactDetector: 离地/卡死/打滑检测（§9）
│   ├── jump_state_machine.py      # JumpStateMachine: 跳跃状态机（§10）
│   ├── keyboard_reader.py         # 键盘输入读取
│   ├── utils.py                   # 工具函数 + ContinuousAngle
│   └── __init__.py                # 包初始化 + 公共 API 导出
├── data/                          # 预计算数据（离线生成，在线查表）
│   ├── lqr_gain_fit.py            # Chebyshev 2D 多项式系数 + get_K/get_e2/get_e3
│   └── k_table_2d.h              # C 头文件: 2D Chebyshev 增益表（MCU 用）
├── tools/                         # 离线计算（运行一次，生成 data/）
│   └── linearize.py               # 手册 §3-5, §12: 动力学线性化 → DARE → 2D 拟合 → 导出
├── 轮腿机器人控制理论手册.md       # 控制理论参考手册
└── README.md                      # 本文件
```

---

## 工具脚本

### 动力学线性化 + LQR 增益表生成

`tools/linearize.py` 使用 **SymPy 拉格朗日力学**推导五连杆轮腿机器人非线性动力学模型，
在多个腿长工作点线性化得到 A_d, B_d 矩阵，求解 DARE 得到 K 矩阵，拟合 2D 多项式系数，
导出到 `data/lqr_gain_fit.py`、`data/k_table_2d.h` 和 `data/AB_sampling_points.npz`。

```bash
# 生成 2D Chebyshev 增益表（14×14 网格）+ C 头文件
python tools/linearize.py

# 更细网格
python tools/linearize.py --grid-size 27
```

**输出文件：**

| 文件 | 用途 |
|------|------|
| `data/lqr_gain_fit.py` | 2D Chebyshev 系数 + get_K/get_e2/get_e3（仿真导入） |
| `data/k_table_2d.h` | 2D Chebyshev 增益表 C 头文件（MCU 移植） |

2D 增益表支持左右腿长不同的非对称工况（转弯、单腿台阶等），拟合精度 < 5% 相对 RMS 误差。

---

## 控制理论背景

本仿真的控制方案在 `轮腿机器人控制理论手册.md` 中有详尽的理论推导，核心要点：

### 10 维状态空间

LQR 控制的 10 维状态向量：

```
x = [x, yaw, pitch, θ_L, θ_R, ẋ, yaẇ, pitcḣ, θ̇_L, θ̇_R]ᵀ
```

其中 `θ_L`、`θ_R` 为左右腿相对于铅垂线的摆角。

### 4 维控制输出

```
u = [τ_L, τ_R, T_wL, T_wR]ᵀ
```

即左右虚拟腿力矩 + 左右轮端力矩。

### 控制管线

```
传感器读数 → 五连杆运动学 → (L, θ) → Kalman 状态估计 → 10 维状态
    ↓
2D Chebyshev 增益调度 K(L_l, L_r) + e2/e3(L_l, L_r)
    ↓
LQR u = -K·(x - x_ref) + PID 腿长力 + Roll 力
    ↓
VMC τ = JᵀF → 关节力矩
    ↓
离地检测 K 抑制 → 力矩限幅 → MuJoCo 执行
```

### 增益调度

2D Chebyshev 多项式 `K(L_l, L_r)` 将腿长变化纳入增益计算（手册 §5.6）。
40 个 K 元素各自为 28 项 Chebyshev 基函数的线性组合，运行时递推求值。

### 着地检测

`GroundContactDetector`（手册 §9）提供三级检测：离地（Fn 滞后+去抖）、
卡腿（力矩饱和+轮速为零）、打滑（轮速与车速失配）。
检测结果抑制 K 矩阵对应行，防止悬空/卡死腿失控。

---

## 实机部署参考

仿真代码考虑了向嵌入式 MCU 移植的需求：

1. **LQR 增益表**通过 `python tools/linearize.py` 导出为 C 头文件 `data/k_table_2d.h`，含 Chebyshev 递推求值函数 `k_table_2d_eval(L_l, L_r, k_out)`
2. **PID 参数**、**Kalman 噪声参数**、**力限幅值**均在 YAML 配置文件中统一管理
3. **UDP 遥测**：仿真运行时通过 `127.0.0.1:12345` 发送 24 个 float（10 状态 + 10 目标 + 2 腿长 + 2 法向力），可直接用 VOFA+ 上位机实时观察
4. 控制频率 **500Hz**（dt = 0.002s）

---

## 常见使用示例

```bash
# 基础键盘遥控
python sim/main.py

# LQR 定点自平衡
python sim/main.py --control-mode lqr

# 抗扰测试：侧向推力
python sim/main.py --control-mode lqr --push-time 4.0 --push-duration 0.2 --push-force-y 80

# 高速巡航测试
python sim/main.py --max-keyboard-speed 2.5 --max-keyboard-yaw-rate 8.0

# 自定义初始状态 + 无头模式
python sim/main.py --headless --duration 5 --initial-pitch 0.15 --control-mode lqr
```
