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
│  │ KalmanOdometry│  LQR Governor │  FiveLinkLeg  │ │
│  │  (状态估计)   │ (键盘→目标值) │  (五连杆运动学) │ │
│  ├─────────────┼──────────────┼────────────────┤ │
│  │  ChangeLengthFit│   PID ×5   │  VMC 力矩映射  │ │
│  │  (增益插值)    │ (腿长/横滚) │ (力→关节力矩)  │ │
│  └─────────────┴──────────────┴────────────────┘ │
├──────────────────────────────────────────────────┤
│              MuJoCo 物理引擎                       │
│         mjcf/serial_leg.xml (机器人模型)           │
└──────────────────────────────────────────────────┘
```

控制流程简述：

1. 从 MuJoCo 读取传感器数据（IMU 四元数/角速度/加速度、关节位置/速度、轮速等）
2. Kalman 滤波器融合 IMU 加速度与轮式里程计，估计机身 x 方向位置与速度
3. 五连杆运动学正解，将关节角转换为等效腿长 `L` 和摆角 `θ`
4. 根据当前腿长插值 LQR 增益矩阵 `K(L_left, L_right)`
5. 键盘模式：参考调节器（Reference Governor）将键盘输入映射为平滑的 LQR 目标轨迹
6. LQR 计算虚拟腿力矩 + 轮端力矩；PID 计算腿长力和横滚力
7. VMC 将虚拟腿力/力矩映射为前后髋关节力矩
8. 写入 MuJoCo `data.ctrl`，步进仿真
9. 通过 UDP 发送状态数据（供外部工具如 VOFA+ 可视化）

---

## 环境要求

| 依赖 | 说明 |
|------|------|
| Python 3.9+ | 建议使用 conda/venv 虚拟环境 |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | 物理引擎（`pip install mujoco`） |
| NumPy | 数值计算 |
| SciPy | 线性化工具需要（`fsolve`, `expm`, `least_squares` 等） |
| PyYAML | 读取配置文件 |
| SymPy | 符号动力学推导（仅 `derive_linear_model.py` 需要） |
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

- 速度/偏航指令以配置的加速度平滑斜坡变化（`speed_accel_mps2` / `yaw_rate_accel_radps2`）
- 松开按键时以释放减速度（release accel）快速停止
- 支持 `instant_stop_on_release` 瞬时刹车

键盘指令转换为 LQR 的目标状态（x、yaw 及其导数），LQR 控制器跟踪该目标。

### LQR 自主平衡模式

（`--control-mode lqr`）机器人维持在初始位置和姿态。可通过命令行参数设置目标偏置：

```bash
python sim/main.py --control-mode lqr --target-x 1.0 --target-x-dot 0.5
```

让机器人以 0.5 m/s 速度向前移动 1 米后停下。

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
| `--lqr-gain-table` | str | 配置文件指定 | LQR 增益表文件路径 |
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
| `control_initial` | LQR 增益表路径、PID 参数、目标偏置、键盘指令参数、Kalman 噪声参数 |

### 关键参数

```yaml
control_initial:
  lqr_gain_table: data/lqr_gain_table_analytic.txt  # 增益表文件
  kalman:
    q_pos: 0.0001    # 位置过程噪声
    q_vel: 0.01      # 速度过程噪声
    r_pos: 1.0       # 里程计位置测量噪声
    r_vel: 0.5       # 里程计速度测量噪声
```

---

## 项目结构

```
Serial_Leg_Simulation/
├── config/
│   └── robot_params.yaml          # 统一参数配置文件
├── mjcf/
│   └── serial_leg.xml             # MuJoCo 机器人模型（XML）
├── sim/                           # 仿真核心代码
│   ├── main.py                    # 主入口，参数解析与仿真循环
│   ├── controller.py              # StandController：控制主逻辑
│   ├── leg.py                     # FiveLinkLeg：五连杆运动学 + VMC 力矩映射
│   ├── lqr_governor.py            # LqrReferenceGovernor：键盘→LQR目标转换
│   ├── change_length_fit.py       # ChangeLengthFit：增益多项式插值 + 平衡偏置
│   ├── kalman.py                  # KalmanOdometry：2 维 Kalman 位置/速度估计
│   ├── pid.py                     # PID 控制器
│   ├── keyboard_reader.py         # pynput 键盘读取（支持长按）
│   └── utils.py                   # 工具函数（配置加载、四元数→欧拉角等）
├── data/
│   ├── lqr_gain_table_analytic.txt # 解析 LQR 增益表（1D 对称，27 点）
│   ├── k_table_2d.h               # C 头文件增益表（6 阶 2D 多项式，MCU 用）
│   └── change_length_fit_analytic.py # 1D 多项式拟合系数（仿真用）
├── tools/
│   └── derive_linear_model.py     # 离线工具：SymPy 拉格朗日建模 + LQR 增益表生成
├── 轮腿机器人控制理论手册.md       # 控制理论参考手册
└── README.md                      # 本文件
```

---

## 工具脚本

### LQR 增益表生成

`tools/derive_linear_model.py` 是整个控制方案的"模型源头"——它使用 **SymPy 解析拉格朗日力学**推导五连杆轮腿机器人的非线性动力学模型，在多个腿长工作点进行线性化，并通过离散代数 Riccati 方程（DARE）求解 LQR 最优增益矩阵。

```bash
# 生成 1D 对称表（27 个腿长采样点）+ 4 阶多项式拟合
python tools/derive_linear_model.py --mode 1d

# 生成 2D 非对称表（14×14 网格）+ 6 阶 2D 多项式拟合 + C 头文件
python tools/derive_linear_model.py --mode 2d --grid-size 14

# 全部生成（1D + 2D）
python tools/derive_linear_model.py --mode both
```

**输出文件：**

| 文件 | 用途 |
|------|------|
| `data/lqr_gain_table_analytic.txt` | 1D 对称增益表，仿真中通过线性插值查表 |
| `data/change_length_fit_analytic.py` | 1D 4 阶多项式拟合系数（Python 类） |
| `data/k_table_2d.h` | 2D 6 阶多项式增益表 C 头文件，可直接用于 MCU |

**2D 增益表**支持左右腿长不同的非对称工况（如转弯、侧倾时），拟合精度通常 < 5% 相对 RMS 误差。

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
姿态/关节传感器 → 五连杆正运动学(L,θ) → Kalman 状态估计
    → 增益插值 K(L) → LQR u = -K(x - x_ref)
    → PID 腿长力 + Roll 力
    → VMC 力→关节力矩映射
    → 力矩限幅 → MuJoCo 执行器
```

### 增益调度

由于腿长变化会显著改变系统动力学，LQR 增益矩阵 `K` 需要在不同腿长下重新计算。仿真中通过预计算的增益表进行线性/多项式插值。

### 着地检测

通过估算轮端法向接触力 `Fn` 判断轮子是否着地。若 `Fn < 20N`，判定该腿悬空，将其 LQR 虚拟力矩通道清零并翻转摆角偏置，防止悬空腿失控。

---

## 实机部署参考

仿真代码考虑了向嵌入式 MCU 移植的需求：

1. **LQR 增益表**可通过 `derive_linear_model.py --mode 2d` 导出为 C 头文件 `data/k_table_2d.h`，包含完整的运行时插值函数 `k_table_2d_eval(L_l, L_r, k_out)`
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

# 使用自定义增益表
python sim/main.py --lqr-gain-table data/lqr_gain_table_analytic.txt --control-mode lqr
```
