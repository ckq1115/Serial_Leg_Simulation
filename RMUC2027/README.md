# RMUC 2027 纯场地图

## 查看真实碰撞与本次修正

在 MuJoCo 原生查看器中打开场地后，通过 Rendering 的 geom 分组显示碰撞体。

- `collision_overview.png`：场地碰撞总览及六洞编号。
- `collision_clearances.png`：六洞修正前后净高剖面。
- `collision_geometry.json`：测量数值与坐标修正记录。

原模型两半基础地面分别倾斜约 1.5°，已按各自刚体坐标反向校平到 Z=0；外观、地形、洞顶、舱体及墙面同步处理，坡道与台阶保留。原高度场最低值还把低处地面截平了，已补回基础面及薄地面条带。

六洞最小通行净高均为 250 mm。A 组洞下保留原有斜坡，洞内其余位置更高，顶板已计入坡面与高度场边界的亚毫米偏差。B/C 组约 1 mm 的薄地面条带计入净高。净高在同一 XY 位置成对测量，不把两侧不同位置的高低差混算。

当前 `scene_rmuc.xml` 和默认运行的 `scene_rmuc_tiled.xml` 均已修正。`assets/levelled/` 为修正后的几何；`assets/` 下原 OBJ 和 `heightfield.bin` 是保留的源数据，用于重建和修正前对照。

```text
RMUC2027/
  scene_rmuc.xml   场地入口
  scene_rmuc_tiled.xml  分块高度场入口
  assets/          彩色外观网格、地形高度场、墙面和隧道碰撞网格
  README.md
```

这是静态场地图：机器人数量为 0，关节数量为 0，执行器数量为 0。
颜色来自 STEP 转换得到的材质分组，沿用原构建的显示配色。

本目录可单独复制使用。与机器人组合运行时，将 `GuodongRobot/` 放在本目录旁边，由 `GuodongRobot/run_field.bat` 启动；组合入口为 `GuodongRobot/scene_guodong_rmuc.xml`。

## 单独打开

安装了 MuJoCo 的 Python 环境即可加载，无需安装 rmuc-mujoco 或 CAD 工具：

```python
import mujoco

model = mujoco.MjModel.from_xml_path("RMUC2027/scene_rmuc.xml")
data = mujoco.MjData(model)
```

从包含 `RMUC2027` 文件夹的目录预览：

```powershell
python -m mujoco.viewer --mjcf .\RMUC2027\scene_rmuc.xml
```

## 导入其他 MuJoCo 工程

把整个 `RMUC2027` 文件夹复制到主 XML 所在目录。在主 XML 的 `<mujoco>`
内部加入一行，并将原来的场地或默认地面替换掉：

```xml
<include file="RMUC2027/scene_rmuc.xml"/>
```

相对路径随文件夹一起移动；无需修改 XML 内部路径。主工程继续管理自己的
机器人、控制器、时间步和求解器。MuJoCo 3.6.0 和 3.17.0 已验证直接加载及
跨目录 include 加载。

## 场地与碰撞

- 单位为米，Z 轴向上；沿用原构建坐标，`(0, 0, 0)` 位于一处平坦地面。
- 41 组彩色网格用于显示，保留完整外观。
- 1605 × 2805、间距 1 cm 的高度场用于地形碰撞。
- 保留四侧围挡、2 组墙面碰撞、6 组隧道顶板碰撞。
- 高度场保存为 MuJoCo 原生 float32 二进制文件，直接读取即可恢复原始运行时
  高度样本，无需 Python 注入高度数据。
- 地形碰撞组为 3，外观组为 1/2；碰撞参数沿用已构建的场地。

高度场是采样地形，围挡是静态近似；本次整理保留原构建的 `DRAFT_BLOCKED`
物理验证标记。比赛结构的运动和规则逻辑由使用它的仿真工程实现。

来源为本地提供的 `RMUC2027_V1.0.0.stp`，SHA-256：
`a5ad3ede3b7674fc98f2e73b71717216383cb2b4b5dcdbe900c2038918bea8e0`。
官方模型及衍生资源的权利归原权利人所有。历史构建资料单独保存在场地文件夹外。

## 两侧发射舱碰撞修正

两个舱体均简化为前方开口的空心盒子，每侧只有 5 个 box 碰撞体：
底板、顶板、两侧壁、后壁。舱内为空，发射方向不设置封口碰撞体。
两侧绕场地中心旋转 180° 对称放置，舱体随对应半场同步校平。
这是一套简化碰撞外壳，内部机构和外部细节只保留外观。

舱体区域误填的高度场已清空至实际底部地形，原有六处通道的碰撞、
其余地形保留相对半场的形状和彩色外观。修正后的场景无需任何额外运行时转换。
