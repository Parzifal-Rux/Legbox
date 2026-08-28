# -*- coding: utf-8 -*-
"""
Legbox 站立仿真 —— 从零开始 v1（Python 单源真相 SSoT 版）

核心设计
--------
1. 运行时参数只在这一个脚本里定义一次（ stiffness / damping / maxForce / 限位 / 姿态 / 初始高度 ）
2. 不靠 asyncio.sleep、不靠"关节位置变没变"猜 Play 状态，而用 Isaac 官方 SimulationContext
   + 当前 timeline 状态 + 仿真时间戳自增 做双重确认
3. 每次执行都 **强制重置到完全一致的初始状态**：
       机器人底盘 (0, 0, H0) + 零速度 + 关节归零折叠
   这样"每次落地角度不一样"问题理论上会消失
4. 并联 LL2/LL4/LR2/LR4 用"中等软约束"（不是 0 也不是 80000），让五连杆几何稳定
   跟随主动链，不抢控制权；WL/WR 只给少量阻尼，不锁死


操作顺序（Isaac Sim GUI 中）
---------------------------
1. File → Open 打开 Legbox.usda（确保 Legbox → references = @./Legbox/Legbox.usd@ 已解析）
2. Window → Script Editor → 打开本文件 → 点 Execute
3. 控制台打印出 "单源参数已生效"、"初始状态已写入"、"请点 Play ▶" 之后，再点 Play
4. 观察三阶段起身 + 保持站立 10s

Trouble Shooting
----------------
- 如果立刻报 `Physics articulation view has not been populated yet`：
  → 通常是 Play 还没跑完第 1 个物理步。本脚本最多等 30s，看到 `等待仿真启动…` 直接点 Play 即可
- 如果机器人 Play 瞬间爆飞：
  → 先看控制台 `初始状态写入` 时的高度 H0，改成略高/略低；再把 STIFFNESS 降到 600
- 如果并联杆抖：把 `PASSIVE_*` 三参数降 30%
"""

import os
import math
import numpy as np

from pxr import Usd, Sdf
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.prims import is_prim_path_valid
import omni.timeline
import asyncio


# =============================================================================
# 一、单源真相区（唯一能改参数的地方，从这里往下所有数值都从本块读）
# =============================================================================

# --- 1. 路径与命名 ---
USDA_ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "Legbox.usda"))
ROBOT_PRIM_PATH = "/World/Legbox"
ACTIVE_JOINTS  = ["LL1", "LL3", "LR1", "LR3"]   # 4 个主动
PASSIVE_JOINTS = ["LL2", "LL4", "LR2", "LR4"]   # 4 个并联被动
WHEEL_JOINTS   = ["WL", "WR"]                   # 2 个轮
# 注：WLL / WRR 闭环约束不进 DOF，忽略

# --- 2. 驱动参数（Project Memory 91 Nm 匹配法）---
# 主动关节：假设允许 0.1 rad 误差下出满力 → Kp ≈ 91 / 0.1 = 910
#           期望约 0.7 阻尼比 → Kd ≈ 910 * 0.3 = 273
ACTIVE_STIFFNESS = 910.0
ACTIVE_DAMPING   = 273.0
ACTIVE_MAX_FORCE = 91.0

# 并联被动：软约束跟随，只承担"保持形状"；约主动 1/5
PASSIVE_STIFFNESS = 180.0
PASSIVE_DAMPING   = 180.0
PASSIVE_MAX_FORCE = 22.0

# 轮：不锁位置，只加很小的阻尼（自平衡/控制时会再写 0）
WHEEL_STIFFNESS = 0.0
WHEEL_DAMPING   = 5.0
WHEEL_MAX_FORCE = 0.0

# --- 3. 初始状态（每次重置到一模一样）---
INIT_ROOT_TRANSLATE = np.array([0.0, 0.0, 0.22], dtype=np.float64)   # 底盘高度 z = 22cm
INIT_ROOT_ORIENT    = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) # wxyz 无旋转

# 3 阶段关节角度表（单位 rad）。右腿取负是为了"外展方向镜像"。
#   折叠(瘫) → 预备 → 站立
POISES = {
    # 瘫倒地面：腿收起来
    "COLLAPSED": {"LL1":  0.0, "LL3":  0.0, "LR1":  0.0, "LR3":  0.0},
    # 预备：腿摆到身体正下方，但还不撑高
    "PREP":      {"LL1":  0.3, "LL3":  0.5, "LR1": -0.3, "LR3": -0.5},
    # 站立：把身体顶起来
    "STAND":     {"LL1":  0.5, "LL3":  0.8, "LR1": -0.5, "LR3": -0.8},
}

# --- 4. 时间线 ---
T_PREP    = 2.0   # s: 瘫 → 预备
T_STAND   = 3.0   # s: 预备 → 站立
T_HOLD    = 10.0  # s: 站立保持
CTRL_RATE = 50.0  # Hz: 控制步频率（实际跟随物理步）


# =============================================================================
# 二、工具函数
# =============================================================================

def set_usd_attr(prim, attr_path, value, type_name):
    """确保 prim 上 attr_path 存在并 = value；不存在就按 type_name 创建"""
    if prim is None or not prim.IsValid():
        return False
    attr = prim.GetAttribute(attr_path)
    if attr.IsValid():
        attr.Set(value)
        return True
    new_attr = prim.CreateAttribute(attr_path, type_name)
    new_attr.Set(value)
    return True


def write_joint_drive_usd(joint_prim, Kp, Kd, Fmax, target=0.0, drive_type="force"):
    """把 USD 驱动 5 个参数一次性写到位（覆盖 usda 作者层 + 引用层）。
    兼容 Isaac Sim 多版本的关键：写 `drive:physics:*` 官方属性路径。
    """
    if joint_prim is None or not joint_prim.IsValid():
        return
    set_usd_attr(joint_prim, "drive:physics:stiffness",      float(Kp),   Sdf.ValueTypeNames.Float)
    set_usd_attr(joint_prim, "drive:physics:damping",        float(Kd),   Sdf.ValueTypeNames.Float)
    set_usd_attr(joint_prim, "drive:physics:maxForce",       float(Fmax), Sdf.ValueTypeNames.Float)
    set_usd_attr(joint_prim, "drive:physics:targetPosition", float(target), Sdf.ValueTypeNames.Float)
    set_usd_attr(joint_prim, "drive:physics:type",           drive_type,  Sdf.ValueTypeNames.Token)


def find_all_joints_under(root_prim):
    """递归返回 {joint_name: prim}；只挑 *TypeName 中含 Joint 不含 Body 的"""
    out = {}
    stack = [root_prim]
    while stack:
        p = stack.pop()
        tn = p.GetTypeName()
        if "Joint" in tn and "Body" not in tn:
            out[p.GetName()] = p
        for c in p.GetChildren():
            stack.append(c)
    return out


def lerp_pose(a, b, alpha):
    """按线性插值合并两份关节角度字典"""
    keys = set(a.keys()) | set(b.keys())
    return {k: (a.get(k, 0.0) * (1.0 - alpha) + b.get(k, 0.0) * alpha) for k in keys}


# =============================================================================
# 三、主体流程（async，为了 Script Editor 里能和 GUI 事件共存）
# =============================================================================

async def main():
    print("=" * 72)
    print(" Legbox 站立 —— 方案 A · Python 单源真相 v1")
    print("=" * 72)

    # 1. 拿 stage / timeline / sim context
    stage = get_current_stage()
    if stage is None:
        print("[FATAL] 还没加载 stage，先 Open Legbox.usda")
        return

    sim = SimulationContext.instance()
    tl  = omni.timeline.get_timeline_interface()

    # 2. 定位机器人 prim
    if not is_prim_path_valid(ROBOT_PRIM_PATH):
        print(f"[FATAL] 找不到 {ROBOT_PRIM_PATH} prim")
        return
    robot_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    joint_prims = find_all_joints_under(robot_prim)
    print(f"[1/5] 发现 {len(joint_prims)} 个关节 prim：{list(joint_prims.keys())}")

    # 3. 把单源参数写进 USD 属性（覆盖 usda / 引用层 / 二进制任何旧值）
    print("[2/5] 将单源驱动参数写入 USD ...")
    drive_spec = (
        ([j for j in ACTIVE_JOINTS  if j in joint_prims], ACTIVE_STIFFNESS,  ACTIVE_DAMPING,  ACTIVE_MAX_FORCE),
        ([j for j in PASSIVE_JOINTS if j in joint_prims], PASSIVE_STIFFNESS, PASSIVE_DAMPING, PASSIVE_MAX_FORCE),
        ([j for j in WHEEL_JOINTS   if j in joint_prims], WHEEL_STIFFNESS,   WHEEL_DAMPING,   WHEEL_MAX_FORCE),
    )
    for names, Kp, Kd, Fmax in drive_spec:
        for n in names:
            write_joint_drive_usd(joint_prims[n], Kp, Kd, Fmax, target=0.0, drive_type="force")
            print(f"   {n:>4s}:  Kp={Kp:>6.0f}  Kd={Kd:>6.0f}  Fmax={Fmax:>6.0f}  type=force")

    # 4. 初始化 Articulation；读 DOF 名；写运行时 set_joint_{stiffness,damping,max_efforts}（真·运行时 Tensor 层，不碰 USD）
    robot = Articulation(ROBOT_PRIM_PATH)
    # 注意：在还没 Play 时，Articulation.initialize() 会失败。我们先存对象，等启动成功第一帧后再 .initialize()
    robot_initialized = False

    def _lazy_ensure_robot_initialized():
        """Play 后第一次物理帧可用时才执行 initialize，再写入运行时驱动参数 + 复位"""
        nonlocal robot_initialized
        if robot_initialized:
            return True
        try:
            robot.initialize()
        except Exception as exc:
            return False
        # 运行时覆盖，与 USD 一致
        dof_names = list(robot.dof_names)
        _Kp = np.zeros(robot.num_dof, dtype=np.float32)
        _Kd = np.zeros(robot.num_dof, dtype=np.float32)
        _Fm = np.zeros(robot.num_dof, dtype=np.float32)
        for name, Kp, Kd, Fmax in (
            [(n, ACTIVE_STIFFNESS,  ACTIVE_DAMPING,  ACTIVE_MAX_FORCE)  for n in ACTIVE_JOINTS] +
            [(n, PASSIVE_STIFFNESS, PASSIVE_DAMPING, PASSIVE_MAX_FORCE) for n in PASSIVE_JOINTS] +
            [(n, WHEEL_STIFFNESS,   WHEEL_DAMPING,   WHEEL_MAX_FORCE)   for n in WHEEL_JOINTS]
        ):
            if name in dof_names:
                idx = dof_names.index(name)
                _Kp[idx] = Kp; _Kd[idx] = Kd; _Fm[idx] = Fmax
        try:
            robot.set_joint_stiffnesses(_Kp)
            robot.set_joint_dampings(_Kd)
            robot.set_joint_max_efforts(_Fm)
        except Exception as exc:
            print(f"[WARN] 运行时 set_joint_* 失败: {exc}; USD 属性仍应生效")

        # 强制复位初始状态（高度、姿态、速度、关节角）
        try:
            robot.set_world_pose(INIT_ROOT_TRANSLATE, INIT_ROOT_ORIENT)
            robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
            robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
            q0 = np.zeros(robot.num_dof, dtype=np.float32)
            for n, v in POISES["COLLAPSED"].items():
                if n in dof_names: q0[dof_names.index(n)] = v
            robot.set_joint_positions(q0)
            # 立刻把 COLLAPSED target 写 USD（与 set_joint_positions 同步）
            for n, v in POISES["COLLAPSED"].items():
                if n in joint_prims:
                    set_usd_attr(joint_prims[n], "drive:physics:targetPosition", float(v), Sdf.ValueTypeNames.Float)
        except Exception as exc:
            print(f"[WARN] 强制复位失败: {exc}")

        print(f"[3/5] Articulation 运行时就绪（DOF={robot.num_dof}）: {dof_names}")
        print(f"      底盘重置到 z={INIT_ROOT_TRANSLATE[2]:.3f}m + 关节折叠 + 速度清零")
        robot_initialized = True
        return True

    # 5. 等待仿真启动：timeline.is_playing() + sim.current_time 自增
    print("[4/5] 等待 Play ▶ ...")
    start_t = None
    prev_t  = None
    streak_playing = 0.0
    streak_stable  = 0.0
    waited = 0.0
    WAIT_DT = 0.1
    TIMEOUT = 60.0
    while waited < TIMEOUT:
        await asyncio.sleep(WAIT_DT)
        waited += WAIT_DT

        playing = bool(tl.is_playing())
        cur = None
        t_adv = False
        try: cur = float(sim.current_time)
        except Exception: cur = None
        if cur is not None and prev_t is not None and cur > prev_t + 1e-5:
            t_adv = True
        prev_t = cur

        # 一旦播放，马上尝试懒初始化 articulation
        if playing:
            streak_playing += WAIT_DT
            _ok = _lazy_ensure_robot_initialized() if not robot_initialized else True
            if _ok and t_adv:
                streak_stable += WAIT_DT
            else:
                streak_stable = 0.0
        else:
            streak_playing = 0.0
            streak_stable = 0.0

        if int(waited * 2) != int((waited - WAIT_DT) * 2):
            # 约每秒一次
            print(f"   {waited:5.1f}s  play={int(playing)}  sim_t={cur if cur is None else f'{cur:.3f}s':>7s}  "
                  f"t_adv={int(t_adv)}  streak_stable={streak_stable:.1f}s")

        if streak_stable >= 1.0 and robot_initialized:
            print(f"[OK] 仿真确认启动 streak={streak_stable:.1f}s，sim_t={cur:.3f}s —— 开始控制")
            break
    else:
        print(f"[ABORT] 等了 {waited:.0f}s 还没稳定启动，结束。请先 Stop 再重试。")
        return

    # 6. 现在有了确定的 dof_names / 起始时间，执行三阶段
    dof_names = list(robot.dof_names)
    num_dof   = robot.num_dof

    # 启动时再一次硬复位（保险：Play 第一帧可能物理引擎把落地速度放进来）
    q0 = np.zeros(num_dof, dtype=np.float32)
    for n, v in POISES["COLLAPSED"].items():
        if n in dof_names: q0[dof_names.index(n)] = v
    robot.set_joint_positions(q0)
    robot.set_joint_velocities(np.zeros(num_dof, dtype=np.float32))
    robot.set_world_pose(INIT_ROOT_TRANSLATE, INIT_ROOT_ORIENT)
    robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
    robot.set_angular_velocity(np.zeros(3, dtype=np.float32))

    # 构造 USD 属性句柄缓存（targetPosition 每步直接写 USD）
    usd_target_handles = {}
    for n in ACTIVE_JOINTS + PASSIVE_JOINTS + WHEEL_JOINTS:
        if n in joint_prims:
            a = joint_prims[n].GetAttribute("drive:physics:targetPosition")
            if not a.IsValid():
                a = joint_prims[n].CreateAttribute("drive:physics:targetPosition", Sdf.ValueTypeNames.Float)
            usd_target_handles[n] = a

    def _apply_active_targets(pose_dict, passive_pose_dict=None):
        """写主动关节 targetPosition；可选写并联被动软约束；轮子不动。"""
        for n, v in pose_dict.items():
            h = usd_target_handles.get(n)
            if h is not None: h.Set(float(v))
        if passive_pose_dict is not None:
            for n, v in passive_pose_dict.items():
                h = usd_target_handles.get(n)
                if h is not None: h.Set(float(v))

    # 先上 COLLAPSED 目标
    _apply_active_targets(POISES["COLLAPSED"])

    # 读控制步长（按物理 dt 算 ctrl_dt）
    try:
        physics_dt = float(sim.get_physics_dt())
    except Exception:
        physics_dt = 1.0 / 240.0
    ctrl_dt = max(physics_dt, 1.0 / CTRL_RATE)
    ctrl_steps_per_ctrl = max(1, int(round(ctrl_dt / physics_dt)))

    print(f"[5/5] 物理步长 dt={physics_dt*1000:.2f}ms，控制每 {ctrl_steps_per_ctrl} 步一次")

    def _run_interpolated(tag, start_pose, end_pose, duration):
        """同步推进 duration 秒，控制频率按 ctrl_steps_per_ctrl，期间每 0.5s 打印状态"""
        print(f"\n>>> 阶段[{tag}]  用时 {duration:.1f}s  start→end")
        n_steps = int(round(duration / physics_dt))
        log_every = int(max(1, round(0.5 / physics_dt)))
        for i in range(n_steps + 1):
            alpha = 0.0 if duration == 0 else (i * physics_dt / duration)
            alpha = min(1.0, max(0.0, alpha))
            if i % ctrl_steps_per_ctrl == 0 or i == n_steps:
                pose = lerp_pose(start_pose, end_pose, alpha)
                _apply_active_targets(pose)
            sim.step(render=True)
            if log_every and i % log_every == 0:
                _print_snapshot(dof_names, f"{tag}@{i*physics_dt:5.2f}s")

    def _run_hold(duration):
        """保持站立 POSE_STAND，被动关节保持 0 目标（软约束靠闭环跟）"""
        print(f"\n>>> [HOLD] 保持站立 {duration:.1f}s")
        n_steps = int(round(duration / physics_dt))
        log_every = int(max(1, round(1.0 / physics_dt)))
        pose = POISES["STAND"]
        for i in range(n_steps + 1):
            if i % ctrl_steps_per_ctrl == 0 or i == n_steps:
                _apply_active_targets(pose)
            sim.step(render=True)
            if log_every and i % log_every == 0:
                _print_snapshot(dof_names, f"HOLD@{i*physics_dt:5.2f}s", pose_target=pose)

    def _print_snapshot(dof_names, tag, pose_target=None):
        q = robot.get_joint_positions()
        root_pos, _ = robot.get_world_pose()
        line = f"[{tag}] z={root_pos[2]:.4f}m"
        for n in ACTIVE_JOINTS:
            if n in dof_names and q is not None:
                deg = float(np.degrees(q[dof_names.index(n)]))
                mark = ""
                if pose_target and n in pose_target:
                    tdeg = float(np.degrees(pose_target[n]))
                    mark = f"(Δ{deg-tdeg:+.1f}°)"
                line += f"  {n}={deg:+.1f}°{mark}"
        print(line)

    # 进入控制主循环
    _print_snapshot(dof_names, "START")
    _run_interpolated("COLL→PREP ", POISES["COLLAPSED"], POISES["PREP"],  T_PREP)
    _run_interpolated("PREP →STAND", POISES["PREP"],      POISES["STAND"], T_STAND)
    _run_hold(T_HOLD)

    # 最终诊断
    print("\n" + "=" * 72)
    print(" 最终结果")
    print("=" * 72)
    q = robot.get_joint_positions()
    root_pos, _ = robot.get_world_pose()
    print(f"底盘 z = {root_pos[2]:.4f} m")
    max_err_deg = 0.0
    for n in ACTIVE_JOINTS:
        idx = dof_names.index(n)
        actual_deg = float(np.degrees(q[idx]))
        target_deg = float(np.degrees(POISES["STAND"].get(n, 0.0)))
        err = abs(actual_deg - target_deg)
        max_err_deg = max(max_err_deg, err)
        print(f"  {n:>4s}: 实际 {actual_deg:+6.1f}°  目标 {target_deg:+6.1f}°  误差 {err:.1f}°")
    print(f"主动关节最大角度误差: {max_err_deg:.2f}°")
    if root_pos[2] > 0.12 and max_err_deg < 10.0:
        print("[结论] ✅ 基本站起来了；并联杆几何 & 闭环约束仍工作")
    else:
        print("[结论] 🧪 高度/误差未达标；下一步可提高刚度/降低目标/加并联软约束")
    print("脚本 end。")


# =============================================================================
# 入口：在 Isaac Script Editor 中 Execute，asyncio.ensure_future 会挂起
# =============================================================================
asyncio.ensure_future(main())