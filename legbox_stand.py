# -*- coding: utf-8 -*-
"""
Legbox 站立测试脚本 v3
- 修复: PhysxSchema 中没有 PhysxJointDriveAPI
- 改用直接设置 USD 属性的方式（兼容所有版本）

在 Isaac Sim Script Editor 中运行：
1. 先点 Play 启动物理
2. 打开 Window > Script Editor
3. File > Open 选择本文件
4. 点 Execute
"""

import numpy as np
from pxr import Usd, Gf
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import get_current_stage

robot_path = "/World/Legbox"
stage = get_current_stage()

# =============== 配置区 ===============
STAND_POSE = {
    "LL1":  0.0,    # 左髋  (主动)
    "LL3":  0.0,    # 左膝  (主动)
    "LR1":  0.0,    # 右髋  (主动)
    "LR3":  0.0,    # 右膝  (主动)
    # LL2, LL4, LR2, LR4 是并联杆，不设目标，被动跟随
    "WL":   0.0,    # 左轮（速度控制）
    "WR":   0.0,    # 右轮
}

STIFFNESS = 5000.0   # 刚度（位置控制）
DAMPING   = 500.0    # 阻尼
MAX_EFFORT = 200.0   # 最大力矩（Nm）
# =====================================

# ---- 第一步：通过 USD 属性直接设置驱动参数 ----
print("[STEP] 设置关节驱动参数...")

joint_prim_paths = {
    "LL1": f"{robot_path}/LL1",
    "LL2": f"{robot_path}/LL2",
    "LL3": f"{robot_path}/LL3",
    "LL4": f"{robot_path}/LL4",
    "LR1": f"{robot_path}/LR1",
    "LR2": f"{robot_path}/LR2",
    "LR3": f"{robot_path}/LR3",
    "LR4": f"{robot_path}/LR4",
    "WL":  f"{robot_path}/WL",
    "WR":  f"{robot_path}/WR",
}

active_joints = ["LL1", "LL3", "LR1", "LR3"]
passive_joints = ["LL2", "LL4", "LR2", "LR4"]
wheel_joints = ["WL", "WR"]

def set_joint_drive(prim, stiffness, damping, max_force):
    """直接设置 USD 属性，不依赖 PhysxSchema 类名"""
    # 属性路径前缀（可能是 linearDrive 或 angularDrive，取决于关节类型）
    # 先试 linearDrive
    prefixes = [
        "physxJoint:linearDrive:physics:",
        "physxJoint:angularDrive:physics:",
        "physics:",
    ]

    set_any = False
    for prefix in prefixes:
        stiff_attr = prim.GetAttribute(prefix + "stiffness")
        if stiff_attr.IsValid():
            stiff_attr.Set(stiffness)
            damp_attr = prim.GetAttribute(prefix + "damping")
            if damp_attr.IsValid():
                damp_attr.Set(damping)
            force_attr = prim.GetAttribute(prefix + "maxForce")
            if force_attr.IsValid():
                force_attr.Set(max_force)
            set_any = True
            break

    if not set_any:
        # 打印所有可用属性帮助调试
        attrs = prim.GetAuthoredAttributeNames()
        drive_attrs = [a for a in attrs if 'stiffness' in a.lower() or 'damping' in a.lower() or 'drive' in a.lower()]
        print(f"    找不到驱动属性！关节相关属性: {drive_attrs}")
        return False
    return True

for name, path in joint_prim_paths.items():
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        print(f"  [WARN] 找不到关节 prim: {path}")
        continue

    if name in active_joints:
        ok = set_joint_drive(prim, STIFFNESS, DAMPING, MAX_EFFORT)
        if ok:
            print(f"  {name}: 主动驱动 stiffness={STIFFNESS}, damping={DAMPING}")
    elif name in passive_joints:
        ok = set_joint_drive(prim, 10.0, 50.0, MAX_EFFORT)
        if ok:
            print(f"  {name}: 被动并联杆 stiffness=10, damping=50")
    elif name in wheel_joints:
        ok = set_joint_drive(prim, 0.0, DAMPING, MAX_EFFORT)
        if ok:
            print(f"  {name}: 轮子速度模式 stiffness=0, damping={DAMPING}")

print("[OK] 驱动参数设置完成")

# ---- 第二步：获取 Articulation 并发送目标 ----
robot = Articulation(robot_path)
robot.initialize()

dof_names = list(robot.dof_names)
num_dof = robot.num_dof
print(f"[OK] 机器人已连接，DOF 数量: {num_dof}")
print(f"  DOF 名称: {dof_names}")

current_pos = robot.get_joint_positions()
print(f"  当前关节角度: {np.round(current_pos, 3).tolist()}")

# 构建目标位置
target_pos = current_pos.copy()
for i, name in enumerate(dof_names):
    if name in STAND_POSE:
        target_pos[i] = STAND_POSE[name]

print(f"  目标关节角度: {np.round(target_pos, 3).tolist()}")

robot.set_joint_position_targets(target_pos)
print("[STEP] 已发送站立位置目标")

# ---- 第三步：观察站立效果 ----
import asyncio

async def check_stand():
    print("[STEP] 等待 3 秒观察站立效果...")
    for t in range(30):
        await asyncio.sleep(0.1)
        if t % 10 == 0:
            pos = robot.get_joint_positions()
            root_pos, _ = robot.get_world_pose()
            errors = []
            for i, name in enumerate(dof_names):
                if name in active_joints:
                    err = abs(pos[i] - target_pos[i])
                    errors.append(err)
            avg_err = np.mean(errors) if errors else 0
            print(f"  [{t*0.1:.1f}s] 底盘高度 z={root_pos[2]:.3f}m, "
                  f"主动关节平均误差={avg_err:.4f} rad")

    pos = robot.get_joint_positions()
    root_pos, _ = robot.get_world_pose()
    print(f"\n=== 站立结果 ===")
    print(f"  底盘高度: {root_pos[2]:.4f} m")
    print(f"  底盘位置: x={root_pos[0]:.4f}, y={root_pos[1]:.4f}, z={root_pos[2]:.4f}")
    print(f"  关节实际角度 vs 目标:")
    for i, name in enumerate(dof_names):
        target_str = f"{target_pos[i]:.4f}" if name in STAND_POSE else "被动"
        print(f"    {name:4s}: {pos[i]:.4f} rad (目标: {target_str})")

    if root_pos[2] > 0.15:
        print("\n[成功] 机器人站起来了！")
    elif root_pos[2] > 0.05:
        print("\n[部分成功] 机器人抬起来了一点，但还不够高")
        print("  建议：增大 STIFFNESS 或给膝关节一个弯曲角度")
    else:
        print("\n[失败] 机器人没有站起来")
        print("  可能原因和解决办法：")
        print("  1. 刚度太小 → 把 STIFFNESS 调到 10000+")
        print("  2. 最大力矩不够 → 把 MAX_EFFORT 调到 500+")
        print("  3. 初始高度太低 → 在 .usda 里把 z 调高到 0.5")
        print("  4. 膝关节需要弯曲 → 给 LL3/LR3 设 -0.5 弧度")

asyncio.ensure_future(check_stand())
