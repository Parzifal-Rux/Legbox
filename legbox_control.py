# -*- coding: utf-8 -*-
"""
Legbox 轮足机器人控制脚本
使用方法：
  1. 在 Isaac Sim 中打开 Legbox.usda
  2. 确保场景中有地面（Ground Plane 或 Collision Plane）
  3. 点击 Play（▶）启动物理仿真
  4. Window → Script Editor，打开此文件
  5. 点击 Execute 运行
"""

import numpy as np
import asyncio
import carb
from omni.isaac.core.articulations import Articulation


async def run_legbox():
    """主控制逻辑，用 async 包装确保 await 正常工作"""

    # === 1. 获取机器人 ===
    robot_path = "/World/Legbox"

    try:
        robot = Articulation(robot_path)
        # 必须调用 initialize，否则 dof_names 等属性可能为空
        robot.initialize()
    except Exception as e:
        print(f"[ERROR] 无法获取机器人: {e}")
        print("请确认：")
        print("  1. 场景已加载 Legbox.usda")
        print("  2. 已点击 Play 启动物理仿真")
        print(f"  3. 路径 {robot_path} 存在且有 ArticulationRoot")
        return

    # === 2. 打印 DOF 信息，确认关节顺序 ===
    dof_names = robot.dof_names
    num_dof = robot.num_dof
    print(f"[OK] 机器人已连接: {robot_path}")
    print(f"  DOF 数量: {num_dof}")
    print(f"  DOF 名称: {dof_names}")

    if num_dof < 10:
        print(f"[ERROR] DOF 数量 {num_dof} < 10，关节驱动可能未正确加载")
        return

    # === 3. 根据名称动态匹配关节索引 ===
    # 不要硬编码索引，而是根据名称查找
    leg_names = ["LL1", "LL3", "LL2", "LL4", "LR1", "LR3", "LR2", "LR4"]
    wheel_names = ["WL", "WR"]

    leg_indices = []
    for name in leg_names:
        found = False
        for i, dof in enumerate(dof_names):
            if name == dof:
                leg_indices.append(i)
                found = True
                break
        if not found:
            print(f"[WARN] 找不到腿部关节 '{name}'，DOF 名称: {dof_names}")
            return

    wheel_indices = []
    for name in wheel_names:
        found = False
        for i, dof in enumerate(dof_names):
            if name == dof:
                wheel_indices.append(i)
                found = True
                break
        if not found:
            print(f"[WARN] 找不到轮子关节 '{name}'，DOF 名称: {dof_names}")
            return

    print(f"  腿部关节索引: {leg_indices}")
    print(f"  轮子关节索引: {wheel_indices}")

    # === 4. 初始站立姿态 ===
    leg_positions = np.zeros(len(leg_indices))
    robot.set_joint_position_targets(leg_positions, joint_indices=leg_indices)
    print("[STEP] 腿部关节归零（站立姿态）")

    # 等待物理稳定
    await asyncio.sleep(1.0)

    # === 5. 转动轮子前进 ===
    wheel_speed = 10.0  # rad/s
    wheel_velocities = np.array([wheel_speed, wheel_speed])
    robot.set_joint_velocity_targets(wheel_velocities, joint_indices=wheel_indices)
    print(f"[STEP] 轮子开始转动: {wheel_speed} rad/s (前进)")

    # === 6. 持续运行并打印状态 ===
    for i in range(50):
        await asyncio.sleep(0.1)
        try:
            vel = robot.get_joint_velocities()
            pos = robot.get_joint_positions()
            root_pos = robot.get_world_pose()
            # get_world_pose 返回 (positions, orientations)
            # positions 形状可能是 (3,) 或 (1, 3)
            if hasattr(root_pos, '__len__') and len(root_pos) >= 1:
                if hasattr(root_pos[0], '__len__'):
                    x_pos = float(root_pos[0][0])
                else:
                    x_pos = float(root_pos[0])
            else:
                x_pos = 0.0

            if i % 10 == 0:
                wl_vel = float(vel[wheel_indices[0]]) if len(vel) > wheel_indices[0] else 0.0
                wr_vel = float(vel[wheel_indices[1]]) if len(vel) > wheel_indices[1] else 0.0
                print(f"  [{i*0.1:.1f}s] 位置 x={x_pos:.3f}, "
                      f"轮速: WL={wl_vel:.2f}, WR={wr_vel:.2f} rad/s")
        except Exception as e:
            if i % 10 == 0:
                print(f"  [{i*0.1:.1f}s] 状态读取异常: {e}")

    # === 7. 停止 ===
    robot.set_joint_velocity_targets(np.array([0.0, 0.0]), joint_indices=wheel_indices)
    print("[DONE] 轮子停止")


# === 入口：启动异步任务 ===
# Isaac Sim Script Editor 支持顶层 await，但用 ensure_future 更可靠
asyncio.ensure_future(run_legbox())


# === 附加：腿部动作测试（可选）===
# 取消注释来测试蹲起

# async def test_legs():
#     import asyncio
#     from omni.isaac.core.articulations import Articulation
#     import numpy as np
#
#     robot = Articulation("/World/Legbox")
#     robot.initialize()
#     dof_names = robot.dof_names
#
#     leg_names = ["LL1", "LL3", "LL2", "LL4", "LR1", "LR3", "LR2", "LR4"]
#     leg_indices = [dof_names.index(n) for n in leg_names if n in dof_names]
#
#     # 蹲下
#     squat = np.array([0.3, -0.6, 0.3, -0.6, 0.3, -0.6, 0.3, -0.6])
#     robot.set_joint_position_targets(squat, joint_indices=leg_indices)
#     await asyncio.sleep(2.0)
#
#     # 站起
#     stand = np.zeros(8)
#     robot.set_joint_position_targets(stand, joint_indices=leg_indices)
#     await asyncio.sleep(2.0)
#     print("腿部动作测试完成")
#
# # asyncio.ensure_future(test_legs())
