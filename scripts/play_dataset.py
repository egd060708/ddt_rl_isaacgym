#!/usr/bin/env python3
"""
机器人数据集播放脚本（支持 d1 四足 / d1h 双足）

从 JSON 数据集中读取每帧：根位置、根四元数、关节位置（其余字段忽略）。
关节数量由当前任务的 cfg.env.num_actions（与仿真 DOF 一致）决定：d1=16，d1h=8。

数据集默认查找路径（按顺序）:
  resources/<robot>/datasets/<文件名>
  resources/<robot>/datasets/all/<文件名>

<robot> 由 --robot 指定，或由 --task 名称推断（任务名含 d1h 则为 d1h，否则为 d1）。

使用示例:
    python scripts/play_dataset.py --task=d1_flat_play --dataset d1_stand_up.txt
    python scripts/play_dataset.py --task=d1h_flat_play --robot d1h --dataset my_motion.txt
    python scripts/play_dataset.py --task=d1_flat_play --dataset all/d1_stand_up.txt
"""

import json
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Isaac Gym 必须在 torch 之前导入
from isaacgym import gymapi, gymtorch

import numpy as np
import torch
from configs import *
from modules import *
from utils import get_args, task_registry
from global_config import ROOT_DIR

# 与 AMP / 数据集约定一致
POS_SIZE = 3
ROT_SIZE = 4


def infer_robot_from_task(task_name: str) -> str:
    """根据任务名推断机器人资源目录: d1 或 d1h。"""
    t = (task_name or "").lower()
    if "d1h" in t:
        return "d1h"
    return "d1"


def resolve_dataset_path(robot: str, dataset_name: str):
    """
    解析数据集文件路径。dataset_name 可含子目录，如 all/d1_stand_up.txt。
    返回: (path_or_None, tried_paths)
    """
    base = os.path.join(ROOT_DIR, "resources", robot, "datasets")
    candidates = [os.path.join(base, dataset_name)]
    # 仅文件名时，再尝试 datasets/all/<name>（与 AMP 数据目录一致）
    if not dataset_name.startswith("all" + os.sep):
        candidates.append(os.path.join(base, "all", os.path.basename(dataset_name)))

    tried = []
    for p in candidates:
        tried.append(p)
        if os.path.isfile(p):
            return p, tried
    return None, tried


def load_dataset(dataset_path: str):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return data


def extract_pose_data(frame: list, joint_pos_size: int):
    """从帧数据中提取根位置、四元数、关节位置（长度 joint_pos_size）。"""
    need = POS_SIZE + ROT_SIZE + joint_pos_size
    if len(frame) < need:
        raise ValueError(
            f"帧长度过短: 需要至少 {need} 个数 (POS+ROT+关节), 实际 {len(frame)}"
        )
    position = np.array(frame[0:POS_SIZE])
    rotation = np.array(frame[POS_SIZE : POS_SIZE + ROT_SIZE])
    joint_positions = np.array(
        frame[POS_SIZE + ROT_SIZE : POS_SIZE + ROT_SIZE + joint_pos_size]
    )
    return position, rotation, joint_positions


def set_robot_pose(
    env,
    env_id: int,
    position: np.ndarray,
    rotation: np.ndarray,
    joint_positions: np.ndarray,
    num_dof: int,
):
    env.root_states[env_id, :3] = torch.tensor(
        position, dtype=torch.float32, device=env.device
    )
    env.root_states[env_id, 3:7] = torch.tensor(
        rotation, dtype=torch.float32, device=env.device
    )

    if len(joint_positions) != num_dof:
        raise ValueError(
            f"关节位置数量错误: 仿真 num_dof={num_dof}, 数据提供 {len(joint_positions)}"
        )

    env.dof_pos[env_id] = torch.tensor(
        joint_positions, dtype=torch.float32, device=env.device
    )
    env.dof_vel[env_id] = 0.0

    env_ids_int32 = torch.tensor([env_id], dtype=torch.int32, device=env.device)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        1,
    )
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        1,
    )


def play_dataset(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    joint_pos_size = int(getattr(env_cfg.env, "num_actions", None) or 16)
    robot = getattr(args, "robot", None) or infer_robot_from_task(args.task)

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    num_dof = int(env.num_dof)
    if joint_pos_size != num_dof:
        print(
            f"[提示] cfg.env.num_actions={joint_pos_size} 与仿真 num_dof={num_dof} 不一致，以 num_dof 为准。"
        )
        joint_pos_size = num_dof

    dataset_name = getattr(args, "dataset", "d1_stand_up.txt")
    dataset_path, tried_paths = resolve_dataset_path(robot, dataset_name)

    print(f"机器人资源: {robot}  |  关节数(播放用): {joint_pos_size}")
    print(f"正在加载数据集: {dataset_path if dataset_path else '(未找到)'}")

    if dataset_path is None or not os.path.exists(dataset_path):
        print("错误: 未找到数据集文件。")
        if tried_paths:
            print("已尝试路径:")
            for p in tried_paths:
                print(f"  - {p}")
        return

    data = load_dataset(dataset_path)
    frames = data["Frames"]
    frame_duration = data.get("FrameDuration", 0.02)
    motion_weight = data.get("MotionWeight", 0.5)

    print(f"数据集信息:")
    print(f"  - 总帧数: {len(frames)}")
    print(f"  - 帧时长: {frame_duration}s")
    print(f"  - 动作权重: {motion_weight}")
    print(f"  - 预计播放时长: {len(frames) * frame_duration:.2f}s")

    first_frame = frames[0]
    print(f"\n第一帧数据验证:")
    print(f"  - 每帧数据量: {len(first_frame)}")
    pos, rot, joints = extract_pose_data(first_frame, joint_pos_size)
    print(f"  - 位置 (POS): {pos}")
    print(f"  - 旋转 (ROT): {rot}")
    print(f"  - 关节位置数: {len(joints)}")

    record_video = getattr(args, "record", False)
    save_images = getattr(args, "save_images", False)

    video = None
    cam_handle = None
    if record_video:
        try:
            import cv2

            camera_props = gymapi.CameraProperties()
            camera_props.width = 1280
            camera_props.height = 720
            cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)

            camera_local_transform = gymapi.Transform()
            camera_local_transform.p = gymapi.Vec3(2.0, 2.0, 1.5)
            camera_local_transform.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 0, 1), np.deg2rad(45)
            )

            body_handle = env.gym.get_actor_rigid_body_handle(
                env.envs[0], env.actor_handles[0], 0
            )
            env.gym.attach_camera_to_body(
                cam_handle,
                env.envs[0],
                body_handle,
                camera_local_transform,
                gymapi.FOLLOW_POSITION,
            )

            video_path = getattr(args, "output", None) or "dataset_playback.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(
                video_path, fourcc, int(1.0 / frame_duration), (1280, 720)
            )
            print(f"\n视频将保存到: {video_path}")
        except ImportError:
            print("警告: 未安装 opencv-python，无法录制视频")
            record_video = False

    img_dir = None
    img_count = 0
    if save_images:
        img_dir = getattr(args, "img_dir", None) or "dataset_frames"
        os.makedirs(img_dir, exist_ok=True)
        print(f"图像将保存到: {img_dir}/")
        if cam_handle is None:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1280
            camera_props.height = 720
            cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)

            camera_local_transform = gymapi.Transform()
            camera_local_transform.p = gymapi.Vec3(2.0, 2.0, 1.5)
            camera_local_transform.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 0, 1), np.deg2rad(45)
            )

            body_handle = env.gym.get_actor_rigid_body_handle(
                env.envs[0], env.actor_handles[0], 0
            )
            env.gym.attach_camera_to_body(
                cam_handle,
                env.envs[0],
                body_handle,
                camera_local_transform,
                gymapi.FOLLOW_POSITION,
            )

    print("\n开始播放数据集...")
    print("按 Ctrl+C 停止播放")
    print(f"播放模式: 循环播放, 帧间隔: {frame_duration}s\n")

    sim_dt = env.dt
    decimation = env.cfg.control.decimation
    step_dt = sim_dt * decimation

    frame_hold_steps = max(1, int(round(frame_duration / step_dt)))
    actual_frame_duration = frame_hold_steps * step_dt

    print(f"仿真参数:")
    print(f"  - 仿真步长 (dt): {sim_dt:.4f}s")
    print(f"  - 控制 decimation: {decimation}")
    print(f"  - 实际控制步长: {step_dt:.4f}s")
    print(f"  - 每帧持续步数: {frame_hold_steps}")
    print(f"  - 实际帧间隔: {actual_frame_duration:.4f}s\n")

    frame_idx = 0
    step_count = 0
    total_frames_played = 0
    loop_count = 0

    try:
        while True:
            frame = frames[frame_idx]
            position, rotation, joint_positions = extract_pose_data(
                frame, joint_pos_size
            )

            set_robot_pose(
                env, 0, position, rotation, joint_positions, num_dof
            )

            for _ in range(frame_hold_steps):
                env.gym.simulate(env.sim)
                env.gym.fetch_results(env.sim, True)

                env.gym.refresh_dof_state_tensor(env.sim)
                env.gym.refresh_actor_root_state_tensor(env.sim)
                env.gym.refresh_rigid_body_state_tensor(env.sim)

                if not args.headless:
                    env.render()
                    env.gym.step_graphics(env.sim)
                    if cam_handle is not None:
                        env.gym.render_all_camera_sensors(env.sim)

                if record_video and video is not None and cam_handle is not None:
                    import cv2

                    img = env.gym.get_camera_image(
                        env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR
                    ).reshape((720, 1280, 4))[:, :, :3]
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    video.write(img)

                if save_images and img_dir is not None and cam_handle is not None:
                    img = env.gym.get_camera_image(
                        env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR
                    ).reshape((720, 1280, 4))[:, :, :3]
                    img_path = os.path.join(
                        img_dir,
                        f"loop{loop_count:03d}_frame_{frame_idx:04d}_step_{_:02d}.png",
                    )
                    from PIL import Image

                    Image.fromarray(img).save(img_path)
                    img_count += 1

                step_count += 1

            total_frames_played += 1
            frame_idx += 1

            if frame_idx >= len(frames):
                loop_count += 1
                frame_idx = 0

    except KeyboardInterrupt:
        print(f"\n\n播放被用户中断")
        print(f"  - 共播放 {loop_count} 轮完整循环 + {frame_idx} 帧")
        print(f"  - 总计 {total_frames_played} 帧")
        if img_count > 0:
            print(f"  - 保存了 {img_count} 张图像")

    finally:
        if video is not None:
            video.release()
            print(f"  - 视频已保存")


def main():
    import argparse

    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument(
        "--dataset",
        type=str,
        default="d1_stand_up.txt",
        help="数据集文件名，可含子目录，如 all/d1_stand_up.txt",
    )
    temp_parser.add_argument(
        "--robot",
        type=str,
        default=None,
        help="d1 或 d1h，对应 resources/<robot>/datasets；不传则根据 --task 是否含 d1h 自动推断",
    )
    temp_parser.add_argument("--record", action="store_true")
    temp_parser.add_argument("--output", type=str, default=None)
    temp_parser.add_argument("--save_images", action="store_true")
    temp_parser.add_argument("--img_dir", type=str, default=None)

    temp_args, remaining_argv = temp_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv

    args = get_args()

    args.dataset = temp_args.dataset
    args.robot = temp_args.robot
    args.record = temp_args.record
    args.output = temp_args.output
    args.save_images = temp_args.save_images
    args.img_dir = temp_args.img_dir

    if not hasattr(args, "task") or args.task is None:
        args.task = "d1_flat_play"

    if args.robot is not None and args.robot not in ("d1", "d1h"):
        print("错误: --robot 只能为 d1 或 d1h")
        sys.exit(1)

    play_dataset(args)


if __name__ == "__main__":
    main()
