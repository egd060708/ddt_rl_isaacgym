#!/usr/bin/env python3
"""
机器人数据集播放脚本
用于播放 /resources/d1/datasets 目录下的数据集文件

数据集每帧包含69个数据：
- POS_SIZE (3): 根节点位置 x, y, z
- ROT_SIZE (4): 根节点旋转四元数 x, y, z, w
- JOINT_POS_SIZE (16): 16个关节位置

使用方法:
    python scripts/play_dataset.py --dataset d1_stand_up.txt
    python scripts/play_dataset.py --dataset d1_turn_left_06.txt --record --output video.mp4
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

# 数据集帧数据结构（根据用户提供的图片）
POS_SIZE = 3           # 位置数据大小
ROT_SIZE = 4           # 旋转数据大小（四元数）
JOINT_POS_SIZE = 16    # 关节位置数据大小
FRAME_HEADER_SIZE = POS_SIZE + ROT_SIZE + JOINT_POS_SIZE  # 23


def load_dataset(dataset_path: str):
    """加载数据集文件
    
    Args:
        dataset_path: 数据集文件路径
        
    Returns:
        dict: 包含 FrameDuration, MotionWeight, Frames 的字典
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data


def extract_pose_data(frame: list):
    """从帧数据中提取位姿信息
    
    Args:
        frame: 一帧数据列表
        
    Returns:
        tuple: (position, rotation, joint_positions)
        - position: [x, y, z]
        - rotation: [x, y, z, w] (四元数)
        - joint_positions: 16个关节位置
    """
    position = np.array(frame[0:POS_SIZE])
    rotation = np.array(frame[POS_SIZE:POS_SIZE + ROT_SIZE])
    joint_positions = np.array(frame[POS_SIZE + ROT_SIZE:POS_SIZE + ROT_SIZE + JOINT_POS_SIZE])
    return position, rotation, joint_positions


def set_robot_pose(env, env_id: int, position: np.ndarray, rotation: np.ndarray, joint_positions: np.ndarray):
    """设置机器人的位姿
    
    Args:
        env: 环境对象
        env_id: 环境ID
        position: 位置 [x, y, z]
        rotation: 旋转四元数 [x, y, z, w]
        joint_positions: 关节位置 (16个)
    """
    # 设置根节点状态
    # root_states 格式: [pos(3), rot(4), lin_vel(3), ang_vel(3)]
    env.root_states[env_id, :3] = torch.tensor(position, dtype=torch.float32, device=env.device)
    env.root_states[env_id, 3:7] = torch.tensor(rotation, dtype=torch.float32, device=env.device)
    
    # 设置关节位置
    # 确保 joint_positions 长度为16
    if len(joint_positions) != 16:
        raise ValueError(f"关节位置数量错误: 期望16个, 实际{len(joint_positions)}个")
    
    env.dof_pos[env_id] = torch.tensor(joint_positions, dtype=torch.float32, device=env.device)
    env.dof_vel[env_id] = 0.  # 重置关节速度为0
    
    # 将状态应用到模拟器
    env_ids_int32 = torch.tensor([env_id], dtype=torch.int32, device=env.device)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        1
    )
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),  # 注意: 使用 env.dof_state 而非 env.dof_states
        gymtorch.unwrap_tensor(env_ids_int32),
        1
    )


def play_dataset(args):
    """播放数据集的主函数"""
    
    # 获取环境配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 覆盖一些参数用于可视化
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)  # 只使用一个环境
    env_cfg.terrain.mesh_type = 'plane'  # 使用平地
    env_cfg.terrain.curriculum = False
    
    # 准备环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 加载数据集
    dataset_name = getattr(args, 'dataset', 'd1_stand_up.txt')
    dataset_path = os.path.join(ROOT_DIR, 'resources', 'd1', 'datasets', dataset_name)
    print(f"正在加载数据集: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集文件不存在: {dataset_path}")
        print(f"可用数据集:")
        datasets_dir = os.path.join(ROOT_DIR, 'resources', 'd1', 'datasets')
        if os.path.exists(datasets_dir):
            for f in os.listdir(datasets_dir):
                if f.endswith('.txt'):
                    print(f"  - {f}")
        return
    
    data = load_dataset(dataset_path)
    frames = data['Frames']
    frame_duration = data.get('FrameDuration', 0.02)
    motion_weight = data.get('MotionWeight', 0.5)
    
    print(f"数据集信息:")
    print(f"  - 总帧数: {len(frames)}")
    print(f"  - 帧时长: {frame_duration}s")
    print(f"  - 动作权重: {motion_weight}")
    print(f"  - 预计播放时长: {len(frames) * frame_duration:.2f}s")
    
    # 验证第一帧数据
    first_frame = frames[0]
    print(f"\n第一帧数据验证:")
    print(f"  - 每帧数据量: {len(first_frame)}")
    pos, rot, joints = extract_pose_data(first_frame)
    print(f"  - 位置 (POS): {pos}")
    print(f"  - 旋转 (ROT): {rot}")
    print(f"  - 关节位置数: {len(joints)}")
    
    # 播放控制
    record_video = getattr(args, 'record', False)
    save_images = getattr(args, 'save_images', False)
    
    # 视频录制设置
    video = None
    cam_handle = None
    if record_video:
        try:
            import cv2
            # 创建相机传感器
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1280
            camera_props.height = 720
            cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
            
            # 设置相机位置
            camera_local_transform = gymapi.Transform()
            camera_local_transform.p = gymapi.Vec3(2.0, 2.0, 1.5)
            camera_local_transform.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 0, 1), np.deg2rad(45)
            )
            
            body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], 0)
            env.gym.attach_camera_to_body(
                cam_handle, env.envs[0], body_handle, 
                camera_local_transform, gymapi.FOLLOW_POSITION
            )
            
            video_path = getattr(args, 'output', None) or 'dataset_playback.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, int(1.0 / frame_duration), (1280, 720))
            print(f"\n视频将保存到: {video_path}")
        except ImportError:
            print("警告: 未安装 opencv-python，无法录制视频")
            record_video = False
    
    # 创建图像保存目录
    img_dir = None
    img_count = 0
    if save_images:
        img_dir = getattr(args, 'img_dir', None) or 'dataset_frames'
        os.makedirs(img_dir, exist_ok=True)
        print(f"图像将保存到: {img_dir}/")
        # 如果保存图像但没有录制视频，也需要相机
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
            
            body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], 0)
            env.gym.attach_camera_to_body(
                cam_handle, env.envs[0], body_handle, 
                camera_local_transform, gymapi.FOLLOW_POSITION
            )
    
    print("\n开始播放数据集...")
    print("按 Ctrl+C 停止播放")
    print(f"播放模式: 循环播放, 帧间隔: {frame_duration}s\n")
    
    # 获取仿真参数
    sim_dt = env.dt  # 仿真时间步长
    decimation = env.cfg.control.decimation  # 控制 decimation
    step_dt = sim_dt * decimation  # 实际控制时间步长
    
    # 计算每帧数据应该持续多少个仿真步
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
        while True:  # 无限循环播放
            # 获取当前帧数据
            frame = frames[frame_idx]
            position, rotation, joint_positions = extract_pose_data(frame)
            
            # 设置机器人位姿
            set_robot_pose(env, 0, position, rotation, joint_positions)
            
            # 在帧间隔内保持该姿势，每步进行物理仿真
            for _ in range(frame_hold_steps):
                # 执行仿真步
                env.gym.simulate(env.sim)
                env.gym.fetch_results(env.sim, True)
                
                # 刷新状态
                env.gym.refresh_dof_state_tensor(env.sim)
                env.gym.refresh_actor_root_state_tensor(env.sim)
                env.gym.refresh_rigid_body_state_tensor(env.sim)
                
                # 渲染
                if not args.headless:
                    env.render()
                    env.gym.step_graphics(env.sim)
                    if cam_handle is not None:
                        env.gym.render_all_camera_sensors(env.sim)
                
                # 录制视频
                if record_video and video is not None and cam_handle is not None:
                    import cv2
                    img = env.gym.get_camera_image(
                        env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR
                    ).reshape((720, 1280, 4))[:, :, :3]
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    video.write(img)
                
                # 保存图像帧
                if save_images and img_dir is not None and cam_handle is not None:
                    img = env.gym.get_camera_image(
                        env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR
                    ).reshape((720, 1280, 4))[:, :, :3]
                    img_path = os.path.join(img_dir, f"loop{loop_count:03d}_frame_{frame_idx:04d}_step_{_:02d}.png")
                    from PIL import Image
                    Image.fromarray(img).save(img_path)
                    img_count += 1
                
                step_count += 1
            
            total_frames_played += 1
            
            # 打印进度（每30帧或循环开始时）
            if frame_idx % 30 == 0 or frame_idx == len(frames) - 1:
                if loop_count == 0:
                    progress = (frame_idx / len(frames)) * 100
                    # print(f"第1轮播放: {frame_idx}/{len(frames)} ({progress:.1f}%)")
                else:
                    # print(f"第{loop_count + 1}轮播放: {frame_idx}/{len(frames)}")
                    pass
            
            # 移动到下一帧
            frame_idx += 1
            
            # 如果播放完所有帧，重新开始（循环播放）
            if frame_idx >= len(frames):
                loop_count += 1
                frame_idx = 0
                # print(f"\n>>> 第 {loop_count + 1} 轮循环开始\n")
            
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
    """主函数 - 处理命令行参数"""
    # 首先解析我们自己的参数
    import argparse
    
    # 创建临时解析器获取 dataset 相关参数
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--dataset', type=str, default='d1_stand_up.txt')
    temp_parser.add_argument('--record', action='store_true')
    temp_parser.add_argument('--output', type=str, default=None)
    temp_parser.add_argument('--save_images', action='store_true')
    temp_parser.add_argument('--img_dir', type=str, default=None)
    
    # 解析已知参数
    temp_args, remaining_argv = temp_parser.parse_known_args()
    
    # 更新 sys.argv 以便 get_args 能正确解析
    sys.argv = [sys.argv[0]] + remaining_argv
    
    # 获取标准参数
    args = get_args()
    
    # 添加自定义参数
    args.dataset = temp_args.dataset
    args.record = temp_args.record
    args.output = temp_args.output
    args.save_images = temp_args.save_images
    args.img_dir = temp_args.img_dir
    
    # 确保有任务名称
    if not hasattr(args, 'task') or args.task is None:
        args.task = 'd1_flat_play'
    
    play_dataset(args)


if __name__ == '__main__':
    main()
