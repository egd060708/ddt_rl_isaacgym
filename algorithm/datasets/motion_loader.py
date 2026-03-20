import os
import glob
import json
import logging

import torch
import numpy as np
from pybullet_utils import transformations

# from rsl_rl.utils import utils
import utils
from algorithm.datasets import pose3d
from algorithm.datasets import motion_util


# 默认与四足 d1 数据集一致；AMP 任务可在 cfg.env.amp_motion_layout 中覆盖
_DEFAULT_MOTION_LAYOUT = dict(
    pos_size=3,
    rot_size=4,
    joint_pos_size=16,
    joint_vel_size=16,
    tar_toe_pos_local_size=12,
    tar_toe_vel_local_size=12,
    linear_vel_size=3,
    angular_vel_size=3,
    amp_feed_forward_style="d1_without_wheel_pos",
    amp_observation_dim=None,
)


def motion_layout_from_legged_cfg(legged_cfg):
    """从 LeggedRobotCfg 读取 env.amp_motion_layout（嵌套 class），返回 dict。"""
    if legged_cfg is None or not hasattr(legged_cfg, "env"):
        return {}
    env = legged_cfg.env
    if not hasattr(env, "amp_motion_layout"):
        return {}
    L = env.amp_motion_layout
    d = {}
    for k in (
        "pos_size",
        "rot_size",
        "joint_pos_size",
        "joint_vel_size",
        "tar_toe_pos_local_size",
        "tar_toe_vel_local_size",
        "linear_vel_size",
        "angular_vel_size",
        "amp_feed_forward_style",
        "amp_observation_dim",
    ):
        if hasattr(L, k):
            d[k] = getattr(L, k)
    return d


class AMPLoader:
    """AMP 动作加载器。布局尺寸可通过 cfg.env.amp_motion_layout 配置（如 d1h 8 关节、6 维足端）。"""

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=None,
        motion_layout=None,
    ):
        """
        motion_layout: dict，可包含 joint_pos_size, joint_vel_size, tar_toe_pos_local_size,
        tar_toe_vel_local_size, linear_vel_size, angular_vel_size,
        amp_feed_forward_style ('d1_without_wheel_pos' | 'd1h_without_wheel_pos' | 'contiguous'),
        amp_observation_dim (可选，显式指定与 env.get_amp_observations 一致的维度)。
        """
        if motion_files is None:
            motion_files = glob.glob("resources/d1/datasets/all/*")
        if not motion_files:
            raise ValueError(
                "AMPLoader: motion_files 为空。请设置 cfg.env.amp_motion_files 或"
                "在 resources/<robot>/datasets/all/ 下放置动作 .txt（JSON 格式）。"
            )

        layout = {**_DEFAULT_MOTION_LAYOUT, **(motion_layout or {})}
        self._apply_layout(layout)

        self.device = device
        self.time_between_frames = time_between_frames

        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file, "r") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])

                if motion_data.shape[1] < self.joint_vel_end_idx:
                    raise ValueError(
                        f"{motion_file}: 每帧长度 {motion_data.shape[1]} < 期望 {self.joint_vel_end_idx} "
                        f"(请检查 amp_motion_layout 与数据集格式是否一致)"
                    )

                for f_i in range(motion_data.shape[0]):
                    root_rot = self.get_root_rot(motion_data[f_i])
                    root_rot = pose3d.QuaternionNormalize(root_rot)
                    root_rot = motion_util.standardize_quaternion(root_rot)
                    motion_data[
                        f_i,
                        self.pos_size : (self.pos_size + self.rot_size),
                    ] = root_rot

                self.trajectories.append(
                    torch.tensor(
                        motion_data[:, self.root_rot_end_idx : self.joint_vel_end_idx],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.trajectories_full.append(
                    torch.tensor(
                        motion_data[:, : self.joint_vel_end_idx],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(
            self.trajectory_weights
        )
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(
                traj_idxs, times + self.time_between_frames
            )
            print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def _apply_layout(self, layout):
        self.pos_size = int(layout["pos_size"])
        self.rot_size = int(layout["rot_size"])
        self.joint_pos_size = int(layout["joint_pos_size"])
        self.joint_vel_size = int(layout["joint_vel_size"])
        self.tar_toe_pos_local_size = int(layout["tar_toe_pos_local_size"])
        self.tar_toe_vel_local_size = int(layout["tar_toe_vel_local_size"])
        self.linear_vel_size = int(layout["linear_vel_size"])
        self.angular_vel_size = int(layout["angular_vel_size"])
        self.amp_feed_forward_style = layout["amp_feed_forward_style"]
        self._amp_observation_dim_override = layout.get("amp_observation_dim")

        self.root_pos_start_idx = 0
        self.root_pos_end_idx = self.root_pos_start_idx + self.pos_size
        self.root_rot_start_idx = self.root_pos_end_idx
        self.root_rot_end_idx = self.root_rot_start_idx + self.rot_size
        self.joint_pose_start_idx = self.root_rot_end_idx
        self.joint_pose_end_idx = self.joint_pose_start_idx + self.joint_pos_size
        self.tar_toe_pos_local_start_idx = self.joint_pose_end_idx
        self.tar_toe_pos_local_end_idx = (
            self.tar_toe_pos_local_start_idx + self.tar_toe_pos_local_size
        )
        self.linear_vel_start_idx = self.tar_toe_pos_local_end_idx
        self.linear_vel_end_idx = self.linear_vel_start_idx + self.linear_vel_size
        self.angular_vel_start_idx = self.linear_vel_end_idx
        self.angular_vel_end_idx = self.angular_vel_start_idx + self.angular_vel_size
        self.joint_vel_start_idx = self.angular_vel_end_idx
        self.joint_vel_end_idx = self.joint_vel_start_idx + self.joint_vel_size
        self.tar_toe_vel_local_start_idx = self.joint_vel_end_idx
        self.tar_toe_vel_local_end_idx = (
            self.tar_toe_vel_local_start_idx + self.tar_toe_vel_local_size
        )
        # full frame 用于 trajectories_full 时截断到 joint_vel_end_idx（不含 tar_toe_vel）

    def reorder_from_pybullet_to_isaac(self, motion_data):
        """仅适用于四足 16 关节 ×4 腿划分；其它布局请勿调用。"""
        if self.joint_pos_size != 16 or self.tar_toe_pos_local_size != 12:
            raise NotImplementedError(
                "reorder_from_pybullet_to_isaac 仅支持 joint_pos_size=16 且 tar_toe_pos=12"
            )
        root_pos = self.get_root_pos_batch(motion_data)
        root_rot = self.get_root_rot_batch(motion_data)

        jp_fr, jp_fl, jp_rr, jp_rl = np.split(
            self.get_joint_pose_batch(motion_data), 4, axis=1
        )
        joint_pos = np.hstack([jp_fl, jp_fr, jp_rl, jp_rr])

        fp_fr, fp_fl, fp_rr, fp_rl = np.split(
            self.get_tar_toe_pos_local_batch(motion_data), 4, axis=1
        )
        foot_pos = np.hstack([fp_fl, fp_fr, fp_rl, fp_rr])

        lin_vel = self.get_linear_vel_batch(motion_data)
        ang_vel = self.get_angular_vel_batch(motion_data)

        jv_fr, jv_fl, jv_rr, jv_rl = np.split(
            self.get_joint_vel_batch(motion_data), 4, axis=1
        )
        joint_vel = np.hstack([jv_fl, jv_fr, jv_rl, jv_rr])

        fv_fr, fv_fl, fv_rr, fv_rl = np.split(
            self.get_tar_toe_vel_local_batch(motion_data), 4, axis=1
        )
        foot_vel = np.hstack([fv_fl, fv_fr, fv_rl, fv_rr])

        return np.hstack(
            [
                root_pos,
                root_rot,
                joint_pos,
                foot_pos,
                lin_vel,
                ang_vel,
                joint_vel,
                foot_vel,
            ]
        )

    def weighted_traj_idx_sample(self):
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True
        )

    def traj_time_sample(self, traj_idx):
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = (
            self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs))
            - subst
        )
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(
            np.int
        )
        traj_w = self.trajectories[0].shape[1]
        all_frame_starts = torch.zeros(len(traj_idxs), traj_w, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), traj_w, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(
            -1
        )
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(
            np.int
        )
        amp_seg_len = self.joint_vel_end_idx - self.joint_pose_start_idx
        all_frame_pos_starts = torch.zeros(
            len(traj_idxs), self.pos_size, device=self.device
        )
        all_frame_pos_ends = torch.zeros(
            len(traj_idxs), self.pos_size, device=self.device
        )
        all_frame_rot_starts = torch.zeros(
            len(traj_idxs), self.rot_size, device=self.device
        )
        all_frame_rot_ends = torch.zeros(
            len(traj_idxs), self.rot_size, device=self.device
        )
        all_frame_amp_starts = torch.zeros(len(traj_idxs), amp_seg_len, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs), amp_seg_len, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = self.get_root_pos_batch(
                trajectory[idx_low[traj_mask]]
            )
            all_frame_pos_ends[traj_mask] = self.get_root_pos_batch(
                trajectory[idx_high[traj_mask]]
            )
            all_frame_rot_starts[traj_mask] = self.get_root_rot_batch(
                trajectory[idx_low[traj_mask]]
            )
            all_frame_rot_ends[traj_mask] = self.get_root_rot_batch(
                trajectory[idx_high[traj_mask]]
            )
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, self.joint_pose_start_idx : self.joint_vel_end_idx
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, self.joint_pose_start_idx : self.joint_vel_end_idx
            ]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(
            -1
        )

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = utils.quaternion_slerp_safe(
            all_frame_rot_starts, all_frame_rot_ends, blend
        )
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def get_frame(self):
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
        times = self.traj_time_sample_batch(traj_idxs)
        return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        root_pos0, root_pos1 = self.get_root_pos(frame0), self.get_root_pos(frame1)
        root_rot0, root_rot1 = self.get_root_rot(frame0), self.get_root_rot(frame1)
        joints0, joints1 = self.get_joint_pose(frame0), self.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = (
            self.get_tar_toe_pos_local(frame0),
            self.get_tar_toe_pos_local(frame1),
        )
        linear_vel_0, linear_vel_1 = (
            self.get_linear_vel(frame0),
            self.get_linear_vel(frame1),
        )
        angular_vel_0, angular_vel_1 = (
            self.get_angular_vel(frame0),
            self.get_angular_vel(frame1),
        )
        joint_vel_0, joint_vel_1 = self.get_joint_vel(frame0), self.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = transformations.quaternion_slerp(
            root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend
        )
        blend_root_rot = torch.tensor(
            motion_util.standardize_quaternion(blend_root_rot),
            dtype=torch.float32,
            device=self.device,
        )
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat(
            [
                blend_root_pos,
                blend_root_rot,
                blend_joints,
                blend_tar_toe_pos,
                blend_linear_vel,
                blend_angular_vel,
                blend_joints_vel,
            ]
        )

    def _expert_features_from_full_frames(self, preloaded, idxs):
        """从完整帧张量构造与判别器一致的专家特征 (B, observation_dim)。idxs: np.ndarray 或 torch.LongTensor。"""
        if isinstance(idxs, np.ndarray):
            idxs_t = torch.from_numpy(idxs).long().to(preloaded.device)
        else:
            idxs_t = idxs
        root_z = preloaded[
            idxs_t, self.root_pos_start_idx + 2 : self.root_pos_start_idx + 3
        ]
        if self.amp_feed_forward_style == "d1_without_wheel_pos":
            if self.joint_pos_size != 16:
                raise ValueError(
                    "d1_without_wheel_pos 需要 joint_pos_size=16"
                )
            jp = self.joint_pose_start_idx
            body = torch.cat(
                [
                    preloaded[idxs_t, jp : jp + 3],
                    preloaded[idxs_t, jp + 4 : jp + 7],
                    preloaded[idxs_t, jp + 8 : jp + 11],
                    preloaded[idxs_t, jp + 12 : jp + 15],
                    preloaded[idxs_t, jp + 16 : self.joint_pose_end_idx],
                    preloaded[idxs_t, self.joint_pose_end_idx : self.joint_vel_end_idx],
                ],
                dim=1,
            )
        elif self.amp_feed_forward_style == "d1h_without_wheel_pos":
            if self.joint_pos_size != 8:
                raise ValueError(
                    "d1h_without_wheel_pos 需要 joint_pos_size=8"
                )
            jp = self.joint_pose_start_idx
            body = torch.cat(
                [
                    preloaded[idxs_t, jp : jp + 3],
                    preloaded[idxs_t, jp + 4 : jp + 7],
                    preloaded[idxs_t, jp + 8 : self.joint_pose_end_idx],
                    preloaded[idxs_t, self.joint_pose_end_idx : self.joint_vel_end_idx],
                ],
                dim=1,
            )
        else:
            body = preloaded[
                idxs_t, self.joint_pose_start_idx : self.joint_vel_end_idx
            ]
        return torch.cat([body, root_z], dim=-1)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(
                    self.preloaded_s.shape[0], size=mini_batch_size
                )
                s = self._expert_features_from_full_frames(self.preloaded_s, idxs)
                s_next = self._expert_features_from_full_frames(
                    self.preloaded_s_next, idxs
                )
            else:
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                full0 = torch.stack(
                    [
                        self.get_full_frame_at_time(int(t), float(tm))
                        for t, tm in zip(traj_idxs, times)
                    ]
                )
                full1 = torch.stack(
                    [
                        self.get_full_frame_at_time(
                            int(t), float(tm) + self.time_between_frames
                        )
                        for t, tm in zip(traj_idxs, times)
                    ]
                )
                B = full0.shape[0]
                ar = torch.arange(B, device=self.device)
                s = self._expert_features_from_full_frames(full0, ar)
                s_next = self._expert_features_from_full_frames(full1, ar)
            yield s, s_next

    @property
    def observation_dim(self):
        if self._amp_observation_dim_override is not None:
            return int(self._amp_observation_dim_override)
        traj_w = self.trajectories[0].shape[1]
        if self.amp_feed_forward_style == "d1_without_wheel_pos":
            return traj_w + 1 - 4
        elif self.amp_feed_forward_style == "d1h_without_wheel_pos":
            return traj_w + 1 - 2
        return traj_w + 1

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(self, pose):
        return pose[self.root_pos_start_idx : self.root_pos_end_idx]

    def get_root_pos_batch(self, poses):
        return poses[:, self.root_pos_start_idx : self.root_pos_end_idx]

    def get_root_rot(self, pose):
        return pose[self.root_rot_start_idx : self.root_rot_end_idx]

    def get_root_rot_batch(self, poses):
        return poses[:, self.root_rot_start_idx : self.root_rot_end_idx]

    def get_joint_pose(self, pose):
        return pose[self.joint_pose_start_idx : self.joint_pose_end_idx]

    def get_joint_pose_batch(self, poses):
        return poses[:, self.joint_pose_start_idx : self.joint_pose_end_idx]

    def get_tar_toe_pos_local(self, pose):
        return pose[
            self.tar_toe_pos_local_start_idx : self.tar_toe_pos_local_end_idx
        ]

    def get_tar_toe_pos_local_batch(self, poses):
        return poses[
            :, self.tar_toe_pos_local_start_idx : self.tar_toe_pos_local_end_idx
        ]

    def get_linear_vel(self, pose):
        return pose[self.linear_vel_start_idx : self.linear_vel_end_idx]

    def get_linear_vel_batch(self, poses):
        return poses[:, self.linear_vel_start_idx : self.linear_vel_end_idx]

    def get_angular_vel(self, pose):
        return pose[self.angular_vel_start_idx : self.angular_vel_end_idx]

    def get_angular_vel_batch(self, poses):
        return poses[:, self.angular_vel_start_idx : self.angular_vel_end_idx]

    def get_joint_vel(self, pose):
        return pose[self.joint_vel_start_idx : self.joint_vel_end_idx]

    def get_joint_vel_batch(self, poses):
        return poses[:, self.joint_vel_start_idx : self.joint_vel_end_idx]

    def get_tar_toe_vel_local(self, pose):
        return pose[
            self.tar_toe_vel_local_start_idx : self.tar_toe_vel_local_end_idx
        ]

    def get_tar_toe_vel_local_batch(self, poses):
        return poses[
            :, self.tar_toe_vel_local_start_idx : self.tar_toe_vel_local_end_idx
        ]
