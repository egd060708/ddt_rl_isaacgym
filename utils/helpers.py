import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

def partial_checkpoint_load(pretrain_dict, model):
    
    # 删除预训练权重中不存在的key
    model_dict = model.state_dict()
    # 更新模型权重
    pretrain_dict = {k: v for k,v in pretrain_dict.items() if k in model_dict}
    # 加载模型权重
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    return model

def move_to(obj, device):
    if isinstance(obj, dict):
       #return {key:obj[key].to(device) for key in obj}
        for key in obj:
            try:
                obj[key] = obj[key].to(device)
            except:
                obj[key] = obj[key]
        return obj
    else:
        raise TypeError("Invalid type for move_to")

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        runs.sort(key=lambda x: os.path.getmtime(os.path.join(root, x)))
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2N3poHim",
         "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,
         "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str, "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,
         "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,
         "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},

        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
         "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int,
         "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int,
         "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

def phase_schedualer(max_iters,phase1_end,phase2_end,lerp_steps,max_imi_weight):
    act_schedual = np.array([False]*max_iters)
    imitation_schedual = np.array([0.0]*max_iters)
    
    act_schedual[phase2_end:] = True
    step_by_value = float(max_imi_weight/lerp_steps)
    imitation_schedual[phase1_end:phase1_end+lerp_steps] = np.arange(0,max_imi_weight,step_by_value)
    imitation_schedual[phase1_end+lerp_steps:phase2_end] = max_imi_weight
    imitation_schedual[phase2_end:phase2_end+lerp_steps] = np.arange(max_imi_weight,0,-step_by_value)
    return act_schedual,imitation_schedual

def hard_phase_schedualer(max_iters,phase1_end):
    act_schedual = np.array([True]*max_iters)
    imitation_schedual = np.array([True]*max_iters)
    lag_schedual = np.array([False]*max_iters)

    act_schedual[phase1_end:] = False
    imitation_schedual[phase1_end:] = False
    lag_schedual[phase1_end:] = True
    return act_schedual,imitation_schedual,lag_schedual


def sanitize_config_for_dump(obj, _depth=0):
    """将配置对象转为可 JSON/YAML 序列化的 Python 原生类型。"""
    if _depth > 64:
        return "<max_depth>"
    # 必须先处理 numpy 标量：NumPy 2 中 np.float64 可能 isinstance(..., float) 为 True，
    # 但仍是 numpy 类型，PyYAML safe_dump 无法序列化，必须用 .item() 转成 Python 内置类型。
    if isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        if obj.size > 4096:
            return {"__ndarray__": f"shape={tuple(obj.shape)} dtype={obj.dtype}"}
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        if obj.numel() > 4096:
            return {"__tensor__": f"shape={tuple(obj.shape)} dtype={obj.dtype}"}
        return obj.detach().cpu().tolist()
    if isinstance(obj, type):
        return {"__class__": f"{obj.__module__}.{obj.__qualname__}"}
    if isinstance(obj, dict):
        return {str(k): sanitize_config_for_dump(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_config_for_dump(v, _depth + 1) for v in obj]
    if isinstance(obj, (set, frozenset)):
        try:
            seq = sorted(obj, key=lambda x: str(type(x)) + str(x))
        except TypeError:
            seq = list(obj)
        return [sanitize_config_for_dump(v, _depth + 1) for v in seq]
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return str(obj)
    if callable(obj) and not isinstance(obj, type):
        return {"__callable__": getattr(obj, "__name__", str(obj))}
    # 其余不可序列化对象
    try:
        return str(obj)
    except Exception:
        return "<non-serializable>"


def save_run_config_snapshot(log_dir, env_cfg, train_cfg, task_name=None, args=None):
    """
    将本次训练使用的环境配置与训练配置保存到与 model_*.pt 相同的日志目录。

    优先写入 config.yaml（需安装 PyYAML: pip install pyyaml）；
    若未安装则写入 config.json。

    Args:
        log_dir: 与 checkpoint 相同的目录，例如 logs/<exp>/<timestamp>_runname/
        env_cfg: LeggedRobotCfg 实例（已与命令行合并后）
        train_cfg: LeggedRobotCfgPPO 实例（已与命令行合并后）
        task_name: 注册任务名，如 d1_flat
        args: get_args() 返回的命名空间，可选，用于记录命令行参数
    """
    if not log_dir:
        return
    os.makedirs(log_dir, exist_ok=True)

    payload = {
        "task_name": task_name,
        "environment": sanitize_config_for_dump(class_to_dict(env_cfg)),
        "training": sanitize_config_for_dump(class_to_dict(train_cfg)),
    }
    if args is not None:
        try:
            ad = vars(args) if hasattr(args, "__dict__") else dict(args)
            payload["command_line_args"] = sanitize_config_for_dump(ad)
        except Exception as e:
            payload["command_line_args"] = {"__error__": str(e)}

    yaml_path = os.path.join(log_dir, "config.yaml")
    json_path = os.path.join(log_dir, "config.json")
    try:
        import yaml

        try:
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    payload,
                    f,
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                    width=120,
                )
            print(f"[config] 已保存运行配置(YAML): {yaml_path}")
        except yaml.representer.RepresenterError:
            import json

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(
                f"[config] YAML 序列化失败，已改为保存 JSON: {json_path}"
            )
    except ImportError:
        import json

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(
            f"[config] 未安装 PyYAML，已保存为 JSON（可 pip install pyyaml 以使用 YAML）: {json_path}"
        )

