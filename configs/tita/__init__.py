from utils.task_registry import task_registry
from .tita import *
from .tita_flat_config import *
task_registry.register("tita_flat",Tita,TitaFlatCfg(),TitaFlatCfgPPO())
task_registry.register("tita_flat_play",Tita,TitaFlatCfg_Play(),TitaFlatCfgPPO())
