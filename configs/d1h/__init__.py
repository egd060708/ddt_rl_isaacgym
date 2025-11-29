from utils.task_registry import task_registry
from .d1h_flat_config import *
task_registry.register("d1h_flat",D1HFlat,D1HFlatCfg(),D1HFlatCfgPPO())
task_registry.register("d1h_flat_play",D1HFlat,D1HFlatCfg_Play(),D1HFlatCfgPPO())

from .d1h_rough_config import *
task_registry.register("d1h_rough",D1HRough,D1HRoughCfg(),D1HRoughCfgPPO())
task_registry.register("d1h_rough_play",D1HRough,D1HRoughCfg_Play(),D1HRoughCfgPPO())
