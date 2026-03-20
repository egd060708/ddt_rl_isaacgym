from utils.task_registry import task_registry
from .d1_rough_config import *
task_registry.register("d1_rough",D1Rough,D1RoughCfg(),D1RoughCfgPPO())  
task_registry.register("d1_rough_play",D1Rough,D1RoughCfg_Play(),D1RoughCfgPPO())  

from .d1_flat_config import *
task_registry.register("d1_flat",D1Flat,D1FlatCfg(),D1FlatCfgPPO())
task_registry.register("d1_flat_play",D1Flat,D1FlatCfg_Play(),D1FlatCfgPPO())

from .d1_amp_flat_config import *
task_registry.register("d1_amp_flat",D1AMPFlat,D1AMPFlatCfg(),D1AMPFlatCfgPPO())
task_registry.register("d1_amp_flat_play",D1AMPFlat,D1AMPFlatCfg_Play(),D1AMPFlatCfgPPO())