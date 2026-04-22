from utils.task_registry import task_registry
from .d1h_flat_config import *
task_registry.register("d1h_flat",D1HFlat,D1HFlatCfg(),D1HFlatCfgPPO())
task_registry.register("d1h_flat_play",D1HFlat,D1HFlatCfg_Play(),D1HFlatCfgPPO())

from .d1h_amp_flat_config import *
task_registry.register("d1h_amp_flat", D1HAMPFlat, D1HAMPFlatCfg(), D1HAMPFlatCfgPPO())
task_registry.register("d1h_amp_flat_play", D1HAMPFlat, D1HAMPFlatCfg_Play(), D1HAMPFlatCfgPPO())
task_registry.register("d1h_amp_flat_baseline", D1HAMPFlat, D1HAMPFlatCfg(), D1HFlatCfgPPO())
task_registry.register("d1h_amp_flat_baseline_play", D1HAMPFlat, D1HAMPFlatCfg_Play(), D1HFlatCfgPPO())

from .d1h_rough_config import *
task_registry.register("d1h_rough",D1HRough,D1HRoughCfg(),D1HRoughCfgPPO())
task_registry.register("d1h_rough_play",D1HRough,D1HRoughCfg_Play(),D1HRoughCfgPPO())
