# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.datasets.early_fusion_dataset_dair import EarlyFusionDatasetDAIR
from opencood.data_utils.dair_datasets.intermediate_fusion_dataset_dair import IntermediateFusionDatasetDAIR
from opencood.data_utils.dair_datasets.intermediate_fusion_flow import IntermediateFusionFlow
# from opencood.data_utils.dair_datasets.late_fusion_dataset_dair import LateFusionDatasetDAIR
# from opencood.data_utils.dair_datasets.intermediate_fusion_dataset_seq import IntermediateFusionDatasetSeq
# from opencood.data_utils.datasets.intermediate_fusion_dataset_test import IntermediateFusionDatasetTest
# from opencood.data_utils.datasets.intermediate_fusion_dataset_carla import IntermediateFusionDatasetCarla
# from opencood.data_utils.datasets.late_fusion_dataset_carla import LateFusionDatasetCarla
# from opencood.data_utils.datasets.intermediate_fusion_ffnet_carla import IntermediateFusionFFnetCarla
# from opencood.data_utils.datasets.intermediate_fusion_dataset_carla_infer import IntermediateFusionDatasetCarlaInfer
# from opencood.data_utils.datasets.intermediate_fusion_motion import IntermediateFusionMotion    
# from opencood.data_utils.datasets.intermediate_flow_motion import IntermediateFlowMotion
# from opencood.data_utils.datasets.late_fusion_dataset_motion import LateFusionDatasetMotion
__all__ = {
    'EarlyFusionDatasetDAIR': EarlyFusionDatasetDAIR,
    'IntermediateFusionDatasetDAIR' : IntermediateFusionDatasetDAIR,
    'IntermediateFusionFlow': IntermediateFusionFlow,
}   
    # 'IntermediateFusionDatasetDAIR': IntermediateFusionDatasetDAIR,
    # 'LateFusionDatasetDAIR': LateFusionDatasetDAIR,
    # 'IntermediateFusionDatasetSeq': IntermediateFusionDatasetSeq,
# the final range for evaluation
GT_RANGE_OPV2V = [-140, -40, -3, 140, 40, 1]
GT_RANGE_V2XSIM = [-32, -32, -3, 32, 32, 1]
# The communication range for cavs
COM_RANGE = 70


def build_dataset(dataset_cfg, visualize=False, train=True):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset
