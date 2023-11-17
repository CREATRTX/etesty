# Copyright (c) OpenMMLab. All rights reserved.
"""MMDetection provides 17 registry nodes to support using modules across
projects. Each node is a child of the root registry in MMEngine.
More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""
from mmengine.registry import \
    WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mmengine.registry import Registry
from nuscenes.panoptic.panoptic_utils import get_frame_panoptic_instances, get_panoptic_instances_stats
from nuscenes.utils.color_map import get_colormap
from nuscenes.utils.data_io import load_bin_file


def truncate_class_name(class_name: str) -> str:
    """
    Truncate a given class name according to a pre-defined map.
    :param class_name: The long form (i.e. original form) of the class name.
    :return: The truncated form of the class name.
    """

    string_mapper = {
        "noise": 'noise',
        "human.pedestrian.adult": 'adult',
        "human.pedestrian.child": 'child',
        "human.pedestrian.wheelchair": 'wheelchair',
        "human.pedestrian.stroller": 'stroller',
        "human.pedestrian.personal_mobility": 'p.mobility',
        "human.pedestrian.police_officer": 'police',
        "human.pedestrian.construction_worker": 'worker',
        "animal": 'animal',
        "vehicle.car": 'car',
        "vehicle.motorcycle": 'motorcycle',
        "vehicle.bicycle": 'bicycle',
        "vehicle.bus.bendy": 'bus.bendy',
        "vehicle.bus.rigid": 'bus.rigid',
        "vehicle.truck": 'truck',
        "vehicle.construction": 'constr. veh',
        "vehicle.emergency.ambulance": 'ambulance',
        "vehicle.emergency.police": 'police car',
        "vehicle.trailer": 'trailer',
        "movable_object.barrier": 'barrier',
        "movable_object.trafficcone": 'trafficcone',
        "movable_object.pushable_pullable": 'push/pullable',
        "movable_object.debris": 'debris',
        "static_object.bicycle_rack": 'bicycle racks',
        "flat.driveable_surface": 'driveable',
        "flat.sidewalk": 'sidewalk',
        "flat.terrain": 'terrain',
        "flat.other": 'flat.other',
        "static.manmade": 'manmade',
        "static.vegetation": 'vegetation',
        "static.other": 'static.other',
        "vehicle.ego": "ego"
    }

    return string_mapper[class_name]


def render_histogram(nusc: NuScenes,
                     sort_by: str = 'count_desc',
                     verbose: bool = True,
                     font_size: int = 20,
                     save_as_img_name: str = None) -> None:
    """
    Render two histograms for the given nuScenes split. The top histogram depicts the number of scan-wise instances
    for each class, while the bottom histogram depicts the number of points for each class.
    :param nusc: A nuScenes object.
    :param sort_by: How to sort the classes to display in the plot (note that the x-axis, where the class names will be
        displayed on, is shared by the two histograms):
        - count_desc: Sort the classes by the number of points belonging to each class, in descending order.
        - count_asc: Sort the classes by the number of points belonging to each class, in ascending order.
        - name: Sort the classes by alphabetical order.
        - index: Sort the classes by their indices.
    :param verbose: Whether to display the plot in a window after rendering.
    :param font_size: Size of the font to use for the plot.
    :param save_as_img_name: Path (including image name and extension) to save the plot as.
from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import EVALUATOR as MMENGINE_EVALUATOR
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import LOG_PROCESSORS as MMENGINE_LOG_PROCESSORS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmengine.registry import \
    RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry(
    'runner', parent=MMENGINE_RUNNERS, locations=['mmdet.engine.runner'])
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor',
    parent=MMENGINE_RUNNER_CONSTRUCTORS,
    locations=['mmdet.engine.runner'])
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry(
    'loop', parent=MMENGINE_LOOPS, locations=['mmdet.engine.runner'])
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook', parent=MMENGINE_HOOKS, locations=['mmdet.engine.hooks'])

# manage data-related modules
DATASETS = Registry(
    'dataset', parent=MMENGINE_DATASETS, locations=['mmdet.datasets'])
DATA_SAMPLERS = Registry(
    'data sampler',
    parent=MMENGINE_DATA_SAMPLERS,
    locations=['mmdet.datasets.samplers'])
TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=['mmdet.datasets.transforms'])

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['mmdet.models'])
# manage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=MMENGINE_MODEL_WRAPPERS,
    locations=['mmdet.models'])
# manage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=MMENGINE_WEIGHT_INITIALIZERS,
    locations=['mmdet.models'])
# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util', parent=MMENGINE_TASK_UTILS, locations=['mmdet.models'])

# manage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry(
    'optimizer',
    parent=MMENGINE_OPTIMIZERS,
    locations=['mmdet.engine.optimizers'])
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry(
    'optim_wrapper',
    parent=MMENGINE_OPTIM_WRAPPERS,
    locations=['mmdet.engine.optimizers'])
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    locations=['mmdet.engine.optimizers'])
# manage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=MMENGINE_PARAM_SCHEDULERS,
    locations=['mmdet.engine.schedulers'])
# manage all kinds of metrics
METRICS = Registry(
    'metric', parent=MMENGINE_METRICS, locations=['mmdet.evaluation'])
# manage evaluator
EVALUATOR = Registry(
    'evaluator', parent=MMENGINE_EVALUATOR, locations=['mmdet.evaluation'])



# manage visualizer
VISUALIZERS = Registry(
    'visualizer',
    parent=MMENGINE_VISUALIZERS,
    locations=['mmdet.visualization'])
# manage visualizer backend
VISBACKENDS = Registry(
    'vis_backend',
    parent=MMENGINE_VISBACKENDS,
    locations=['mmdet.visualization'])

# manage logprocessor
LOG_PROCESSORS = Registry(
    'log_processor',
    parent=MMENGINE_LOG_PROCESSORS,
    # TODO: update the location when mmdet has its own log processor
    locations=['mmdet.engine'])
