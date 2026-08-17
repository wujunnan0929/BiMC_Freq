from yacs.config import CfgNode as CN
from utils.util import set_gpu, set_seed
import argparse

def print_args(cfg):
    print("************")
    print("** Config **")
    print("************")
    print(cfg)
    print("************")


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """

    # Device setting
    cfg.DEVICE = CN()
    cfg.DEVICE.DEVICE_NAME = ''
    cfg.DEVICE.GPU_ID = ''

    cfg.METHOD = ''
    cfg.SEED = -1

    # For dataset config
    cfg.DATASET = CN()
    cfg.DATASET.NAME = ''
    cfg.DATASET.ROOT = ''
    cfg.DATASET.GPT_PATH = ''
    cfg.DATASET.NUM_CLASSES   = -1
    cfg.DATASET.NUM_INIT_CLS  = -1
    cfg.DATASET.NUM_INC_CLS   = -1
    cfg.DATASET.NUM_BASE_SHOT = -1
    cfg.DATASET.NUM_INC_SHOT  = -1
    cfg.DATASET.BETA = -1.0
    cfg.DATASET.ENSEMBLE_ALPHA = -1.0
    
    # For data
    cfg.DATALOADER = CN()
    cfg.DATALOADER.TRAIN = CN()
    cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE = -1
    cfg.DATALOADER.TRAIN.BATCH_SIZE_INC = -1
    cfg.DATALOADER.TEST = CN()
    cfg.DATALOADER.TEST.BATCH_SIZE = -1
    cfg.DATALOADER.NUM_WORKERS = -1

    # For model
    cfg.MODEL = CN()
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = ''

    # For methods
    cfg.TRAINER = CN()
    cfg.TRAINER.BiMC = CN()
    cfg.TRAINER.BiMC.PREC = ''
    cfg.TRAINER.BiMC.VISION_CALIBRATION = False
    cfg.TRAINER.BiMC.LAMBDA_I = -1.0
    cfg.TRAINER.BiMC.TAU = -1
    cfg.TRAINER.BiMC.TEXT_CALIBRATION = False
    cfg.TRAINER.BiMC.LAMBDA_T = -1.0
    cfg.TRAINER.BiMC.GAMMA_BASE = -1.0
    cfg.TRAINER.BiMC.GAMMA_INC = -1.0
    cfg.TRAINER.BiMC.USING_ENSEMBLE = False

    # Training-free frequency-aware prototype calibration. The default is off
    # so existing BiMC configuration files reproduce the original method.
    cfg.TRAINER.BiMC.FREQUENCY = CN()
    cfg.TRAINER.BiMC.FREQUENCY.ENABLED = False
    cfg.TRAINER.BiMC.FREQUENCY.LOW_CUTOFF = 0.18
    cfg.TRAINER.BiMC.FREQUENCY.HIGH_CUTOFF = 0.45
    cfg.TRAINER.BiMC.FREQUENCY.CENTER_RESIDUAL_BANDS = True
    cfg.TRAINER.BiMC.FREQUENCY.FFT_BATCH_SIZE = 8
    cfg.TRAINER.BiMC.FREQUENCY.FFT_DEVICE = "auto"
    cfg.TRAINER.BiMC.FREQUENCY.VIEW_MODE = "disjoint"
    cfg.TRAINER.BiMC.FREQUENCY.HIGH_ENHANCE = 0.5
    cfg.TRAINER.BiMC.FREQUENCY.NOVEL_VISION_CALIBRATION = True
    cfg.TRAINER.BiMC.FREQUENCY.SEMANTIC_WEIGHT = 0.25
    cfg.TRAINER.BiMC.FREQUENCY.MAX_SEMANTIC_WEIGHT = 0.65
    cfg.TRAINER.BiMC.FREQUENCY.SEMANTIC_GATE_MODE = "amplify"
    cfg.TRAINER.BiMC.FREQUENCY.DESCRIPTION_WEIGHT = 0.5
    cfg.TRAINER.BiMC.FREQUENCY.USE_EXPLICIT_DESCRIPTIONS = False
    cfg.TRAINER.BiMC.FREQUENCY.EXPLICIT_DESCRIPTION_PATH = ""
    cfg.TRAINER.BiMC.FREQUENCY.DESCRIPTION_TOPK = 3
    cfg.TRAINER.BiMC.FREQUENCY.DESCRIPTION_TEMPERATURE = 0.07
    cfg.TRAINER.BiMC.FREQUENCY.UNCERTAINTY_SCALE = 2.0
    cfg.TRAINER.BiMC.FREQUENCY.ALIGNMENT_SCALE = 4.0
    cfg.TRAINER.BiMC.FREQUENCY.FUSION_TEMPERATURE = 1.0
    cfg.TRAINER.BiMC.FREQUENCY.ADAPTIVE_FUSION = True
    cfg.TRAINER.BiMC.FREQUENCY.BAND_PRIOR = [1.0, 1.0, 1.0]
    cfg.TRAINER.BiMC.FREQUENCY.FREQ_ALPHA = 0.35
    cfg.TRAINER.BiMC.FREQUENCY.RELIABILITY_ALPHA = False
    cfg.TRAINER.BiMC.FREQUENCY.MIN_FREQ_ALPHA = 0.0
    cfg.TRAINER.BiMC.FREQUENCY.RELIABILITY_UNCERTAINTY_SCALE = 2.0
    cfg.TRAINER.BiMC.FREQUENCY.RELIABILITY_SHOT_TAU = 5.0
    cfg.TRAINER.BiMC.FREQUENCY.RELIABILITY_POWER = 1.0
    cfg.TRAINER.BiMC.FREQUENCY.PROMPTS = [
        "a photo of a {}, emphasizing its global shape, silhouette, and coarse spatial layout.",
        "a photo of a {}, emphasizing its parts, spatial structure, and medium-scale patterns.",
        "a photo of a {}, emphasizing its fine texture, edges, colors, and local details.",
    ]
    cfg.TRAINER.BiMC.FREQUENCY.LOW_KEYWORDS = [
        "shape", "silhouette", "overall", "body", "size", "large", "small", "long", "round",
    ]
    cfg.TRAINER.BiMC.FREQUENCY.MIDDLE_KEYWORDS = [
        "part", "head", "wing", "tail", "leg", "beak", "spatial", "structure",
    ]
    cfg.TRAINER.BiMC.FREQUENCY.HIGH_KEYWORDS = [
        "texture", "pattern", "stripe", "spot", "color", "edge", "feather", "fur", "detail",
    ]



    

def setup_cfg(dataset_cfg_file, method_cfg_file, opts=None):
    cfg = CN()
    extend_cfg(cfg)

    # 1. From the dataset config file
    cfg.merge_from_file(dataset_cfg_file)

    # 2. From the method config file
    cfg.merge_from_file(method_cfg_file)

    # 3. Optional command-line overrides for controlled ablations
    if opts:
        cfg.merge_from_list(opts)

    cfg.freeze()
    return cfg


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run the pipeline")

    parser.add_argument('--data_cfg', type=str, help="Path to the data configuration file")
    parser.add_argument('--train_cfg', type=str, help="Path to the training configuration file")
    parser.add_argument(
        '--opts',
        default=None,
        nargs=argparse.REMAINDER,
        help="Override config values, e.g. TRAINER.BiMC.FREQUENCY.VIEW_MODE natural",
    )

    args = parser.parse_args()

    data_cfg = args.data_cfg
    train_cfg = args.train_cfg

    cfg = setup_cfg(data_cfg, train_cfg, args.opts)

    # Set the random seed and GPU ID
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)

    # Import and run the trainer
    from engine.engine import Runner
    engine = Runner(cfg)
    engine.run()


if __name__ == '__main__':
    main()
