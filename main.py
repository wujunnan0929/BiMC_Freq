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

    # Optional, training-free frequency contribution diagnostic.
    cfg.ANALYSIS = CN()
    cfg.ANALYSIS.FREQUENCY = CN()
    cfg.ANALYSIS.FREQUENCY.ENABLED = False
    cfg.ANALYSIS.FREQUENCY.ANALYSIS_ONLY = False
    cfg.ANALYSIS.FREQUENCY.OUTPUT_DIR = './outputs/frequency_analysis'
    cfg.ANALYSIS.FREQUENCY.BAND_NAMES = ['low', 'mid', 'high']
    cfg.ANALYSIS.FREQUENCY.BAND_EDGES = [0.0, 0.15, 0.35, 1.0]
    cfg.ANALYSIS.FREQUENCY.BAND_MODE = 'fixed'
    cfg.ANALYSIS.FREQUENCY.ENERGY_SAMPLES_PER_CLASS = 20
    cfg.ANALYSIS.FREQUENCY.SAMPLES_PER_CLASS = 5
    cfg.ANALYSIS.FREQUENCY.WEIGHT_TEMPERATURE = 0.05
    cfg.ANALYSIS.FREQUENCY.COMPETITOR_SCOPE = 'all'
    cfg.ANALYSIS.FREQUENCY.FFT_DEVICE = 'auto'
    cfg.ANALYSIS.FREQUENCY.PREDICTIVITY = CN()
    cfg.ANALYSIS.FREQUENCY.PREDICTIVITY.ENABLED = False
    cfg.ANALYSIS.FREQUENCY.PREDICTIVITY.EVAL_SCOPE = 'final'
    cfg.ANALYSIS.FREQUENCY.PREDICTIVITY.BAND = 'high'
    cfg.ANALYSIS.FREQUENCY.PREDICTIVITY.SOFT_GATE_TEMPERATURE = 0.01
    cfg.ANALYSIS.FREQUENCY.PREDICTIVITY.RELIABILITY_EPS = 1e-6



    

def setup_cfg(dataset_cfg_file, method_cfg_file, seed=None):
    cfg = CN()
    extend_cfg(cfg)

    # 1. From the dataset config file
    cfg.merge_from_file(dataset_cfg_file)

    # 2. From the method config file
    cfg.merge_from_file(method_cfg_file)

    if seed is not None:
        cfg.SEED = seed

    cfg.freeze()
    return cfg


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run the pipeline")

    parser.add_argument('--data_cfg', type=str, help="Path to the data configuration file")
    parser.add_argument('--train_cfg', type=str, help="Path to the training configuration file")
    parser.add_argument('--seed', type=int, default=None, help="Optional random seed override")

    args = parser.parse_args()

    data_cfg = args.data_cfg
    train_cfg = args.train_cfg

    cfg = setup_cfg(data_cfg, train_cfg, seed=args.seed)

    # Set the random seed and GPU ID
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)

    # Import and run the trainer
    from engine.engine import Runner
    engine = Runner(cfg)
    engine.run()


if __name__ == '__main__':
    main()
