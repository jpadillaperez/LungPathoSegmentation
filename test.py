"""Taken from: TeCNO gitlab repo master branch"""
import configargparse
from pathlib import Path
from lightning.pytorch.trainer import Trainer, seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from utils.utils import argparse_summary, get_class_by_path
from utils.configargparse_arguments import build_configargparser
from datetime import datetime
import warnings
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore", category=UserWarning)


SEED = 2334


def test(hparams, ModuleClass, ModelClass, DatasetClass, logger):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    # load model
    print('In train.py: Loading model...')
    model = ModelClass(hparams=hparams)
    print('...done.')
    # load dataset
    print('In train.py: Loading dataset...')
    dataset = DatasetClass(hparams=hparams)
    print('...done.')
    # load module
    print('In train.py: Loading module...')
    module = ModuleClass(hparams, model)
    print('...done.')

    # ------------------------
    # 3 INIT TRAINER --> continues training
    # ------------------------

    trainer = Trainer(
        num_nodes=1,
        logger=logger,
        # fast_dev_run: if true, runs one training and one validation batch
        fast_dev_run=hparams.fast_dev_run,
        # min_epochs: forces training to a minimum number of epochs
        #min_epochs=hparams.min_epochs,
        # max_epochs: limits training to a maximum number of epochs
        #max_epochs=hparams.max_epochs,
        # saves the state of the last training epoch (all model parameters)
        #enable_checkpointing=True,
        #weights_summary='full',
        deterministic=True,
        #num_sanity_val_steps=0,
        log_every_n_steps=hparams.log_every_n_steps,
        # auto_lr_find: if true, will find a learning rate that optimizes initial learning for faster convergence
        #auto_lr_find=True,
        inference_mode=True,
        # auto_scale_batch_size: if true, will initially find the largest batch size that fits into memory
        #auto_scale_batch_size=True,
        #limit_train_batches=hparams.limit_train_batches,
        #limit_val_batches=hparams.limit_val_batches,
        #limit_test_batches=hparams.limit_test_batches
    )
    # ------------------------
    # 4 START Testing
    # ------------------------
    model_state_dict = module.model.state_dict()
    pretrained_weights = torch.load(
        "/data/COVID_longitudinal/25vs25/PMSD_SS23/Jorge/Models/CovidSeg/epoch=5-Val Mean Dice=0.65.ckpt")['state_dict']
   

    for a in model_state_dict:
        model_state_dict[a] = pretrained_weights["model." + a]
    module.model.load_state_dict(model_state_dict)

    #module.model.load_state_dict(pretrained_weights)

    #hparams.resume_from_checkpoint = "/mnt/polyaxon/outputs1/crogers/pyfl_coco/experiments/72365/checkpoints/epoch=5-Val Mean Dice=0.65.ckpt"

    #trainer.predict(module.model, dataset)
    trainer.test(module.model, dataset)#, ckpt_path=hparams.resume_from_checkpoint)


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = Path(__file__).parent
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)

    # each LightningModule defines arguments relevant to it
    # ------------------------
    # LOAD MODULE
    # ------------------------
    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)
    parser = ModuleClass.add_module_specific_args(parser)
    # ------------------------
    # LOAD MODEL
    # ------------------------
    model_path = f"models.{hparams.model}"
    ModelClass = get_class_by_path(model_path)
    parser = ModelClass.add_model_specific_args(parser)
    # ------------------------
    # LOAD DATASET
    # ------------------------
    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)
    parser = DatasetClass.add_dataset_specific_args(parser)
    # ------------------------
    #  PRINT PARAMS & INIT LOGGER
    # ------------------------
    hparams = parser.parse_args()
    # setup logging
    exp_name = (
        hparams.module.split(".")[-1]
        + "_"
        + hparams.dataset.split(".")[-1]
        + "_"
        + hparams.model.replace(".", "_")
    )

    date_str = datetime.now().strftime("%y%m%d-%H%M%S_")
    hparams.name = date_str + exp_name
    hparams.output_path = Path(
        hparams.output_path).absolute() / hparams.name
    project_name_prefix = ""

    wandb_logger = WandbLogger(name=hparams.name, project="test run")

    argparse_summary(hparams, parser)
    print('Output path: ', hparams.output_path)

    loggers = wandb_logger

    seed_everything(SEED)

    # ---------------------
    # RUN TEST
    # ---------------------
    
    test(hparams, ModuleClass, ModelClass, DatasetClass, loggers)
