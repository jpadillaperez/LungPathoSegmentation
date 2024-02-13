import configargparse
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateFinder
from lightning.pytorch.trainer import Trainer, seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from utils.utils import argparse_summary, get_class_by_path
from utils.configargparse_arguments import build_configargparser
from datetime import datetime
from types import SimpleNamespace
import torch.multiprocessing
import warnings
import torch
import wandb
import yaml
import os

print('Device name {} with memory {:.2f} GB'.format(torch.cuda.get_device_name(torch.cuda.current_device()), torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory/1e9))
print('Number of CPU threads: {}'.format(torch.get_num_threads()))

torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision('medium')
torch.set_grad_enabled(True)
warnings.filterwarnings("ignore", category=UserWarning)


def train():
    """
    Main training routine specific for this project
    """
    # ------------------------
    # 1 INIT WANDB
    # ------------------------
    root_path = Path(__file__).parent
    with open(os.path.join(root_path, "config.yaml"), "r") as f:
        hparams = yaml.safe_load(f)

    wandb.init(project="Pathology-FCDenseNet", config=hparams)
    wandb_logger = WandbLogger(project="Pathology-FCDenseNet", config=hparams)

    # ------------------------
    # 2 PARSE ARGS
    # ------------------------
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser, hparams = build_configargparser(parser)

    ModuleClass = get_class_by_path(hparams.module)
    parser = ModuleClass.add_module_specific_args(parser)

    ModelClass = get_class_by_path(hparams.model)
    parser = ModelClass.add_model_specific_args(parser)

    DatasetClass = get_class_by_path(hparams.dataset)
    parser = DatasetClass.add_dataset_specific_args(parser)

    hparams = parser.parse_args()
    hparams.name = datetime.now().strftime("%y%m%d-%H%M%S_") + hparams.module.split(".")[-1] + "_" + hparams.dataset.split(".")[-1] + "_" + hparams.model.replace(".", "_")
    hparams.output_path = Path(hparams.output_path).absolute() / hparams.name
    
    argparse_summary(hparams, parser)
    seed_everything(hparams.seed)

    # ------------------------
    # 3 LOAD MODEL, DATASET, MODULE
    # ------------------------
    print('In train.py: Loading model...')
    model = ModelClass(hparams=hparams)
    print('...done.')

    print('In train.py: Loading dataset...')
    dataset = DatasetClass(hparams=hparams)
    print('...done.')

    print('In train.py: Loading module...')
    module = ModuleClass(hparams, model)
    print('...done.')

    # ------------------------
    # 4 CALLBACKS
    # ------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.output_path,
        save_top_k=hparams.save_top_k,
        save_last=True,
        verbose=True,
        monitor=hparams.checkpoint_metric,
        mode=hparams.checkpoint_metric_mode,
        filename=f'{{epoch}}-{{{hparams.checkpoint_metric}:.2f}}'
    )

    trainer = Trainer(
        num_nodes=1,
        logger=wandb_logger,
        fast_dev_run=False,
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        enable_checkpointing=hparams.resume_from_checkpoint,
        callbacks=[checkpoint_callback],
        #weights_summary='full',
        deterministic=True,
        num_sanity_val_steps=hparams.num_sanity_val_steps,
        log_every_n_steps=hparams.log_every_n_steps,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        limit_train_batches=hparams.limit_train_batches,
        limit_val_batches=hparams.limit_val_batches,
        limit_test_batches=hparams.limit_test_batches
    )
    # ------------------------
    # 5 START TRAINING
    # ------------------------
    print('Starting training...')
    trainer.fit(module, dataset)



if __name__ == "__main__":
    root_path = Path(__file__).parent

    with open(os.path.join(root_path, "config.yaml"), "r") as f:
        hparams = yaml.safe_load(f)

    if hparams["mode"] == "train":
        train()

    elif hparams["mode"] == "sweep":
        with open(os.path.join(root_path, "sweep_config.yaml"), "r") as f:
            sweep_config = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep=sweep_config, project="Pathology-FCDenseNet")
        wandb.agent(sweep_id, function=train)


