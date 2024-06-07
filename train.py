import configargparse
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateFinder
from lightning.pytorch.trainer import Trainer, seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from utils.utils import get_class_by_path, wandb_config_summary
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
#torch.multiprocessing.set_sharing_strategy('file_system')
#torch.set_float32_matmul_precision('medium')
torch.use_deterministic_algorithms(mode=True, warn_only=True)
#torch.set_grad_enabled(True)
warnings.filterwarnings("ignore", category=UserWarning)


def train():
    """
    Main training routine specific for this project
    """
    # ------------------------
    # 1 INIT WANDB
    # ------------------------
    # Initialize wandb
    wandb.init(project="Pathology-FCDenseNet")

    # Load hparams from config file
    root_path = Path(__file__).parent
    with open(os.path.join(root_path, "config.yaml"), "r") as f:
        hparams = yaml.safe_load(f)
    hparams["name"] = datetime.now().strftime("%y%m%d-%H%M%S_") + hparams["module"].split(".")[-1] + "_" + hparams["dataset"].split(".")[-1] + "_" + hparams["model"].replace(".", "_")
    hparams["output_path"] = Path(hparams["output_path"]).absolute() / hparams["name"]
    
    if not os.path.exists(hparams["output_path"]):
        os.makedirs(hparams["output_path"])

    for key, value in wandb.config.items():
        hparams[key] = value
        print(f"Using {key} with value {value} from sweep")
    
    wandb_logger = WandbLogger(project="Pathology-FCDenseNet", config=hparams)
    wandb_config_summary(hparams)
    seed_everything(hparams["seed"])

    #Copy current config file to output path
    with open(os.path.join(hparams["output_path"], "config.yaml"), "w") as f:
        yaml.dump(hparams, f)

    # ------------------------
    # 2 LOAD MODEL, DATASET, MODULE
    # ------------------------
    print('In train.py: Loading model...')
    ModelClass = get_class_by_path(hparams["model"])
    model = ModelClass(hparams=hparams)
    print('...done.')

    print('In train.py: Loading dataset...')
    DatasetClass = get_class_by_path(hparams["dataset"])
    dataset = DatasetClass(hparams=hparams)
    print('...done.')

    print('In train.py: Loading module...')
    ModuleClass = get_class_by_path(hparams["module"])
    module = ModuleClass(hparams, model)
    print('...done.')

    # ------------------------
    # 3 CALLBACKS
    # ------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams["output_path"],
        save_top_k=hparams["save_top_k"],
        save_last=True,
        verbose=True,
        monitor=hparams["checkpoint_metric"],
        mode=hparams["checkpoint_metric_mode"],
        filename=f'{{epoch}}-{{{hparams["checkpoint_metric"]}:.2f}}'
    )

    trainer = Trainer(
        num_nodes=1,
        logger=wandb_logger,
        fast_dev_run=False,
        min_epochs=hparams["min_epochs"],
        max_epochs=hparams["max_epochs"],
        callbacks=[checkpoint_callback],
        #weights_summary='full',
        #deterministic=True,
        num_sanity_val_steps=hparams["num_sanity_val_steps"],
        log_every_n_steps=hparams["log_every_n_steps"],
        check_val_every_n_epoch=hparams["check_val_every_n_epoch"],
        limit_train_batches=hparams["limit_train_batches"],
        limit_val_batches=hparams["limit_val_batches"],
        limit_test_batches=hparams["limit_test_batches"]
    )
    # ------------------------
    # 4 START TRAINING
    # ------------------------
    print('Starting training...')
    if hparams["resume_from_checkpoint"]:
        trainer.fit(module, dataset, ckpt_path=hparams["resume_from_checkpoint"])
    else:
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

    elif hparams["mode"] == "test":
        raise NotImplementedError("Test mode not implemented yet")

    else:
        raise ValueError("Mode not recognized. Please use 'train' or 'sweep'")


