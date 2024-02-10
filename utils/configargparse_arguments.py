import configargparse


def build_configargparser(parser):
    model_group = parser.add_argument_group(title='Model options')
    dataset_group = parser.add_argument_group(title='Dataset options')
    module_group = parser.add_argument_group(title='Module options')
    trainer_group = parser.add_argument_group(title='Trainer options')

    # gpu args
    trainer_group.add_argument('--gpus', type=int, nargs='+', default=0, help='how many gpus / -1 means all')
    trainer_group.add_argument(
        '--cpus', type=int, nargs='+', default=0, help='how many cpus / -1 means all')

    trainer_group.add_argument('--accelerator', type=str, default='ddp', 
                               help='supports four options dp, ddp, ddp_spawn, ddp2')
    trainer_group.add_argument('--resume_from_checkpoint', type=str, default=None,
                               help='resume training from a checkpoint whose path is specified here (default: None)')
    trainer_group.add_argument('--log_every_n_steps', type=int, default=50,
                               help='how often to log within n steps (default: 50)')
    trainer_group.add_argument('--run_test', type=bool, default=False,
                               help='whether to run test or not (default: False)')

    trainer_group.add_argument('--seed', type=int, default=2334)
    trainer_group.add_argument('--pixel_threshold', type=float, default=0.025)

    # config module e.g. classification_multiclass
    module_group.add_argument('--module', type=str, required=True)

    # config model e.g. densenet.DenseNet121
    model_group.add_argument('--model', type=str, required=True)
    model_group.add_argument('--longitudinal', type=bool, default=False)

    module_group.add_argument('--pretraining_path', type=str, default="", required=False)

    # config dataset e.g. ham10k.HAM10k
    dataset_group.add_argument('--data_root', type=str, default='', required=True)
    dataset_group.add_argument('--dataset', type=str, required=True) ## Nicht lightning
    
    # config trainer
    trainer_group.add_argument('--overfit_batches', type=float, default=0.0,
                               help='overfit a percentage of trainign data (default: 0.0)')
    trainer_group.add_argument('--num_workers', type=int, default=8,
                               help='set the number of workers to be used on your machine (default: 8)')
    trainer_group.add_argument('--learning_rate', type=float, default=0.001,
                               help='learning rate for training (default: 0.001)')
    trainer_group.add_argument('--batch_size', type=int, default=32, help='batch size for DataLoader (default: 32')
    # log root
    trainer_group.add_argument('--num_sanity_val_steps', default=5, type=int,
                               help='number of validation sanity steps to be run before training (default: 5)')
    # max and min epoch
    trainer_group.add_argument('--max_epochs', type=int, default=1000,
                               help='limit training to a maximum number of epochs (default: 1000)')
    trainer_group.add_argument('--min_epochs', type=int, default=1,
                               help='force training to a minimum number of epochs (default: 1')
    # check logging frequency
    trainer_group.add_argument('--check_val_every_n_epoch', type=int, default=1,
                               help='check val every n train epochs (default: 1)')
    trainer_group.add_argument('--save_top_k', type=int, default=1,
                               help='save the best k models. -1: save all models (default: 1)')
    trainer_group.add_argument('--early_stopping_metric', type=str, default='val_loss',
                               help='monitor a validation metric and stop the training when no improvement is observed '
                                    '(default: val_loss)')
    trainer_group.add_argument('--checkpoint_metric', type=str, default='val_loss')

    # logging
    trainer_group.add_argument('--log_save_interval', type=int, default=100,
                               help='write logs to disk this often (default: 100)')
    trainer_group.add_argument('--row_log_interval', type=int, default=100,
                               help='how often to add logging rows (does not write to disk) (default: 100)')

    trainer_group.add_argument('--fast_dev_run', type=bool, default=False,
                               help='run one training and one validation batch (default: False)')
    trainer_group.add_argument('--name', type=str, default=None)
    trainer_group.add_argument('--on_polyaxon', action='store_true')
    trainer_group.add_argument('--output_path', type=str, default='logs')

    trainer_group.add_argument('--limit_val_batches', type=int, default=1.0)
    trainer_group.add_argument('--limit_test_batches', type=int, default=1.0)
    trainer_group.add_argument('--limit_train_batches', type=int, default=1.0)


    known_args, _ = parser.parse_known_args()
    return parser, known_args