import lightning as l
from torch import optim, nn
import torchmetrics
import torch
from monai.losses import DiceLoss
from lightning import seed_everything
from utils.confusion import DiceMetric
from utils.visualization import display_2d_images
from utils.metric_helper import calculate_mean_pathology_dice, get_dice_scores, save_metrics, get_eval_means_per_patient
import numpy as np
import torch.nn.functional as F
from datasets.utils import init_weights


class Segmentation2D(l.LightningModule):
    def __init__(self, hparams, model):
        super(Segmentation2D, self).__init__()
        self.save_hyperparameters(hparams) # Save hyperparameters in the hparams variable
        self.new_device = torch.device("cuda" if torch.cuda.is_available() and hparams.gpus != 0 else "cpu")
        self.model = model
        self.model.to(self.new_device)
        init_weights(self.model, 'xavier')
        self.loss = DiceLoss()
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=4)
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=4)
        self.metrics = {'Diceval': DiceMetric(), 'Dicetest': DiceMetric()}
        self.label_list = ['background', 'healthy_lung', 'GGO', 'consolidation']
        self.dice_per_volume = {}

    def forward(self, x):
        output = self.model.forward(x, self.new_device)
        return output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0)

        lr_scheduler = {'scheduler': optim.lr_scheduler.StepLR( optimizer, step_size=10, gamma=0.1)}

        return [optimizer], [lr_scheduler]

    def on_train_start(self):
        seed_everything(self.hparams.seed)

    def training_step(self, train_batch, batch_idx):
        #---- Get data ----#
        y_true = train_batch['label'].type(dtype=torch.long).to(self.new_device)
        x = train_batch['image'].unsqueeze(1).to(self.new_device)

        #---- Forward pass ----#
        y_pred = self.forward(x)
        y_pred = nn.Softmax2d()(y_pred)

        #---- Calculate loss ----#
        loss = self.loss(y_pred, y_true)

        #---- Log metrics ----#
        if self.hparams.log_train_images:
            if batch_idx == 17 or batch_idx == 60 or batch_idx == 111 or batch_idx == 485:
                display_2d_images(self.logger, self.label_list, self.current_epoch, batch_idx, x.cpu(), y_pred.cpu(), y_true.cpu(), train_batch['vol_idx'], phase='Training', epoch=True)

        diceloss_healthy = self.loss(y_pred[0, 1, :, :], y_true[0, 1, :, :])
        diceloss_ggo = self.loss(y_pred[0, 2, :, :], y_true[0, 2, :, :])
        diceloss_consolidation = self.loss(y_pred[0, 3, :, :], y_true[0, 3, :, :])
        
        self.log('Train Dice Loss', loss, on_step=False, on_epoch=True)
        self.log('Train DiceLoss GGO', diceloss_ggo, on_step=False, on_epoch=True)
        self.log('Train DiceLoss consolidation', diceloss_consolidation, on_step=False, on_epoch=True)
        self.log('Train DiceLoss healthy', diceloss_healthy, on_step=False, on_epoch=True)

        return {'loss': loss}

#-----------------VALIDATION-----------------#

    def validation_step(self, val_batch, batch_idx):
        #---- Get data ----#
        y_true = val_batch['label'].type(dtype=torch.long).to(self.new_device)
        x = val_batch['image'].unsqueeze(dim=1).requires_grad_(False).to(self.new_device)

        #---- Forward pass ----#
        y_pred = self.forward(x)
        y_pred = nn.Softmax2d()(y_pred)

        #---- Calculate loss ----#
        val_loss = self.loss(y_pred, y_true).requires_grad_(False)

        #---- Calculate dice per patient ----#
        if batch_idx == 0:
            self.dice_per_volume = {}
        split_y_pred = torch.split(y_pred , 1, dim=0)
        split_y_true = torch.split(val_batch['label'], 1, dim=0)
        for sample in range(x.size()[0]):
            dice = get_dice_scores(self.metrics, 'val', split_y_pred[sample].cpu(), split_y_true[sample].cpu(), self.label_list)
            id = val_batch['pat_id'][sample]
            if id in self.dice_per_volume.keys():
                self.dice_per_volume[id].append(dice)
            else:
                self.dice_per_volume[id] = [dice]

        #---- Visualize ----#
        if batch_idx == 25:
            display_2d_images(self.logger, self.label_list, self.current_epoch, batch_idx, x.cpu(), y_pred.cpu(), y_true.cpu(), val_batch['vol_idx'], phase='Validation')

        #---- Log metrics ----#
        self.log('Validation Dice Loss', val_loss.item(), on_step=False, on_epoch=True)
        self.log('Validation Accuracy', self.val_accuracy(y_pred, y_true.type(torch.IntTensor).to(self.new_device)), on_step=False, on_epoch=True)
        self.log('Mean Pathology Dice', calculate_mean_pathology_dice(dice), on_step=False, on_epoch=True)

        return {'val_loss': val_loss, 'dice': dice}


    def on_validation_epoch_end(self):

        volume_means, volume_stds, dice_per_patient = get_eval_means_per_patient(self.dice_per_volume)
        mean_dice = np.mean(list(volume_means.values()))
        self.log('Val Mean Dice', mean_dice)

        # Define a default value for missing keys
        default_value = np.nan

        # Safely log values, using the get method of dictionaries which returns a default value if the key is not found
        self.log('Val Dice Background', volume_means.get('Diceval_background', default_value))
        self.log('Val Dice Healthy lung', volume_means.get('Diceval_healthy_lung', default_value))
        self.log('Val Dice GGO', volume_means.get('Diceval_GGO', default_value))
        self.log('Val Dice Consolidation', volume_means.get('Diceval_consolidation', default_value))
        self.log('Val Dice Background std', volume_stds.get('Diceval_background', default_value))
        self.log('Val Dice Healthy lung std', volume_stds.get('Diceval_healthy_lung', default_value))
        self.log('Val Dice GGO std', volume_stds.get('Diceval_GGO', default_value))
        self.log('Val Dice Consolidation std', volume_stds.get('Diceval_consolidation', default_value))

        save_metrics(self.logger, self.hparams.output_path, self.hparams.out_channels, self.current_epoch,
                     outputs=dice_per_patient, title='validation', boxplot=True, csv=True, formats=['svg'])


#-----------------TEST-----------------#

    def test_step(self, test_batch, batch_idx):
        #---- Get data ----#
        label = test_batch['label'].type(dtype=torch.long).to(self.new_device)
        x = test_batch['image'].unsqueeze(1).to(self.new_device)

        #---- Forward pass ----#
        y_pred = self.forward(x)
        y_pred = nn.Softmax2d()(y_pred)

        #---- Calculate Dice ----#
        if batch_idx == 0:
            self.dice_per_volume = {}
        split_y_pred = torch.split(y_pred, 1, dim=0)
        split_y_true = torch.split(y_true, 1, dim=0)
        for sample in range(x.size()[0]):
            dice = get_dice_scores(self.metrics, 'test', split_y_pred[sample].cpu(), split_y_true[sample].cpu(),
                                   self.label_list)
            id = test_batch['pat_id'][sample]
            if id in self.dice_per_volume.keys():
                self.dice_per_volume[id].append(dice)
            else:
                self.dice_per_volume[id] = [dice]

        #---- Visualize ----#
        if batch_idx == 63 or batch_idx == 967 or batch_idx == 838:
            display_2d_images(self.logger, self.label_list, self.current_epoch, batch_idx, x.cpu(), y_pred.cpu(), y_true.cpu(), test_batch['vol_idx'], phase='Test')
        
        #---- Log metrics ----#
        self.log('Test Accuracy', self.test_accuracy(y_pred, y_true.type(torch.IntTensor).to(self.new_device)), on_step=False, on_epoch=True)

        return {'dice': dice}


    def on_test_epoch_end(self):
        volume_means, volume_stds, dice_per_patient = get_eval_means_per_patient(self.dice_per_volume)
        mean_dice = np.mean(list(volume_means.values()))
        mean_std = np.std(list(volume_means.values()))

        default_value = np.nan

        self.log('Test Mean Dice', mean_dice)
        self.log('Test Mean Std', mean_std)
        self.log('Test Dice Background', volume_means.get('Dicetest_background', default_value))
        self.log('Test Dice healthy lung', volume_means.get('Dicetest_healthy_lung', default_value))
        self.log('Test Dice GGO', volume_means.get('Dicetest_GGO', default_value))
        self.log('Test Dice Consolidation', volume_means.get('Dicetest_consolidation', default_value))
        self.log('Test Dice Background std', volume_stds.get('Dicetest_background', default_value))
        self.log('Test Dice healthy lung std', volume_stds.get('Dicetest_healthy_lung', default_value))
        self.log('Test Dice GGO std', volume_stds.get('Dicetest_GGO', default_value))
        self.log('Test Dice Consolidation std', volume_stds.get('Dicetest_consolidation', default_value))
        save_metrics(self.logger, self.hparams.output_path, self.hparams.out_channels, self.current_epoch, outputs=dice_per_patient, title='test', boxplot=True, csv=True, formats=['svg'])


    @staticmethod
    def add_module_specific_args(parser):
        #specific_args = get_argparser_group(title="Module options", parser=parser)
        return parser