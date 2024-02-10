#import pytorch_lightning as pl
import lightning as l
from torch import optim, nn
import torchmetrics
import torch
from monai import losses
from lightning import seed_everything
from utils.confusion import DiceMetric
from utils.visualization import display_2d_images, display_long, display_all_in_one
from utils.metric_helper import calculate_mean_pathology_dice, get_dice_scores, save_metrics, get_eval_means_per_patient
import numpy as np
import torch.nn.functional as F
from datasets.utils import init_weights


class Segmentation(l.LightningModule):
    def __init__(self, hparams, model):
        super(Segmentation, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = model
        #init_weights(self.model, 'xavier')



        # load pretrained weights
        if self.hparams.pretraining_path != "":
            self.pretraining_path = self.hparams.pretraining_path
            model_state_dict = model.state_dict()
            pretrained_weights = torch.load(self.pretraining_path)['state_dict']

            for a in model_state_dict:
                if a != 'decoder.finalConv.weight' and a != 'decoder.finalConv.bias':
                    model_state_dict[a] = pretrained_weights["model." + a]
            self.model.load_state_dict(model_state_dict)

        # model to cuda
        self.newdevice = torch.device("cuda" if torch.cuda.is_available() and hparams.gpus != 0 else "cpu")
        self.model.to(self.newdevice)

        # initialize metrics
        #CHANGED: self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_labels=4)
        self.val_accuracy = torchmetrics.Accuracy(
            task='multiclass', num_classes=4)
        self.test_accuracy = torchmetrics.Accuracy(
            task='multiclass', num_classes=4)

        self.softmax = nn.Softmax2d()
        self.diceloss = losses.DiceLoss()
        self.metrics = {'Diceval': DiceMetric(), 'Dicetest': DiceMetric()}
        self.label_list = ['background', 'healthy_lung', 'GGO', 'consolidation']
        self.dice_per_volume = {}

    def forward(self, x_list):
        x_ref = x_list[0].to(self.newdevice)
        x = x_list[1].to(self.newdevice)
        output = self.model.forward(x_ref, x, self.newdevice)
        return output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0) #, amsgrad=True)
        lr_scheduler = {'scheduler': optim.lr_scheduler.StepLR(
                                        optimizer,
                                        step_size=10,
                                        gamma=0.1
                                    )}
        return [optimizer], [lr_scheduler]

    def on_train_start(self):
        #set seed
        seed_everything(self.hparams.seed)

    def training_step(self, train_batch, batch_idx):
        # Note: the dynamic dataset dataloader returns a dict with keys (['image', 'label', 'pat_id', 'vol_idx'])
        # get x and x_ref
        images = train_batch['image']  # batch, volume, channel, size, size (for batch=2: 2, 2, 1, 300, 300)
        x = images.unsqueeze(dim=1).requires_grad_(True).to(self.newdevice)
        print("Size of x: ", x.size())

        vol_idx = train_batch['vol_idx']
        pat_id = train_batch['pat_id']

        # labels: batch, volume, channel, size, size (for batch=2 and volume=2: 2, 2, 4, 300, 300)
        # get y_true
        y_true = torch.squeeze(train_batch['label'][:, 1, :, :], dim=1).type(dtype=torch.long).to(self.newdevice)
        y_true_metric = y_true.type(torch.IntTensor).to(self.newdevice)
        y_true = y_true.unsqueeze(dim=1)

        print("Size of y_true: ", y_true.size())
        print("Unique values of y_true: ", torch.unique(y_true))

        y_pred = self.forward(x)
        

        # calculate loss
        y_pred = self.softmax(y_pred)
        loss = self.diceloss(torch.argmax(y_pred, dim=1), y_true.squeeze(dim=1)).requires_grad_(True)

        ## visualize
        #if batch_idx == 17 or batch_idx == 60 or batch_idx == 111 or batch_idx == 485:
        #    display_2d_images(self.logger, self.label_list, self.current_epoch, batch_idx, x.cpu(), y_pred.cpu(), y_true.cpu(), vol_idx, phase='Training', epoch=True)
        #    if self.current_epoch == 0:
        #         display_long(self.logger, self.current_epoch, batch_idx, pat_id, x.cpu(), x_ref.cpu(), vol_idx, phase='Training', epoch=True)

        #print the device of each tensor and if grad is enabled
        #print('x.device: ', x.device)
        #print('y_pred.device: ', y_pred.device)
        #print('y_true.device: ', y_true.device)
        #print('loss.device: ', loss.device)
        #print('x.requires_grad: ', x.requires_grad)
        #print('y_pred.requires_grad: ', y_pred.requires_grad)
        #print('y_true.requires_grad: ', y_true.requires_grad)
        #print('loss.requires_grad: ', loss.requires_grad)

        return {'loss': loss, 'y_true': y_true, 'y_pred': y_pred, 'y_true_metric': y_true_metric}

    def training_step_end(self, train_step_output):
        print('train_step_output: ', train_step_output)
        # metrics
        diceloss_ggo = self.diceloss(train_step_output['y_pred'][0, 2, :, :], train_step_output['y_true'][0, 2, :, :])
        diceloss_consolidation = self.diceloss(train_step_output['y_pred'][0, 3, :, :], train_step_output['y_true'][0, 3, :, :])
        diceloss_healthy = self.diceloss(train_step_output['y_pred'][0, 1, :, :], train_step_output['y_true'][0, 1, :, :])
        self.log('Train Dice Loss', train_step_output['loss'], on_step=True, on_epoch=True)
        self.log('Train DiceLoss GGO', diceloss_ggo, on_step=False, on_epoch=True)
        self.log('Train DiceLoss consolidation', diceloss_consolidation, on_step=False, on_epoch=True)
        self.log('Train DiceLoss healthy', diceloss_healthy, on_step=False, on_epoch=True)

    def validation_step(self, val_batch, batch_idx):
        # get x and x_ref
        images = val_batch['image']  # batch, volume, channel, size, size (for batch=2 and volume=2: 2, 2, 1, 300, 300)
        x = images.unsqueeze(dim=1)

        #vol_idx = val_batch['vol_idx']
        pat_id = val_batch['pat_id']

        # get y_true
        y_true = torch.squeeze(val_batch['label'][:, 1, :, :], dim=1).type(dtype=torch.long).to(self.newdevice)
        y_true_metric = y_true.type(torch.IntTensor).to(self.newdevice)
        y_true = y_true.unsqueeze(dim=1)

        #forward pass
        y_pred = self.forward(x)
        y_pred = self.softmax(y_pred)

        val_loss = self.diceloss(torch.argmax(y_pred, dim=1), y_true.squeeze(dim=1))

        # calculate patient dice
        if batch_idx == 0:
            self.dice_per_volume = {}
        split_y_pred = torch.split(y_pred , 1, dim=0)
        split_y_true = torch.split(val_batch['label'], 1, dim=0)


        for sample in range(x.size()[0]):
            dice = get_dice_scores(self.metrics, 'val', split_y_pred[sample].cpu(), split_y_true[sample].cpu(),
                                   self.label_list)
            id = pat_id[sample]
            if id in self.dice_per_volume.keys():
                self.dice_per_volume[id].append(dice)
            else:
                self.dice_per_volume[id] = [dice]

        # calculate mean pathology dice
        mean_pathology_dice = calculate_mean_pathology_dice(dice)

        # visualize
        #if batch_idx == 7 or batch_idx == 25 or batch_idx == 44 or batch_idx == 130 or batch_idx == 177 or batch_idx == 245:
        #    display_2d_images(self.logger, self.label_list, self.current_epoch, batch_idx, x.cpu(), y_pred.cpu(), y_true.cpu(), vol_idx, phase='Validation')

        return {'val_loss': val_loss, 'y_pred': y_pred.detach(), 'y_true': y_true.detach(), 'y_true_metric': y_true_metric.detach(), 'dice': dice,
                'mean_dice_pathologies': mean_pathology_dice}

    def on_validation_step_end(self, val_set_output):
        self.val_accuracy(val_set_output['y_pred'], val_set_output['y_true_metric'])
        self.log('Validation Dice Loss', val_set_output['val_loss'], on_step=False, on_epoch=True)
        self.log('Validation Accuracy', self.val_accuracy, on_step=False, on_epoch=True)
        self.log('Mean Pathology Dice', val_set_output['mean_dice_pathologies'], on_step=False, on_epoch=True)
        return {'dice': val_set_output['dice']}

    def on_validation_epoch_end(self):
        volume_means, volume_stds, dice_per_patient = get_eval_means_per_patient(self.dice_per_volume)
        mean_dice = np.mean(list(volume_means.values()))
        self.log('Val Mean Dice', mean_dice)
        self.log('Val Dice Background', volume_means['Diceval_background'])
        self.log('Val Dice Healthy lung', volume_means['Diceval_healthy_lung'])
        self.log('Val Dice GGO', volume_means['Diceval_GGO'])
        self.log('Val Dice Consolidation', volume_means['Diceval_consolidation'])
        self.log('Val Dice Background std', volume_stds['Diceval_background'])
        self.log('Val Dice Healthy lung std', volume_stds['Diceval_healthy_lung'])
        self.log('Val Dice GGO std', volume_stds['Diceval_GGO'])
        self.log('Val Dice Consolidation std', volume_stds['Diceval_consolidation'])
        save_metrics(self.logger, self.hparams.output_path, self.hparams.out_channels, self.current_epoch,
                     outputs=dice_per_patient, title='validation', boxplot=True, csv=True, formats=['svg'])


    def test_step(self, test_batch, batch_idx):
        # get x and x_ref
        images = test_batch['image']  # batch, volume, channel, size, size (for batch=2 and volume=2: 2, 2, 1, 300, 300)
        x_ref = images[:, 0, :, :, :]
        x = images[:, 1, :, :, :]

        vol_idx = test_batch['vol_idx']
        pat_id = test_batch['pat_id']

        # labels: batch, volume, channel, size, size (for batch=2 and volume=2: 2, 2, 4, 300, 300)
        # get y_true
        y_true = torch.squeeze(test_batch['label'][:, 1, :, :], dim=1).type(dtype=torch.long).to(self.newdevice)
        y_true_metric = y_true.type(torch.IntTensor).to(self.newdevice)

        #forward pass
        x_list = [x_ref, x]
        y_pred = self.forward(x_list)
        y_pred = self.softmax(y_pred)

        # calculate dice per patient
        if batch_idx == 0:
            self.dice_per_volume = {}
        split_y_pred = torch.split(y_pred, 1, dim=0)
        split_y_true = torch.split(y_true, 1, dim=0)
        for sample in range(x.size()[0]):
            dice = get_dice_scores(self.metrics, 'test', split_y_pred[sample].cpu(), split_y_true[sample].cpu(),
                                   self.label_list)
            id = pat_id[sample]
            if id in self.dice_per_volume.keys():
                self.dice_per_volume[id].append(dice)
            else:
                self.dice_per_volume[id] = [dice]

        #visualize
        if batch_idx == 63 or batch_idx == 967 or batch_idx == 838:
            display_long(self.logger, self.current_epoch, batch_idx, pat_id, x.cpu(), x_ref.cpu(), vol_idx, phase='Test', epoch=True)
            display_2d_images(self.logger, self.label_list, self.current_epoch, batch_idx, x.cpu(), y_pred.cpu(), y_true.cpu(), vol_idx, phase='Test')
            #display_all_in_one(self.logger, self.current_epoch, batch_idx, x.cpu(), x_ref.cpu(), y_pred.cpu(), y_true.cpu(), vol_idx, 'Test')
        return { 'y_true': y_true, 'y_pred': y_pred, 'y_true_metric': y_true_metric, 'dice': dice}



    def test_step_end(self, test_step_output):
        self.test_accuracy(test_step_output['y_pred'], test_step_output['y_true_metric'])
        self.log('Test Accuracy', self.test_accuracy, on_step=False, on_epoch=True)
        return {'dice': test_step_output['dice']}


    def on_test_epoch_end(self):
        volume_means, volume_stds, dice_per_patient = get_eval_means_per_patient(self.dice_per_volume)
        mean_dice = np.mean(list(volume_means.values()))
        mean_std = np.std(list(volume_means.values()))
        self.log('Test Mean Dice', mean_dice)
        self.log('Test Mean Std', mean_std)
        self.log('Test Dice Background', volume_means['Dicetest_background'])
        self.log('Test Dice healthy lung', volume_means['Dicetest_healthy_lung'])
        self.log('Test Dice GGO', volume_means['Dicetest_GGO'])
        self.log('Test Dice Consolidation', volume_means['Dicetest_consolidation'])
        self.log('Test Dice Background std', volume_stds['Dicetest_background'])
        self.log('Test Dice healthy lung std', volume_stds['Dicetest_healthy_lung'])
        self.log('Test Dice GGO std', volume_stds['Dicetest_GGO'])
        self.log('Test Dice Consolidation std', volume_stds['Dicetest_consolidation'])
        save_metrics(self.logger, self.hparams.output_path, self.hparams.out_channels, self.current_epoch, outputs=dice_per_patient, title='test', boxplot=True, csv=True, formats=['svg'])
        


    @staticmethod
    def add_module_specific_args(parser):
        # specific_args = get_argparser_group(title="Module options", parser=parser)
        return parser