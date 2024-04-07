import torch
import matplotlib.pyplot as plt
import wandb
from matplotlib import cm
import matplotlib as mpl

# 3D case: display all samples i.e. all volumes
def display_3d_images(logger, label_list, current_epoch, batch_idx, x, y_pred, y_true, vol_idx, phase='', epoch=False):
    no_labels = y_true.size()[1]
    if no_labels == len(label_list):
        cmap = cm.tab10
        cmaplist = [cmap(i) for i in range(10)]
        cmaplist.append((0, 0, 0, 1))

        # create the new map
        cmap1 = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', [cmaplist[index] for index in [10, 0]], 2)
        cmap2 = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', [cmaplist[index] for index in [10, 1]], 2)
        cmap3 = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', [cmaplist[index] for index in [10, 2]], 2)
        cmap4 = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', [cmaplist[index] for index in [10, 3]], 2)
        cmap5 = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', [cmaplist[index] for index in [10, 4]], 2)
        cmap6 = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', [cmaplist[index] for index in [10, 5]], 2)

        cmap_list = [cmap1, cmap2, cmap3, cmap4, cmap5, cmap6]
        
    plt.clf()
    for sample in range(x.size()[0]):
        figure, axs = plt.subplots(2, no_labels + 1, figsize=(12, 4))
        # start with original CT
        axs[0, 0].imshow(torch.rot90(x[sample, 0, :, :, x.size()[3] // 2], 1, [0, 1]), cmap='gray')
        axs[0, 0].set_title('Original CT', fontsize=8)
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 0].set_ylabel('CT')
        axs[0, 0].grid(False)
        axs[1, 0].imshow(torch.rot90(x[sample, 0, :, :, x.size()[3] // 2], 1, [0, 1]), cmap='gray')
        axs[1, 0].axis('off')
        axs[1, 1].set_ylabel('Predictions')
        axs[0, 1].set_ylabel('Ground Truths')
        for i in range(no_labels):
            # Plot ground truth
            axs[0, i + 1].imshow(torch.rot90(y_true[sample, i, :, :, x.size()[3] // 2], 1, [0, 1]), cmap=cmap_list[i])
            axs[0, i + 1].set_title(f'Label {i} ({label_list[i]})\nGround Truth ', fontsize=8)
            axs[0, i + 1].set_xticks([])
            axs[0, i + 1].set_yticks([])
            axs[0, i + 1].grid(False)
            # plot prediction
            axs[1, i + 1].imshow(torch.rot90(torch.ge(y_pred, 0.5).float()[sample, i, :, :, x.size()[3] // 2], 1, [0, 1]),
                                    cmap=cmap_list[i])
            axs[1, i + 1].set_title(f'Label {i} ({label_list[i]})\nPrediction', fontsize=8)
            axs[1, i + 1].set_xticks([])
            axs[1, i + 1].set_yticks([])
            axs[1, i + 1].grid(False)
        figure.tight_layout()
        if epoch:
            logger.experiment.log({f'{phase} phase, Epoch: {current_epoch}, Batch: {batch_idx}, '
                                      f'Sample {sample}, Volume ID {vol_idx[sample]}': wandb.Image(figure)})
        else:
            logger.experiment.log({f'{phase} phase, Batch: {batch_idx}, '
                                      f'Sample {sample}, Volume ID {vol_idx[sample]}': wandb.Image(figure)})
        plt.close(figure)

        

# 2D case: display all samples i.e. all slices
def display_2d_images(logger, label_list, current_epoch, batch_idx, x, y_pred, y_true, vol_idx, phase='', epoch=False):
    no_labels = y_true.size()[1]
    if no_labels == len(label_list):
        cmap = cm.tab10  # define the colormap
        cmaplist = [cmap(i) for i in range(10)]
        cmaplist.append((0, 0, 0, 1))

        # create the new map
        cmap1 = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', [cmaplist[index] for index in [10, 0]], 2)
        cmap2 = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', [cmaplist[index] for index in [10, 1]], 2)
        cmap3 = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', [cmaplist[index] for index in [10, 2]], 2)
        cmap4 = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', [cmaplist[index] for index in [10, 3]], 2)
        cmap5 = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', [cmaplist[index] for index in [10, 4]], 2)
        cmap6 = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', [cmaplist[index] for index in [10, 5]], 2)

        cmap_list = [cmap1, cmap2, cmap3, cmap4, cmap5, cmap6]

    plt.clf()
    for sample in range(x.size()[0]):
        figure, axs = plt.subplots(2, no_labels + 1, figsize=(12, 4))
        # start with original CT
        axs[0, 0].imshow(torch.rot90(x[sample, 0, :, :], 1, [0, 1]), cmap='gray')
        axs[0, 0].set_title('Original CT', fontsize=8)
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 0].set_ylabel('CT')
        axs[0, 0].grid(False)
        axs[1, 0].imshow(torch.rot90(x[sample, 0, :, :], 1, [0, 1]), cmap='gray')
        axs[1, 0].axis('off')
        axs[1, 1].set_ylabel('Predictions')
        axs[0, 1].set_ylabel('Ground Truths')
        for i in range(no_labels):
            # Plot ground truth
            axs[0, i + 1].imshow(torch.rot90(y_true[sample, i, :, :], 1, [0, 1]), cmap=cmap_list[i])
            axs[0, i + 1].set_title(f'Label {i} ({label_list[i]})\nGround Truth ', fontsize=8)
            axs[0, i + 1].set_xticks([])
            axs[0, i + 1].set_yticks([])
            axs[0, i + 1].grid(False)
            # plot prediction
            axs[1, i + 1].imshow(torch.rot90(torch.ge(y_pred, 0.5).float()[sample, i, :, :], 1, [0, 1]),
                                 cmap=cmap_list[i])
            axs[1, i + 1].set_title(f'Label {i} ({label_list[i]})\nPrediction', fontsize=8)
            axs[1, i + 1].set_xticks([])
            axs[1, i + 1].set_yticks([])
            axs[1, i + 1].grid(False)
        figure.tight_layout()
        if epoch:
            logger.experiment.log({f'{phase} phase, Epoch: {current_epoch}, Batch: {batch_idx}, '
                                      f'Sample {sample}, Volume ID {vol_idx[sample]}': wandb.Image(figure)})
        else:
            logger.experiment.log({f'{phase} phase, Batch: {batch_idx}, '
                                      f'Sample {sample}, Volume ID {vol_idx[sample]}': wandb.Image(figure)})
        plt.close(figure)


# 2D case: display all samples i.e. all slices
def display_all_in_one(logger, current_epoch, batch_idx, x, x_ref, y_pred, y_true, vol_idx, phase='', epoch=False):
    y_pred_split = torch.split(y_pred, 1, dim=0)
    y_true_split = torch.split(y_true, 1, dim=0)
    for sample in range(x.size()[0]):
        figure, axs = plt.subplots(1, 4, figsize=(24, 8))
        # start with original CT
        axs[0].imshow(torch.rot90(x_ref[sample, 0, :, :], 1, [0, 1]), cmap='gray')
        axs[0].set_title('Reference scan', fontsize=8)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_ylabel('CT')
        axs[0].grid(False)
        axs[1].imshow(torch.rot90(x[sample, 0, :, :], 1, [0, 1]), cmap='gray')
        axs[1].set_title('Target scan', fontsize=8)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_ylabel('CT')
        axs[1].grid(False)
        pred = y_pred_split[sample].argmax(1)
        pred_rot = torch.rot90(pred, 1, [1, 2])
        pred_rot.int().cpu().detach().numpy()
        rgb_values = label_to_rgb()
        pred_imgs = [rgb_values[p] for p in pred_rot]
        y_true = y_true_split[sample].argmax(1)
        true_rot = torch.rot90(y_true, 1, [1, 2])
        true_rot.int().cpu().detach().numpy()
        rgb_values = label_to_rgb()
        true_imgs = [rgb_values[p] for p in true_rot]

        for pred_img in pred_imgs:
            axs[2].imshow(pred_img)
            axs[2].set_title('Prediction', fontsize=8)
            axs[2].set_xticks([])
            axs[2].set_yticks([])
            axs[2].grid(False)

        for true_img in true_imgs:
            axs[3].imshow(true_img)
            axs[3].set_title('Ground Truth', fontsize=8)
            axs[3].set_xticks([])
            axs[3].set_yticks([])
            axs[3].grid(False)
        figure.tight_layout()
        if epoch:
            logger.experiment.log({f'{phase} phase, Epoch: {current_epoch}, Batch: {batch_idx}, '
                                      f'Sample {sample}, Volume ID {vol_idx[sample]}': wandb.Image(figure)})
        else:
            logger.experiment.log({
                                         f'{phase} phase, Batch: {batch_idx}, 'f'Sample {sample}, Volume ID {vol_idx[sample]}': wandb.Image(
                                             figure)})

        plt.close(figure)


def display_long(logger, current_epoch, batch_idx, pat_id, x, x_ref, vol_idx, phase='', epoch=False):
    for sample in range(x.size()[0]):
        x_rot = torch.rot90(x[sample, 0], 1, [0, 1])
        x_ref_rot = torch.rot90(x_ref[sample, 0], 1, [0, 1])
        figure, axs = plt.subplots(1, 2, figsize=(12, 4))
        # start with original CT
        axs[0].imshow(x_ref_rot, cmap='gray')
        axs[0].set_title('CT (t=0)', fontsize=8)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].grid(False)
        axs[1].imshow(x_rot, cmap='gray')
        axs[1].set_title('CT (t=1)', fontsize=8)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].grid(False)

        figure.tight_layout()
        if epoch:
            logger.experiment.log(
                {f'{phase} phase, Epoch: {current_epoch}, Batch: {batch_idx}, Pat_ID: {pat_id[sample]}'
                 f'Sample {sample}, Volume ID {vol_idx[sample]}': wandb.Image(figure)})
        else:
            logger.experiment.log({f'{phase} phase, Batch: {batch_idx}, Pat_ID: {pat_id[sample]} '
                                      f'Sample {sample}, Volume ID {vol_idx[sample]}': wandb.Image(figure)})
        plt.close(figure)


def label_to_rgb(label_tensor=torch.Tensor([0, 1, 2, 3, 4]), cmap_name='tab10'):
    cmap_dict = {
        'tab10': [cm.tab10(i)[:3] for i in range(10)]
    }

    blue = cmap_dict['tab10'][0]
    orange = cmap_dict['tab10'][1]
    green = cmap_dict['tab10'][2]
    cmap_dict['tab10'][0] = (0, 0, 0)
    cmap_dict['tab10'][1] = blue
    cmap_dict['tab10'][2] = orange
    cmap_dict['tab10'][3] = green

    label_rgb = torch.zeros(
        [5, 3],
        dtype=label_tensor.dtype,
        device=label_tensor.device
    )

    for c in range(5):
        for i in range(3):
            label_rgb[c, i] = cmap_dict[cmap_name][c][i]

    return label_rgb.cpu().detach().numpy()
