import numpy as np
from collections import OrderedDict
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import seaborn as sns



def calculate_mean_pathology_dice(dice):
        mean_pathology_dice = 0
        list = ['Diceval_GGO', 'Diceval_consolidation']
        count = 0
        for i in list:
            if dice[i] is not None:
                mean_pathology_dice += dice[i]
                count += 1
        if mean_pathology_dice > 0 and count > 0:
            mean_pathology_dice = mean_pathology_dice/count
        return mean_pathology_dice


def get_eval_means(outputs):
        output_keys = outputs[0]['dice'].keys()
        means = OrderedDict({key: [] for key in output_keys})
        for output in outputs:
            for key in output_keys:
                means[key].append(output['dice'][key])
        for key in output_keys:
            valid_means = [v for v in means[key] if v is not None]
            means[key] = sum(valid_means) / len(valid_means) if len(valid_means) else None

        means = {k: v for k, v in means.items() if v is not None}
        return means


def get_eval_means_per_patient(dice_dict):
    volume_keys = dice_dict.keys()

    d_per_vol = {}
    std_per_vol = {}

    for vol_key in volume_keys:
        volume = dice_dict[vol_key]
        keys = volume[0].keys()
        means = OrderedDict({key: [] for key in keys})
        stds = OrderedDict({key: [] for key in keys})
        for output in volume:
            for key in keys:
                means[key].append(output[key])
                stds[key].append(output[key])
        for key in keys:
            valid_means = [v for v in means[key] if v is not None]
            stds[key] = np.std(valid_means) if len(valid_means) else None
            means[key] = sum(valid_means) / len(valid_means) if len(valid_means) else None
        means = {k: v for k, v in means.items() if v is not None}
        stds = {k: v for k, v in stds.items() if v is not None}
        d_per_vol[vol_key] = means
        std_per_vol[vol_key] = stds

    final_keys = []
    for key, value in d_per_vol.items():
        final_keys = value.keys()
        if len(final_keys) == 5:
            break
    final_means = OrderedDict({key: [] for key in final_keys})
    final_stds = OrderedDict({key: [] for key in final_keys})
    for output, value in d_per_vol.items():
        for key in final_keys:
            if key in value:
                final_means[key].append(value[key])
    for output, value in std_per_vol.items():
        for key in final_keys:
            if key in value:
                final_stds[key].append(value[key])
    for key in final_keys:
        valid_means = [v for v in final_means[key] if v is not None]
        valid_stds = [v for v in final_stds[key] if v is not None]
        final_means[key] = sum(valid_means) / len(valid_means) if len(valid_means) else None
        final_stds[key] = sum(valid_stds) / len(valid_stds) if len(valid_stds) else None
    final_means = {k: v for k, v in final_means.items() if v is not None}
    final_stds = {k: v for k, v in final_stds.items() if v is not None}
    return final_means, final_stds, d_per_vol


def get_eval_std(outputs):
        output_keys = outputs[0]['dice'].keys()
        stds = OrderedDict({key: [] for key in output_keys})
        for output in outputs:
            for key in output_keys:
                stds[key].append(output['dice'][key])
        for key in output_keys:
            valid_stds = [v for v in stds[key] if v is not None]
            stds[key] = np.std(valid_stds) if len(valid_stds) else None

        stds = {k: v for k, v in stds.items() if v is not None}
        return stds

def get_dice_scores(metrics, phase,  pred, target, labels):
            name = 'Dice' + phase
            metrics_dict = {}
            metric = metrics[name](pred, target)
            valid_metric = [m for m in metric if m is not None]
            mean_metric = sum(valid_metric) / len(valid_metric) if len(valid_metric) else None
            metrics_dict[name] = mean_metric.item()
            for i in range(len(metric)):
                if metric[i] is not None:
                    metrics_dict[f"{name}_{labels[i]}"] = metric[i].item()
                else:
                    metrics_dict[f"{name}_{labels[i]}"] = metric[i]
            return metrics_dict


def save_metrics(logger, output_path, out_channels, current_epoch, outputs, title, boxplot=True, csv=False, formats=['svg']):
        # create dataframe for convenience
        df = pd.DataFrame(outputs)

        epoch_str = '' if title != 'validation' else f'_epoch_{current_epoch:03}'

        if csv:
            # save metrics to csv
            csv_path = output_path / 'metrics'
            csv_path.mkdir(parents=True, exist_ok=True)

            df.to_csv(csv_path / f"{title}_metrics{epoch_str}.csv")
            df.describe().to_csv(csv_path / f"{title}_metrics_statistics{epoch_str}.csv")

        if boxplot:
            metric = 'dice'
            box_path = output_path / 'boxplots'
            box_path.mkdir(parents=True, exist_ok=True)
            # write boxplots to files and tensorboard
            plt.clf()
            #fig, ax = plt.subplots()
            sns.set(style='whitegrid')
            boxplot = sns.boxplot( data=df.T )
            #plt.show()
            boxplot.set_ylim([0., 1.])
            boxplot.set_xticklabels(['Mean', 'Background', 'Healthy lung', 'GGO', 'Consolidation'])
            boxplot.set(xlabel="Class", ylabel="Dice score")

            boxplot.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            fig = boxplot.get_figure()
            fig.set_size_inches((2. + 2. * out_channels, 7.))

            for format in formats:
                p = box_path / metric
                p.mkdir(exist_ok=True)
                fig.savefig(
                    p / f"{title}_boxplot_{metric}{epoch_str}.{format}"
                )

            logger.experiment.log({f'Dice Boxplot, {title} phase': wandb.Image(fig)})
