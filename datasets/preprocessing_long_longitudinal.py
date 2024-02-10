"""
Adapted from
https://gitlab.lrz.de/CAMP_IFL/dynamic-dataset/dynamic-dataset/dynamic_dataset/datasets/dynamic/preprocessing_long.py
See https://gitlab.lrz.de/CAMP_IFL/dynamic-dataset/-/tree/master
on how to install and use the dynamic dataset.
"""

import os
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
import copy
from datasets.utils import (
    load_config_yaml,
    split_idxs,
    rm_tree,
    verify_config_hash,
    save_config_hash,
    crop_to_mask
)
import atexit
from sklearn.model_selection import StratifiedKFold, KFold
import nibabel as nib
import time


class DatasetPreprocessorLongitudinal:
    def __init__(
            self,
            config_yml_path,
            output_path,
            force_preprocessing=True,
            cleanup=False,
            verbose=True,
            seed=42,
            max_step=2,  # get all patients with max 2 scans
            min_step=2,  # get all patients with min 2 scans
            register_config="all"  # all #pf #fp #static
    ):

        # check inputs
        register_config_types = ['all', 'pf', 'fp', 'static']
        if register_config not in register_config_types:
            raise ValueError("Invalid register_config type. Expected one of: %s" % register_config_types)

        # load dataset config

        config_yml_path = Path(config_yml_path)
        assert config_yml_path.is_file(), f"config yaml could not be found at '{config_yml_path}', {os.curdir}"
        self.cfg = load_config_yaml(config_yml_path)
        self.verbose = verbose

        self.long_step = max_step
        self.register_config = register_config
        self.min_step = min_step

        if verbose:
            print(f"Loaded {self.cfg['name']} setup from {config_yml_path}", flush=True)

        # set data root dir
        # self.data_root = Path(data_root)
        self.pp_path = config_yml_path.parent
        self.output_path = Path(output_path)
        assert (
            self.pp_path.is_dir()
        ), f"preprocessed directory could not be found at '{self.pp_path}'"

        assert (
            self.output_path.is_dir()
        ), f"output directory could not be found at '{self.output_path}'"

        # load setup config
        setup_yml_path = self.pp_path / "setup.yml"
        assert Path(
            setup_yml_path
        ).is_file(), f"setup yaml could not be found at '{config_yml_path}'"
        self.setup = load_config_yaml(setup_yml_path)
        if verbose:
            print(f"Loaded {self.setup['name']} setup from {setup_yml_path}", flush=True)

        # set temporary dir for npy files and csv
        self.npy_path = self.output_path / ("npy_" + self.cfg["name"])
        self.tmp_df_path = self.npy_path / "tmp_df.csv"

        # setup cleanup
        if cleanup:
            atexit.register(self.cleanup)

        # load base patient dataframe
        self.df_path = self.pp_path / "base_df.csv"
        self.df = pd.read_csv(self.df_path)

        # get all scans with long_step smaller than max long_step
        self.df = self.df[self.df["long_step"] < self.long_step]

        if verbose:
            print(f"Dataframe loaded from {self.df_path}")

        if "drop_na_col" in self.cfg.keys():
            if self.cfg["drop_na_col"] is not None:
                df = self.df.dropna(subset=[self.cfg["drop_na_col"]])
                self.df = df.reset_index(drop=True)

        df = self.df[self.df[f"nii_{self.cfg['labelmap']}"] == True]
        self.df = df.reset_index(drop=True)

        counts = df["base_patient"].value_counts()
        # get patients with less then min_step scans
        patients_with_only_one_step = counts[counts<self.min_step]

        df = df[~df["base_patient"].isin(np.asarray(patients_with_only_one_step.index))]
        df.reset_index(drop=True, inplace=True)

        # check if patients are selected manually
        if "manual_split" in self.cfg.keys():
            print("Manual splits detected, stored in 'manual_split'")
            self.df["manual_split"] = None
            for split, pat_ids in self.cfg["manual_split"].items():
                for pat_id in pat_ids:
                    self.df.loc[
                        self.df[self.setup["id_col"]] == pat_id, "manual_split"
                    ] = split
            # select only volumes that have a split assigned
            self.df.dropna(subset=["manual_split"], inplace=True)

        # temporary file in npy folder to lock only execute pre-processing once

        lock_file = (self.npy_path / 'lock')
        if lock_file.is_file():
            print('found lock file - waiting until pre-processing is finished', end='')
            while lock_file.is_file():
                time.sleep(5)
                print("sleeping")

                print('.', end='')
            print(' continuing')
        # exit()
        print("done making lock file", flush=True)
        # check if temporary dir exists already

        if (
                self.npy_path.is_dir()
                and not force_preprocessing
                and self.tmp_df_path.is_file()
                and verify_config_hash(config_yml_path, self.npy_path)
        ):
            if verbose:
                print(
                    f"npy folder found at {self.npy_path}! (delete folder for new preprocessing or set force_preprocessing)"
                )
            print(f"{self.setup['name']} '{self.cfg['name']}' preprocessed data found")
            self.base_df = self.df
            self.df = pd.read_csv(self.tmp_df_path)

        else:

            try:
                self.npy_path.mkdir(exist_ok=force_preprocessing)
            except FileExistsError:
                print(
                    f"npy folder found at {self.npy_path}! (delete folder for new preprocessing or set force_preprocessing)"
                )
            # create lockfile
            lock_file.touch()

            # preprocess all data with npz files and safe npy
            print(f"Preprocessing {self.setup['name']} '{self.cfg['name']}'..")

            self._preprocess_all()

            # select only volumes that have been preprocessed
            df = self.df.dropna(subset=["dim0"])
            num_vol = len(df)
            num_pat = len(self.df)
            if num_vol < num_pat:
                print(
                    f"WARNING: only {num_vol} out of {num_pat} have been preprocessed. Dropping rest of entries.."
                )
            self.df = df.reset_index(drop=True)
            if 'manual_split' in self.cfg.keys():
                print('manual split found in config - skipping automatic splitting')
            else:
                if num_pat < 10:
                    print('less than 10 patients. 50-50 split in train and val')
                    test_size = 0
                    val_size = 0.5
                    train_size = 0.5
                else:
                    # SIMPLE TRAIN, VAL, TEST SPLITTING
                    test_size = self.cfg["test_size"]
                    val_size = self.cfg["val_size"]
                    train_size = 1 - val_size - test_size
                print(
                    f"Creating split 'train_val_test_split': {test_size:.0%} test, {val_size:.0%} val and {train_size:.0%} train"
                )
                splits = ["train", "val", "test"]
                idxs = np.arange(len(self.df))
                idxs_split = split_idxs(
                    idxs, test_size=test_size, val_size=val_size, seed=seed, shuffle=True
                )
                self.df["train_val_test_split"] = None
                for split in splits:
                    self.df.loc[idxs_split[split], "train_val_test_split"] = split

                # 5-FOLD-SPLIt
                if len(self.df) > 5:
                    stratify = self.cfg['stratify']
                    idxs = np.arange(len(self.df))
                    n_splits = 5

                    if stratify:
                        strat_label = np.zeros_like(idxs)
                        for i, label in enumerate(self.cfg['labels']):
                            strat_label += 2 ** i * (self.df[f"num_{label}"] > 0).to_numpy(dtype=int)
                        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                        for k, (train_idx, val_idx) in enumerate(skf.split(idxs, strat_label)):
                            split_col = f"split_{n_splits}fold_{k}"
                            self.df[split_col] = None
                            self.df.loc[train_idx, split_col] = "train"
                            self.df.loc[val_idx, split_col] = "val"
                        strat_print = ", stratified"
                    else:
                        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                        for k, (train_idx, val_idx) in enumerate(kf.split(idxs)):
                            split_col = f"split_{n_splits}fold_{k}"
                            self.df[split_col] = None
                            self.df.loc[train_idx, split_col] = "train"
                            self.df.loc[val_idx, split_col] = "val"
                        strat_print = ''

                    print(
                        f"Created k-fold cross validation split: 'split_{n_splits}fold_k' - {n_splits}-fold, shuffle, seed 42{strat_print} - splits: 'train', 'val'"
                    )

                else:
                    print('Omitting 5-fold-split due to limited number of volumes')

            # copy config and create hash
            new_cfg_path = self.npy_path / Path(config_yml_path).name
            new_cfg_path.write_text(Path(config_yml_path).read_text())
            save_config_hash(new_cfg_path, self.npy_path)

            # save temporary dataframe
            self.df.to_csv(self.tmp_df_path)

            # remove lockfile
            lock_file.unlink()

            if verbose:
                print(f"Temporary data has been extracted to {self.npy_path}")
                print("Successfully preprocessed data")

    def print_cfg(self):
        # print config
        print("\nConfiguration:")
        print(yaml.dump(self.cfg))

    def export_config(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path / 'patients.csv')
        (path / f'config_{self.cfg["name"]}.yml').write_text(yaml.dump(self.cfg))

    def cleanup(self):
        # cleaning up tmp dir
        rm_tree(self.npy_path)

    def _preprocess_all(self):
        """
        loop through all patients in df with npz files
        map channels and labels
        preprocess and save npy files

        """
        print("pre preprocessing started")

        cfg = self.cfg["preprocessing"]
        df = self.df

        # only 1 dim per volume
        for i in range(3):
            df[f"dim{i}"] = None
        for label in self.cfg["labels"]:
            df[f"num_{label}"] = 0
        df["split_patient"] = None
        # drop rows where selected labelmap is not present
        df.drop(
            df[
                df[f'nii_{self.cfg["labelmap"]}'] == False
            ].index, inplace=True)

        counts = df["base_patient"].value_counts()
        patients_with_only_one_step = counts[counts<self.min_step]

        df = df[~df["base_patient"].isin(np.asarray(patients_with_only_one_step.index))]
        df.reset_index(drop=True, inplace=True)
        self.df = df

        print(self.df["base_patient"].unique())

        for base_patient in tqdm(self.df["base_patient"].unique()):
            ## base patient
            patient_long = df[df["base_patient"] == base_patient].sort_values(by=['long_step'])
            pat_id_long, data_long, seg_long, mask_long = [], [], [], []
            for idx, pat in patient_long.iterrows():
                pat_id = pat["pat_id"]
                data = {}
                for channel in self.cfg['channels']:
                    p = self.pp_path / 'nii' / channel / f"{pat_id}_{channel}.nii.gz"
                    data = nib.load(p).get_fdata()

                # load seg
                labelmap = self.cfg["labelmap"]
                p = self.pp_path / 'nii' / labelmap / f"{pat_id}_{self.cfg['labelmap']}.nii.gz"
                seg = nib.casting.float_to_int(nib.load(p).get_fdata(), np.uint8)

                lung = (seg > 0).astype(int) #everything that is not 0

                # map to configured labels
                seg = self._remap_labels(seg, labelmap)

                label_counts = self._get_label_counts(seg)
                for k, v in label_counts.items():
                    self.df.loc[idx, f"num_{k}"] = np.array(v, dtype=np.int64)

                # perform preprocessing (only done once)
                if cfg["clip"]:
                    data = self._clip(data, low_thr=cfg["clip_low_thr"], up_thr=cfg["clip_up_thr"])

                cropped_data, cropped_seg, cropped_lung, mask = crop_to_mask(data, seg, lung)

                if cfg["crop"]:
                    data = cropped_data
                    seg = cropped_seg

                # normalize
                if cfg["normalize"]:
                    data = self._normalize(data, mask)

                pat_id_long.append(pat_id)
                data_long.append(data)
                seg_long.append(seg)
                mask_long.append(mask)

            combinations = []
            if len(patient_long) == 2:
                if self.register_config == "static":
                    combinations = [[0, 0], [0, 1]]
                else:
                    combinations = [[0, 1]]

            elif len(patient_long) == 3:

                if self.register_config == "static":
                    combinations = [[0, 0], [0, 1], [0, 2]]
                else:
                    combinations = [[0, 1], [1, 2], [0, 2]]

            elif len(patient_long) == 4:
                if self.register_config == "static":
                    combinations = [[0, 0], [0, 1], [0, 2], [0, 3]]
                else:
                    combinations = [[0, 1], [1, 2], [2, 3], [0, 2], [0, 3], [1, 3]]

            for selection in combinations:
                # past to future
                if self.register_config == "pf" or self.register_config == "all" or self.register_config == "static":
                    data_long_pf = copy.deepcopy([data_long[i] for i in selection])
                    seg_long_pf = copy.deepcopy([seg_long[i] for i in selection])
                    base_lung_pf = (seg_long_pf[1] > 0).astype(int)  # future
                    self.registration(base_patient, base_lung_pf, data_long_pf, seg_long_pf ,  f"{selection[0]}{selection[1]}")

                # future to past
                if self.register_config == "fp" or self.register_config == "all":
                    myorder = [1, 0]
                    data_long_fp = copy.deepcopy([data_long[i] for i in selection])
                    data_long_fp = [data_long_fp[i] for i in myorder]
                    seg_long_fp = copy.deepcopy([seg_long[i] for i in selection])
                    seg_long_fp = [seg_long_fp[i] for i in myorder]
                    base_lung_fp = (seg_long_fp[1] > 0).astype(int)  # past
                    self.registration(base_patient, base_lung_fp, data_long_fp, seg_long_fp, f"{selection[1]}{selection[0]}")

    def registration(self, base_patient, base_lung,  data_long, seg_long, save_identification):
            moving_lung = (seg_long[0] > 0).astype(int)
            transformation = deformable_registration(base_lung, moving_lung, self.setup["resolution"])
            seg_long[0] = transform_moving_volume(seg_long[1], seg_long[0], transformation,
                            self.setup["resolution"], method=sitk.sitkNearestNeighbor)
            data_long[0] = transform_moving_volume(data_long[1], data_long[0], transformation,
                            self.setup["resolution"], method=sitk.sitkLinear)

            data_long[0] = np.expand_dims(data_long[0], axis=0)
            data_long[1] = np.expand_dims(data_long[1], axis=0)

            data = np.stack(data_long, axis=0)
            seg = np.stack(seg_long, axis=0)

            print(f"{base_patient} - data: {data.shape} - seg: {seg.shape}")

            if save_identification != "01":
                self.df = self.df.append({'pat_id': f"{base_patient}_{save_identification}",
                                          'base_patient': f"{base_patient}_{save_identification}",
                                          'split_patient': f"{base_patient}"},
                                         ignore_index=True)
                for i in range(3):
                    self.df.loc[(self.df["base_patient"] == f"{base_patient}_{save_identification}"), f"dim{i}"] = data[0].shape[i + 1]
            else:
                self.df.loc[(self.df["base_patient"] == base_patient), "split_patient"] = f"{base_patient}"
                # save number of layers to df
                for i in range(3):
                    self.df.loc[(self.df["base_patient"] == base_patient), f"dim{i}"] = data[0].shape[i + 1]

            self._save_as_npy(base_patient, data, seg, save_identification)


    def _get_label_counts(self, seg):
        counts = {}
        for c, label in enumerate(self.cfg["labels"]):
            counts[label] = (seg == c).sum()

        return counts

    def _remap_channels(self, data):
        """map selected modalities to input channels"""
        channels = self.cfg["channels"]
        new_data = []
        for c, modality in enumerate(channels):
            new_data.append(np.expand_dims(data[modality], axis=0))
        new_data = np.hstack(new_data)
        return new_data

    def _remap_labels(self, seg, labelmap):
        """"map selected labels to segmentation map values"""
        new_seg = np.zeros(seg.shape, dtype=seg.dtype)
        for new_label_value, label_name in enumerate(self.cfg["labels"]):
            label_value = self.setup["labels"][labelmap][label_name]
            new_seg[seg == label_value] = new_label_value
        return new_seg

    def _normalize(self, data, mask):
        """normalize grey values optionally taking into account non-zero maks"""
        data = data.astype(np.float32)
        cfg = self.cfg["preprocessing"]
        if not cfg["norm_mask"]:
            mask = np.ones_like(mask)

        if cfg["norm_method"] == "minmax":
                # taken from quicknat
            data[mask] = (data[mask] - np.min(data[mask])) / (
                np.max(data[mask]) - np.min(data[mask])
            )
        elif cfg["norm_method"] == "std":
            # taken from nnunet
            data[mask] = (data[mask] - data[mask].mean()) / (
                data[mask].std() + 1e-8
            )
            data[mask == 0] = 0
        return data

    def _clip(self, data, low_thr=-1024, up_thr=600):
        data[data < low_thr] = low_thr
        data[data > up_thr] = up_thr
        return data


    def _save_as_npy(self, pat_id, data, seg, save_identification):
        """save channels and labels to npy arrays for fast loading"""
        save_dict = {}
        save_dict["data"] = data
        save_dict["seg"] = seg

        for key in save_dict.keys():
            path = self.npy_path / (pat_id + "_" + key + "_" + save_identification)
            np.save(path.with_suffix(".npy"), save_dict[key])
