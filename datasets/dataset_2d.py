"""
Adapted from
https://gitlab.lrz.de/CAMP_IFL/dynamic-dataset/dynamic-dataset/dynamic_dataset/datasets/dynamic/dataset_2d.py
See https://gitlab.lrz.de/CAMP_IFL/dynamic-dataset/-/tree/master
on how to install and use the dynamic dataset.
"""

from torch.utils.data import Dataset
import torch
import yaml
import numpy as np
import torch.nn.functional as F
from pathlib import Path


class DynamicDataset2D(Dataset):
    def __init__(
        self,
        preprocessor,
        resize_labels=None,
        resize_images = None,
        remove_pleural_effusion = None,
        plane="axial",
        subset=None,
        verbose=False,
        test = False
    ):
        self.preprocessor = preprocessor
        # load dataset config
        self.cfg = self.preprocessor.cfg
        self.verbose = verbose
        self.test = test
        self.channels = self.cfg["channels"]
        self.num_channels = len(self.cfg["channels"])
        self.labels = self.cfg["labels"]
        assert plane in [
            "axial",
            "sagittal",
            "coronal",
        ], "plane needs to be one of 'axial', 'sagittal', 'coronal'"
        self.plane = plane
        self.axis_by_plane = {"sagittal": 0, "coronal": 1, "axial": 2}
        self.axis = self.axis_by_plane[plane]
        self.image_keys = ['label', 'image']

        # link to setup configuration of dataset
        self.setup = self.preprocessor.setup

        # set temporary dir for npy files
        self.npy_path = self.preprocessor.npy_path
        if not self.npy_path.is_dir():
            raise ValueError(f"npy dir could not be found at {self.npy_path}")

        # dict for data to be loaded
        self.data = {}
        self.resize_labels = resize_labels
        self.remove_pleural_effusion = remove_pleural_effusion
        self.resize_images = resize_images

        # load base patient dataframe
        self.df = self.preprocessor.df
        self.base_df = self.preprocessor.base_df

        #extension options for files
        self.options = ["_00", "_01", "_10", "_02", "_20", "_12", "_21", "_03", "_30", "_13", "_31", "_23", "_32" ]

        ##Comment if not longitudinal
        # select subset for training and validation
        #if subset is not None and self.test == False:
        #    self.unique_patients = self.df["split_patient"].unique()
        #    self.df_new = self.base_df
        #    self.base_df = self.df_new[self.df_new["pat_id"].isin(self.unique_patients)]
        #    self.base_df = self.base_df.reset_index(drop=True)
        #    self.base_df = self.base_df[self.base_df['base_patient'].isin(subset)]
        #    self.base_df = self.base_df.reset_index(drop=True)
        #    patientlist = self.base_df["base_patient"]
        #    self.df = self.df[self.df["split_patient"].isin(patientlist)]
        #    self.df = self.df.reset_index(drop=True)

        # select subset for testing
        #if subset is not None and self.test:
        #    self.unique_patients = self.df["split_patient"].unique()
        #    self.df_new = self.base_df
        #    self.base_df = self.df_new[self.df_new["base_patient"].isin(self.unique_patients)]
        #    self.base_df = self.base_df.reset_index(drop=True)
        #    self.base_df = self.base_df.loc[self.base_df["long_step"] == 0]
        #    self.base_df = self.base_df.reset_index(drop=True)
        #    self.base_df = self.base_df.loc[subset]
        #    self.base_df = self.base_df.reset_index(drop=True)
        #    patientlist = self.base_df["pat_id"]
        #    self.df = self.df[self.df["pat_id"].isin(patientlist)]
        #    self.df = self.df.reset_index(drop=True)
        ##

        self.max_dims = [self.df[f"dim{i}"].max() for i in range(3)]

        # calculate total number of slices
        self.num_volumes = len(self.df)
        self._init_dataset()
        if verbose:
            self._print_summary()

    def _init_dataset(self):
        # create slice index
        num_slices_col = f"dim{self.axis}"
        self.num_slices = self.df[num_slices_col].sum(axis=0)
        self.idx_to_vol_slc = []
        self.slc_per_volume = []
        for vol_idx in range(len(self.df)):
            max_slice = self.df.loc[vol_idx, num_slices_col]
            for slc_idx in range(max_slice):
                self.idx_to_vol_slc.append([vol_idx, slc_idx])
            self.slc_per_volume.append(max_slice)
        self.idx_to_vol_slc = np.array(self.idx_to_vol_slc)

    def export_config(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path / 'patients.csv')
        (path / f'config_{self.cfg["name"]}.yml').write_text(yaml.dump(self.cfg))

    def _print_summary(self):
        print(
            f"2D Dataset '{self.setup['name']}' loaded with configuration '{self.cfg['name']}', {self.num_volumes} volumes, {self.num_slices} slices"
        )

    def print_config(self):
        print("\nConfiguration:")
        print(yaml.dump(self.cfg))

    def __len__(self):
        return self.num_slices

    def __getitem__(self, index):
        data_dict = self._get_slice_by_idx(index)

        #if self.transform2d is not None:
        #    data_dict = self.transform2d(data_dict)
        #else:
        #data_dict = {k: torch.from_numpy(v) for k, v in data_dict.items()}

        #data_dict['label'] = F.one_hot(data_dict['label'].long().squeeze(
        #    dim=0), num_classes=len(self.labels)).permute(-1, 0, 1)

        

        if self.resize_images is not None and self.resize_labels is not None:
            data_dict = {k: torch.from_numpy(v) for k, v in data_dict.items()}

            if data_dict['image'].ndim == 2:
                data_dict['image'] = data_dict['image'].unsqueeze(dim=0).unsqueeze(dim=0)
                data_dict['image'] = self.resize_images(data_dict['image'])
                data_dict['image'] = data_dict['image'].squeeze(dim=0).squeeze(dim=0)
            else:
                data_dict['image'] = self.resize_images(data_dict['image'])

            if data_dict['label'].ndim == 2:
                data_dict['label'] = data_dict['label'].unsqueeze(dim=0).unsqueeze(dim=0)
                data_dict['label'] = self.resize_labels(data_dict['label'])
                if self.remove_pleural_effusion is not None:
                    data_dict['label'] = self.remove_pleural_effusion(data_dict['label'])
                    data_dict['label'] = F.one_hot(data_dict['label'].long().squeeze(dim=0), num_classes=4).permute(0, 3, 1, 2).squeeze(dim=0).squeeze(dim=0)
            else:
                data_dict['label'] = F.one_hot(data_dict['label'].long().squeeze(dim=0), num_classes=5).permute(0, 3, 1, 2)

        else:
            data_dict = {k: torch.from_numpy(v) for k, v in data_dict.items()}

       
        data_dict.update({'pat_id': self.get_pat_id_from_idx(index), 'vol_idx': self._get_vol_idx_from_idx(index)})

        return data_dict

    def _load_npy(self, path):
        mmap_mode = "r"
        return np.load(path.with_suffix(".npy"), mmap_mode=mmap_mode)

    def _get_pat_id_from_vol_idx(self, vol_idx):
        """get pat id from vol_idx"""
        if self.test:
            #return self.df.loc[vol_idx, "pat_id"]
            return self.df.loc[vol_idx, "base_patient"]
        else:
            return self.df.loc[vol_idx, self.setup["id_col"]]

    def get_vol_idx_from_pat_id(self, pat_id):
        """get vol idx from pat id"""
        if self.test:
            #return self.df[self.df["pat_id"] == pat_id].index[0]
            return self.df[self.df["base_patient"] == pat_id].index[0]
        else:
            return self.df[self.df[self.setup["id_col"]] == pat_id].index[0]
        
    def _get_vol_idx_from_idx(self, idx):
        vol_idx, _ = self.idx_to_vol_slc[idx]
        return vol_idx

    def get_pat_id_from_idx(self, idx):
        """get pat id from idx"""
        vol_idx = self._get_vol_idx_from_idx(idx)
        pat_id = self._get_pat_id_from_vol_idx(vol_idx)
        return pat_id

    def _get_slice_from_volume(self, volume, slc_idx):
        axis = -(3-self.axis)
        return np.take(volume, slc_idx, axis=axis)

    def _get_vol_by_vol_idx(self, vol_idx):
        pat_id = self._get_pat_id_from_vol_idx(vol_idx)
        save_pat_id = pat_id
        if pat_id in self.data.keys():
            data = self.data[pat_id]
        else:
            data = {}
            ##comment if longitudinal
            #identification_string = pat_id[-3:]
            #if identification_string in self.options:
            #    pat_id = pat_id[:-3]
            #else:
            #    identification_string = "_01"
            ##
            
            for key in ["data", "seg"]:
                path = self.npy_path / (pat_id + "_" + key)
                #path = self.npy_path / \
                #    (pat_id + "_" + key + identification_string)
                data[key] = self._load_npy(path)
            self.data[save_pat_id] = data
            data['image'] = data.pop("data")
            data['label'] = data.pop("seg")
        return data

    def _get_slice_by_idx(self, idx):
        vol_idx, slc_idx = self.idx_to_vol_slc[idx]
        volume = self._get_vol_by_vol_idx(vol_idx)
        data_dict = {}
        for k, v in volume.items():
            data_dict[k] = self._get_slice_from_volume(v, slc_idx)
        return data_dict

    def get_mfb_weights(self, excl_index=[]):
        num_labels = len(self.labels)
        idxs = range(num_labels)
        idxs = np.delete(idxs, excl_index)
        num_labels = len(idxs)

        weights = np.ones(num_labels, dtype=np.float32)
        frequencies = np.zeros(num_labels, dtype=np.float32)

        for i, idx in enumerate(idxs):
            frequencies[i] = self.df[f"num_{self.cfg['labels'][idx]}"].mean(axis=0)
        median_freq = np.median(frequencies)

        for i in range(num_labels):
            weights[i] = median_freq / frequencies[i]

        return weights