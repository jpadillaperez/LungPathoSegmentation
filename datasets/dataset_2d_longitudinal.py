"""
Adapted from
https://gitlab.lrz.de/CAMP_IFL/dynamic-dataset/dynamic-dataset/dynamic_dataset/datasets/dynamic/dataset_2d_longitudinal.py
See https://gitlab.lrz.de/CAMP_IFL/dynamic-dataset/-/tree/master
on how to install and use the dynamic dataset.
"""

from .dataset_2d import DynamicDataset2D
import torch
import numpy as np
import torch.nn.functional as F


class DynamicDataset2DLongitudinal(DynamicDataset2D):
    def __init__(
            self,
            preprocessor,
            resize_labels=None,
            resize_images=None,
            remove_pleural_effusion=None,
            plane="axial",
            subset=None,
            long_steps_from_ref=2,
            test=False,
    ):
        DynamicDataset2D.__init__(
            self,
            preprocessor,
            resize_labels=resize_labels,
            resize_images=resize_images,
            remove_pleural_effusion=remove_pleural_effusion,
            plane=plane,
            subset=subset,
            test=test
        )
        assert long_steps_from_ref >= 2  # has to be at least 2
        self.long_steps_from_ref = long_steps_from_ref
        self.test = test
        self._init_dataset_longitudinal()


    def _init_dataset_longitudinal(self):
        # create slice index
        self.unique_patients = self.df["base_patient"].unique()
        self.df_base = self.df
        if self.test:
            self.df = self.df_base[self.df_base["base_patient"].isin(self.unique_patients)]
        else:
            self.df = self.df_base[self.df_base["pat_id"].isin(self.unique_patients)]
        self.df.reset_index(drop=True, inplace=True)
        self.num_volumes = len(self.df)
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

    def __len__(self):
        return self.num_slices

    def _get_slice_by_idx(self, idx):
        vol_idx, slc_idx = self.idx_to_vol_slc[idx]
        # volume check
        volume = self._get_vol_by_vol_idx(vol_idx)
        assert volume["image"].shape[0] >= 2

        data_dict = {}
        for k, v in volume.items():
            data_dict[k] = self._get_slice_from_volume(v, slc_idx)

        return data_dict

    def __getitem__(self, index):
        data_dict = self._get_slice_by_idx(index)

        if self.resize_images is not None and self.resize_labels is not None:
            data_dict = {k: torch.from_numpy(v) for k, v in data_dict.items()}
            data_dict['image'] = self.resize_images(data_dict['image'])
            data_dict['label'] = self.resize_labels(data_dict['label'])

            if self.remove_pleural_effusion is not None:
                data_dict['label'] = self.remove_pleural_effusion(data_dict['label'])
                data_dict['label'] = F.one_hot(data_dict['label'].long().squeeze(dim=0), num_classes=4).permute(0, 3, 1, 2)
            else:
                data_dict['label'] = F.one_hot(data_dict['label'].long().squeeze(dim=0), num_classes=5).permute(0, 3, 1, 2)
        else:
            data_dict = {k: torch.from_numpy(v) for k, v in data_dict.items()}

        data_dict.update({'pat_id': self.get_pat_id_from_idx(index), 'vol_idx': self._get_vol_idx_from_idx(index)})

        return data_dict