from torch.utils.data import Dataset
import torch
import yaml
import numpy as np
import torch.nn.functional as F
from pathlib import Path


class DynamicDataset3D(Dataset):
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

        self.max_dims = [self.df[f"dim{i}"].max() for i in range(3)]

        # calculate total number of slices
        self.num_volumes = len(self.df)
        if verbose:
            self._print_summary()

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
        return self.num_volumes

    def __getitem__(self, index):
        data_dict = self.get_vol_by_vol_idx(index)

        if self.resize_images is not None and self.resize_labels is not None:
            data_dict = {k: torch.from_numpy(v) for k, v in data_dict.items()}

            # resize images
            data_dict['image'] = data_dict['image'].unsqueeze(dim=0).unsqueeze(dim=0)
            data_dict['image'] = self.resize_images(data_dict['image'])
            data_dict['image'] = data_dict['image'].squeeze(dim=0)#.squeeze(dim=0)

            # resize labels
            data_dict['label'] = data_dict['label'].unsqueeze(dim=0).unsqueeze(dim=0)
            data_dict['label'] = self.resize_labels(data_dict['label'])
            data_dict['label'] = data_dict['label'].squeeze(dim=0).squeeze(dim=0)

            if self.remove_pleural_effusion is not None:
                data_dict['label'] = self.remove_pleural_effusion(data_dict['label'])
                data_dict['label'] = F.one_hot(data_dict['label'].long(), num_classes=4).permute(3, 0, 1, 2)
            else:
                data_dict['label'] = F.one_hot(data_dict['label'].long(), num_classes=5).permute(3, 0, 1, 2)

        else:
            data_dict = {k: torch.from_numpy(v) for k, v in data_dict.items()}

       
        data_dict.update({'pat_id': self.get_pat_id_from_idx(index), 'vol_idx': index})

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

    def get_pat_id_from_idx(self, idx):
        """get pat id from idx"""
        pat_id = self._get_pat_id_from_vol_idx(idx)
        return pat_id

    def get_vol_by_vol_idx(self, vol_idx):
        pat_id = self._get_pat_id_from_vol_idx(vol_idx)
        save_pat_id = pat_id
        if pat_id in self.data.keys():
            data = self.data[pat_id]
        else:
            data = {}
            for key in ["data", "seg"]:
                path = self.npy_path / (pat_id + "_" + key)
                data[key] = self._load_npy(path)
            self.data[save_pat_id] = data
            data['image'] = data.pop("data")
            data['label'] = data.pop("seg")
        return data

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