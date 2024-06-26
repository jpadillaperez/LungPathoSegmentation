import lightning as l
from pathlib import Path
from datasets.preprocessing_long_2D import DatasetPreprocessor
from datasets.preprocessing_long_3D import DatasetPreprocessor3D
from torch.utils.data import DataLoader

class Base(l.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.params = hparams
        self.output_path = Path(self.params["output_path"])
        self.axis_by_plane = {"sagittal": 0, "coronal": 1, "axial": 2}
        self.plane_by_axis = ["sagittal", "coronal", "axial"]
        self.axis = self.axis_by_plane[self.params["plane"]]
        self.batch_size = {phase: self.params["batch_size"] for phase in ['train', 'test', 'val']}
        self.fold_idxs = None
        self.datasets = {}

    def prepare_data(self):
        self.preprocessor = DatasetPreprocessor(
            config_yml_path=Path(self.params["data_root"]) / self.params["dataset_config"],
            output_path= Path(self.params["data_root"]),
            force_preprocessing=self.params["force_preprocessing"],
        )
        #self.preprocessor.export_config(self.output_path / 'dataset')
        self.labels = self.preprocessor.cfg["labels"]

        # resolution dataset has been resampled to
        self.resolution = self.preprocessor.setup["resolution"]
        self.hparams["resolution"] = self.resolution

    def get_dataset(self, subset=None, phase=None):
        raise NotImplementedError
        return dataset

    def get_stratifier(self):
        raise NotImplementedError
        return stratifier

    def get_split_idxs(self):
        raise NotImplementedError
        return fold_idxs

    def get_transforms_2d(self, phase=None):
        return None

    def get_transforms_3d(self, phase=None):
        return None
    
    def get_target_weights(self):
        return None

    def setup(self, stage=None):
        if self.fold_idxs is None:
            self.fold_idxs = self.get_split_idxs()
        
        if stage == 'fit' or stage is None:
            for phase in ['train', 'val']:
                self.datasets[phase] = self.get_dataset(subset=self.fold_idxs[phase], phase=phase)
            self.datasets['all'] = self.get_dataset(phase='test') # TODO maybe remove
        if stage == 'test' or stage is None:
            self.datasets[stage] = self.get_dataset(self.fold_idxs[stage], phase=stage)
            
        if stage == 'predict':
            raise NotImplementedError
            self.datasets[stage] = self.get_dataset(phase=stage)

    def get_dataloader(self, phase):
        sampler = self.get_sampler(phase)
        return DataLoader(
            dataset=self.datasets[phase],
            batch_size=self.batch_size[phase],
            shuffle=(phase=='train' and sampler is None),
            num_workers=self.hparams["num_workers"],
            pin_memory=False,
            drop_last=phase=='train',
            sampler=sampler
        )

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        return self.get_dataloader('test')

    #adapt it
    def predict_dataloader(self):
        raise NotImplementedError
        return self.get_dataloader('predict')
    
    def get_sampler(self, phase):
        return None


class Base3D(l.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.params = hparams
        self.output_path = Path(self.params["output_path"])
        self.batch_size = {phase: self.params["batch_size"] for phase in ['train', 'test', 'val']}
        self.fold_idxs = None
        self.datasets = {}

    def prepare_data(self):
        self.preprocessor = DatasetPreprocessor3D(
            config_yml_path=Path(self.params["data_root"]) / self.params["dataset_config"],
            output_path= Path(self.params["data_root"]),
            force_preprocessing=self.params["force_preprocessing"],
        )
        #self.preprocessor.export_config(self.output_path / 'dataset')
        self.labels = self.preprocessor.cfg["labels"]

        # resolution dataset has been resampled to
        self.resolution = self.preprocessor.setup["resolution"]
        self.hparams["resolution"] = self.resolution

    def get_dataset(self, subset=None, phase=None):
        raise NotImplementedError
        return dataset

    def get_stratifier(self):
        raise NotImplementedError
        return stratifier

    def get_split_idxs(self):
        raise NotImplementedError
        return fold_idxs

    def get_transforms_2d(self, phase=None):
        return None

    def get_transforms_3d(self, phase=None):
        return None
    
    def get_target_weights(self):
        return None

    def setup(self, stage=None):
        if self.fold_idxs is None:
            self.fold_idxs = self.get_split_idxs()
        
        if stage == 'fit' or stage is None:
            for phase in ['train', 'val']:
                self.datasets[phase] = self.get_dataset(subset=self.fold_idxs[phase], phase=phase)
            self.datasets['all'] = self.get_dataset(phase='test') # TODO maybe remove
        if stage == 'test' or stage is None:
            self.datasets[stage] = self.get_dataset(self.fold_idxs[stage], phase=stage)
            
        if stage == 'predict':
            raise NotImplementedError
            self.datasets[stage] = self.get_dataset(phase=stage)

    def get_dataloader(self, phase):
        sampler = self.get_sampler(phase)
        return DataLoader(
            dataset=self.datasets[phase],
            batch_size=self.batch_size[phase],
            shuffle=(phase=='train' and sampler is None),
            num_workers=self.hparams["num_workers"],
            pin_memory=False,
            drop_last=phase=='train',
            sampler=sampler
        )

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        return self.get_dataloader('test')

    #adapt it
    def predict_dataloader(self):
        raise NotImplementedError
        return self.get_dataloader('predict')
    
    def get_sampler(self, phase):
        return None
