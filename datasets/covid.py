from datasets.base import Base
from datasets.dataset_2d import DynamicDataset2D
from torch.utils.data import DataLoader
from utils.utils import get_argparser_group
from datasets.cross_validation import  TrainValSplit, TestSplit
from torchvision import transforms
from utils.transforms import RemovePleuralEffusion
from PIL import Image

class Segmentation2D(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = dict()
        self.batch_size["val"] = self.params.batch_size
        self.batch_size["test"] = self.params.batch_size
        self.batch_size["train"] = self.params.batch_size
        

    def get_dataset(self, subset=None, phase=None):
        dataset = DynamicDataset2D(
                preprocessor=self.preprocessor,
                subset=subset,
                resize_labels=self.resize_labels(),
                resize_images=self.resize_images(),
                remove_pleural_effusion=self.remove_pleural_effusion(),
                plane='axial',
                test=self.params.run_test,
            )
        return dataset


    def get_split_idxs(self):
        if self.params.run_test:
            train_test_split = TestSplit()
        else:
            train_test_split = TrainValSplit()
        return train_test_split.get_idxs_split()

    def resize_labels(self):
        return transforms.Resize((300, 300), interpolation=Image.NEAREST)

    def resize_images(self):
        return transforms.Resize((300, 300), interpolation=Image.BILINEAR)

    def remove_pleural_effusion(self):
        return RemovePleuralEffusion()
        

    def get_dataloader(self, phase):
        sampler = self.get_sampler(phase)
        loader = DataLoader(
            dataset=self.datasets[phase],
            batch_size=self.batch_size[phase],
            shuffle=(phase == 'train' and sampler is None),
            num_workers=self.params.num_workers,
            pin_memory=True,
            drop_last=phase == 'train',
            sampler=sampler
        )
        return loader

    @staticmethod
    def add_dataset_specific_args(parser):
        specific_args = get_argparser_group(title="Dataset options", parser=parser)
        specific_args.add("--plane", default="axial", type=str)
        specific_args.add("--in_channels", default=1, type=int)
        specific_args.add("--out_channels", default=4, type=int)
        #specific_args.add("--dataset_setup", default="220102_kri-covid19-longitudinal-coco-pf-4scans-linear-resampled-1.8", type=str) # select dataset
        specific_args.add("--dataset_config", default="config_patho.yml", type=str)
        specific_args.add("--force_preprocessing", default=False, type=bool)
        specific_args.add("--drop_pat_ids", default=[], nargs='*', type=str)
        return parser




