"""
Adapted from
https://gitlab.lrz.de/CAMP_IFL/dynamic-dataset/-/blob/master/dynamic_dataset/dataloader/covid.py
See
https://gitlab.lrz.de/CAMP_IFL/dynamic-dataset/-/tree/master
on how to install and use the dynamic dataset."""


from preprocessing_long_2D import DatasetPreprocessor
from dataset_2d import DynamicDataset2D

dataset = DynamicDataset2D(
    preprocessor=DatasetPreprocessor(
        config_yml_path='C://Users/Jorge/Desktop/tmp/config_patho.yml',
        output_path= 'C://Users/Jorge/Desktop/tmp',
        force_preprocessing = False,
    ),
    verbose=True,
    plane='axial'
)




