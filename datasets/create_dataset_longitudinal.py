"""
Adapted from
https://gitlab.lrz.de/CAMP_IFL/dynamic-dataset/-/blob/master/dynamic_dataset/dataloader/covid.py
See
https://gitlab.lrz.de/CAMP_IFL/dynamic-dataset/-/tree/master
on how to install and use the dynamic dataset."""


from preprocessing_long import DatasetPreprocessorLongitudinal
from dataset_2d_longitudinal import DynamicDataset2DLongitudinal

dataset = DynamicDataset2DLongitudinal(
    preprocessor=DatasetPreprocessorLongitudinal(
        config_yml_path='//10.23.0.13/polyaxon/data1/dyndata-longitudinal/220102_kri-covid19-longitudinal-wave2-coco-pf-4scans-linear-resampled-1.8/config_patho.yml',
        output_path= 'C://Users/Jorge/Desktop/tmp',
        force_preprocessing = False,
        max_step =4,  # max long steps
        register_config="static", #registration direction: pf (past to future), fp (future to past), all (pf and fp), static (registration not important, always long_step 0 als registered reference scan)
        min_step=2 #min long steps
    ),
    plane='axial',
    transform2d=None
)




