# Longitudinal Covid-19 Segmentation restructured

## Create new dataset
- run create_dataset.py
- set the following parameters
  - *config_yml_path* - set the path to the config_yml file
  - *output_path* - set path to where the dataset should be saved
  - *force_preprocessing* - if set to *True* runs preprocessing even if there are already preprocessed files
  - *max_step* - max amount of volumes of a patient to use
  - *min_step* - minimum amount of volumes of a patient to use, minimum 2
  - *register_config*
    - *pf* - past to future
    - *fp* - future to past
    - *all* - past to future and future to past
    - *static* - first scan always t=0 and second scan t=0,1,..; creates dataset that can be used for static training


## Segmentation
- adjust *config_segmentation.yml*
  - set *pretraining_path* to initialize weights with weights form a pretrained model or uncomment for random initalization
  - for **longitudinal** segmentation:
    - *longitudinal* must be set
  - for **static** segmentation:
    - uncomment *longitudinal*
  - for test run:
    - uncomment *run_test* to run an evaluation on the test set form the second wave
    
- For training run:
```
train.py -c modules/segmentation/config/config_segmentation.yml
```
- For testing:
  - set model path in *test.py*
  - run:
```
test.py -c modules/segmentation/config/config_segmentation.yml
```



## Classification
- adjust *config_classification.yml*
  - set *pretraining_path* to initialize weights with weights form a pretrained model or uncomment for random initalization
  - set *pixel_threshold* to a percentage of the total pixels (default 0.025 (2.5%))
  - for **longitudinal** classification:
    - *longitudinal* must be set
  - for **static** classification:
    - uncomment *longitudinal*
  - for test run:
    - uncomment *run_test* to run an evaluation on the test set form the second wave

- For training run: 
```
train_classification.py -c modules/classification/config/config_classification.yml
```
- For testing:
  - set model path in *test.py*
  - run:
```
test_classification.py -c modules/classification/config/config_classification.yml
```
