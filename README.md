This repository is used for submitting code of team polixir.ai for [pcqm4m-v2 track of ogb-lsc](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)

## How to reproduce score

### Preparation
Open config.py and set "ogb_data_path" and "middle_data_path" to two different folders on your local drive that should have 100GB free space.

run commands:
```shell
# download from ogb server, which takes about 1 hour depending on your internet speed.
python download_dataset.py
# convert ogb dataset to internal format, takes about 6 minutes on a machine with 80 cores of cpu.
python convert_dataset.py
# add 3d coordinate contained in sdf file to internal format, takes about 26 minutes.
python convert_dataset_sdf.py
```

### Reproduce single model validation score
run commands:
```shell
python validate_single_ddp.py
```
it will print validation score every epoch, after 100 epochs(about 50 hours on 8 x rtx3090), you will see the score is around 0.847


### Reproduce ensemble model test-dev score
you can train final models like this:
```shell
python train_final.py 0
python train_final.py 10
python train_final.py 19
python train_final.py 29
python train_final.py 39
python train_final.py 49
python train_final.py 24
python train_final.py 0
```
We split the whole train and valid data into 50 splits. Number 29 indicate that we test the 29-th split of the dataset and train with the rest. All models are saved into models_valid folder. 

### Computational budget
because we do not extract 3d coordinate for test set, so the feature extraction process is extremely fast which is about 15 seconds.
inference time for single model is about 110 seconds for test set.
we ensemble 8 models which cost about 110*8+15=895 seconds, that is to say the whole testing process is about 15 minutes in total.

