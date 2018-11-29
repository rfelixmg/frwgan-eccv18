# Multi-modal Cycle-consistent Generalized Zero-Shot Learning
## cycle-WGAN ECCV 18

**Paper**: [download paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/RAFAEL_FELIX_Multi-modal_Cycle-consistent_Generalized_ECCV_2018_paper.pdf)

Code for model presented on our paper accepted on European Conference on Computer Vision 2018.

**Abstract**: In generalized zero shot learning (GZSL), the set of classes are split into seen and unseen classes, where training relies on the semantic features of the seen and unseen classes and the visual representations of only the seen classes, while testing uses the visual representations of the seen and unseen classes.  Current methods address GZSL by learning a transformation from the visual to the semantic space, exploring the assumption that the distribution of classes in the semantic and visual spaces is relatively similar.  Such methods tend to transform unseen testing visual representations into one of the seen classes' semantic features instead of the semantic features of the correct unseen class, resulting in low accuracy GZSL classification.  Recently, generative adversarial networks (GAN) have been explored to synthesize visual representations of the unseen classes from their semantic features - the synthesized representations of the seen and unseen classes are then used to train the GZSL classifier.  This approach has been shown to boost GZSL classification accuracy, but there is one important missing constraint: there is no guarantee that synthetic visual representations can generate back their semantic feature in a multi-modal cycle-consistent manner.  This missing constraint can result in synthetic visual representations that do not represent well their semantic features, which means that the use of this constraint can improve GAN-based approaches. In this paper, we propose the use of such constraint based on a new regularization for the GAN training that forces the generated visual features to reconstruct their original semantic features. Once our model is trained with this multi-modal cycle-consistent semantic compatibility, we can then synthesize more representative visual representations for the seen and, more importantly, for the unseen classes.  Our proposed approach shows the best GZSL classification results in the field in several publicly available datasets.

## Citation
```
@inproceedings{felix2018multi,
  title={Multi-modal Cycle-Consistent Generalized Zero-Shot Learning},
  author={Felix, Rafael and Kumar, BG Vijay and Reid, Ian and Carneiro, Gustavo},
  booktitle={European Conference on Computer Vision},
  pages={21--37},
  year={2018},
  organization={Springer}
}
```

## Dependencies

In order to reproduce the code, please check the `requirements.txt` file. 
Attach: [requirements.txt](./requirements.txt).

In addition, it is necessary to clone the repository [util](https://github.com/rfelixmg/util), which is an extra dependency for this project with some functionalities.

**Package**: [util](https://github.com/rfelixmg/util) -- https://github.com/rfelixmg/util.git


# Basic Usage:

The main file for training cycle-WGAN is [train.py](https://github.com/rfelixmg/frwgan-eccv18/blob/master/train.py)

```
# Help:
python train.py --help

# Example
python train.py --baseroot /tmp/test/ --train_cls 1 --train_reg 1 --train_gan 1 --architecture_file ./src/bash/eccv18/awa/architecture_rwgan.json --gpu_devices "0" --gpu_memory 0.8 --save_model 0 --savepoints "[30,40,300]"

usage: train.py [-h] [--root ROOT] [--namespace NAMESPACE]
                [--gpu_devices GPU_DEVICES] [--gpu_memory GPU_MEMORY]
                [--savepoints SAVEPOINTS] [--setup SETUP]
                [--save_model SAVE_MODEL] [--save_from SAVE_FROM]
                [--save_every SAVE_EVERY] [--saveall SAVEALL]
                [--dbname DBNAME] [--dataroot DATAROOT] [--datadir DATADIR]
                [--baseroot BASEROOT] [--description DESCRIPTION]
                [--plusinfo PLUSINFO] [--sideinfo SIDEINFO]
                [--exp_directories EXP_DIRECTORIES] [--auxroot AUXROOT]
                [--timestamp TIMESTAMP] [--load_model LOAD_MODEL]
                [--checkpoint CHECKPOINT] [--checkpoints_max CHECKPOINTS_MAX]
                [--validation_split VALIDATION_SPLIT]
                [--metric_list METRIC_LIST]
                [--checkpoint_metric CHECKPOINT_METRIC]
                [--architecture_file ARCHITECTURE_FILE]
                [--train_cls TRAIN_CLS] [--train_reg TRAIN_REG]
                [--train_gan TRAIN_GAN] [--att_type ATT_TYPE]

```

## Pre-defined experiments

We added a few routines that running the entire training, which reproduces State-of-the-art results reported in our paper. You will find these experiments in ./src/sota/{dataset_name}/. For each dataset, we defined:
> 1. **train_gan.sh** -- script to perform the training of cycle-WGAN.
> 2. **generate_dataset.sh** -- script which uses the trained cycle-WGAN to generate pseudo samples.
> 3. **benchmark.sh** -- script to train the fully connected classifier to perform GZSL classification.
> 4. **tester.sh** -- script which uses the trained cycle-WGAN to generate pseudo samples.
> 5. **architecture/cycle-wgan.json** -- architecture file for cycle-WGAN.
> 6. **architecture/fully-connected.json** -- architecture file for classifier.

**Running**
```
# make sure to adjust the paths and gpu usage parameters
bash ./src/sota/train_gan.sh
```

# Datasets

## Original files
**Download**: [dataset](https://cvml.ist.ac.at/AwA2/)  ~ https://cvml.ist.ac.at/AwA2

This dataset provides a platform to benchmark transfer-learning algorithms, in particular, attribute base classification and zero-shot learning. It can act as a drop-in replacement to the original Animals with Attributes (AwA) dataset, as it has the same class structure and almost the same characteristics. In addition, on this website, you will find CUB, SUN, FLO.

## Dataset in H5File

In order to run our experiments, and facilitate visualization of the dataset, we introduce the GZSL datasets in H5file format. You can download the datasets CUB, SUN, FLO, and AWA1 in the h5file format, which can be downloaded [dataset.zip](https://drive.google.com/open?id=1N9K-w993Cv0zgZOkF3Rpls5qrfOMpQTj) (~5GB in zip file). Alternativelly, run the following bash lines:
```
!wget https://drive.google.com/open?id=1N9K-w993Cv0zgZOkF3Rpls5qrfOMpQTj -O ./data/datasets.zip
unzip ./data/datasets.zip -d ./data/
```
**HDF5 file:** The current code uses a HDF5 structure to organize the dataset. In order to facilitate the display of the dataset you might want to use HDFView 2.9 (for linux users)




# Initial setup

The general setup for running the experiments can be achieve running the following bash lines.
```
git clone https://github.com/rfelixmg/frwgan-eccv18.git
cd frwgan-eccv18/
git clone https://github.com/rfelixmg/util.git
conda install requirements.txt

wget https://drive.google.com/open?id=1N9K-w993Cv0zgZOkF3Rpls5qrfOMpQTj -O ./data/datasets.zip
unzip ./data/datasets.zip -d ./data/
```
