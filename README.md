# Multi-modal Cycle-consistent Generalized Zero-Shot Learning
## frwgan eccv 18

**Paper**: [download paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/RAFAEL_FELIX_Multi-modal_Cycle-consistent_Generalized_ECCV_2018_paper.pdf)

Code for model presented on our paper accepted on European Conference on Computer Vision 2018.

**Abstract**: In generalized zero shot learning (GZSL), the set of classes are split into seen and unseen classes, where training relies on the semantic features of the seen and unseen classes and the visual representations of only the seen classes, while testing uses the visual representations of the seen and unseen classes.  Current methods address GZSL by learning a transformation from the visual to the semantic space, exploring the assumption that the distribution of classes in the semantic and visual spaces is relatively similar.  Such methods tend to transform unseen testing visual representations into one of the seen classes' semantic features instead of the semantic features of the correct unseen class, resulting in low accuracy GZSL classification.  Recently, generative adversarial networks (GAN) have been explored to synthesize visual representations of the unseen classes from their semantic features - the synthesized representations of the seen and unseen classes are then used to train the GZSL classifier.  This approach has been shown to boost GZSL classification accuracy, but there is one important missing constraint: there is no guarantee that synthetic visual representations can generate back their semantic feature in a multi-modal cycle-consistent manner.  This missing constraint can result in synthetic visual representations that do not represent well their semantic features, which means that the use of this constraint can improve GAN-based approaches. In this paper, we propose the use of such constraint based on a new regularization for the GAN training that forces the generated visual features to reconstruct their original semantic features. Once our model is trained with this multi-modal cycle-consistent semantic compatibility, we can then synthesize more representative visual representations for the seen and, more importantly, for the unseen classes.  Our proposed approach shows the best GZSL classification results in the field in several publicly available datasets.


## Dependencies

In order to reproduce the code, please check the `requirements.txt` file. Attach: ([requirements.txt](./requirements.txt)).

An extra dependency for this project consist of a util package containing all peripheral functionalities.
**Package**: [util](https://github.com/rfelixmg/util)


## Running:

```
# Help:
python train.py --help

# Training:
python train.py --baseroot /tmp/test/ --train_cls 1 --train_reg 1 --train_gan 1 --architecture_file ./src/bash/eccv18/awa/architecture_rwgan.json --gpu_devices "0" --gpu_memory 0.8 --save_model 0 --savepoints "[30,40,300]"

# Usage for training:

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

## Dataset:

**Download**: [dataset](https://cvml.ist.ac.at/AwA2/) 
[https://cvml.ist.ac.at/AwA2]

This dataset provides a platform to benchmark transfer-learning algorithms, in particular attribute base classification and zero-shot learning. It can act as a drop-in replacement to the original Animals with Attributes (AwA) dataset, as it has the same class structure and almost the same characteristics. In addition, in this website you will find CUB, SUN, FLO.

** HDF5 file:** The current code uses a HDF5 structure to organize the dataset. You call use h5py to create the following structure:

```
Legend: (+ group: hdf5 name for "folder", - dataset hdf5 name for "matrix")
Dataset: data.h5
Structure:
+ test:
++ {seen,unseen}}:
--- {X,Y}
+++ A:
---- {continuous, binary}
+ {train, val}:
-- {X,Y}
++ A:
--- {continuous, binary}



```
```
AttributeDictionary: knn.h5
+ {openset,openval,zsl}:
-- data
-- ids
++ {class2id,id2class,id2knn,knn2id}
--- data
```

In order to facilitate the display of the dataset you might want to use HDFView 2.9 (for linux users)


## License
```
MIT License

Copyright (c) 2018 Rafael Felix Alves

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
