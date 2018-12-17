#!/usr/bin/env bash

# Dataset root
_DBDIR_=./data/
# Model type
_MODEL_TYPE_=cycle_wgan
# Trainable architecture
_TRAINABLE_=fullyconnected

_GPU_DEV_="0"
_GPU_MEM_=0.9
_CONFERENCE_=sota
_DATABASE_=awa1

_BASE_=./experiments/$_CONFERENCE_/$_DATABASE_/$_MODEL_TYPE_/
_ARCHITECTURE_=./src/$_CONFERENCE_/$_DATABASE_/architecture/$_TRAINABLE_.json
_GAN_FOLDER_=$_BASE_/gan-model/

_BASE_FOLDER_=$_BASE_/

mkdir --parents $_BASE_FOLDER_


# Fake features directory
_DATADIR_=$_GAN_FOLDER_/source/data/
# Fake features H5file
_AUG_FILE_=$_DATADIR_/data.h5

_SAVEPOINTS_="[37]"
_SAVE_=0
# Every N epochs, perform evaluation on validation set
_EVERY_=1
# validation split from training [0,1]
_VAL_SPLIT_=0.1

# Data augmentation option (merge or replace real dataset)
_AUG_OP_=replace
# domain: openset or zsl
_DOMAIN_=openset

_DESCRIPTION_=classifier
python -m routines.benchmark --description $_DESCRIPTION_ --baseroot $_BASE_FOLDER_ --augm_file $_AUG_FILE_ --augm_operation $_AUG_OP_ --dbname $_DATABASE_ --architecture_file $_ARCHITECTURE_ --save_model $_SAVE_ --savepoints $_SAVEPOINTS_ --every $_EVERY_  --domain $_DOMAIN_ --gpu_devices $_GPU_DEV_ --gpu_memory $_GPU_MEM_ --validation_split $_VAL_SPLIT_ --dataroot $_DBDIR_

