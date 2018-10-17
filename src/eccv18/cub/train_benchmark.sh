#!/usr/bin/env bash

#rwgan,cwgan
_MODEL_TYPE_=rwgan

_GPU_DEV_="0"
_GPU_MEM_=0.9
_CONFERENCE_=eccv18
_DATABASE_=cub
_EXP_BATCH_=batch_001

# Saving in your home folder
_BASE_=~/$_CONFERENCE_/$_DATABASE_/$_MODEL_TYPE_/$_EXP_BATCH_/

_ARCHITECTURE_=./src/$_CONFERENCE_/$_DATABASE_/architecture_fc.json

_GAN_REF_=None ###INSERT_HERE_FOLDER_GENERATED_BY_TRAIN_GAN###
#example _GAN_REF_=0001_TEST_161018_143113


_GAN_FOLDER_=$_BASE_/$_GAN_REF_/
_BASE_FOLDER_=$_BASE_/benchmark

mkdir --parents $_BASE_FOLDER_


#Original dataset
__DATA_FOLDER=None ###INSERT_HERE_YOUR_DATA_FOLDER###

_DATADIR_=$_GAN_FOLDER_/source/data/
_AUG_FILE_=$_DATADIR_/data.h5
_SAVE_=0
_SAVEPOINTS_="[80]"
_EVERY_=1
_AUG_OP_=merge
_DOMAIN_=openval

python -m routines.benchmark --baseroot $_BASE_FOLDER_ --augm_file $_AUG_FILE_ --augm_operation $_AUG_OP_ --dbname $_DATABASE_ --architecture_file $_ARCHITECTURE_ --save_model $_SAVE_ --savepoints $_SAVEPOINTS_ --every $_EVERY_  --domain $_DOMAIN_ --dataroot $__DATA_FOLDER

