#!/usr/bin/env bash
_MODEL_TYPE_=rwgan
_GPU_DEV_="0"
_GPU_MEM_=0.9
_CONFERENCE_=eccv18
_DATABASE_=cub
_EXP_BATCH_=batch_001
_BASE_FOLDER_=~/$_CONFERENCE_/$_DATABASE_/$_MODEL_TYPE_/$_EXP_BATCH_/

__DATA_FOLDER= <<< INSERT_HERE_YOUR_DATA_FOLDER >>>>
mkdir $_BASE_FOLDER_ --parents

_ARCHITECTURE_=./src/bash/$_CONFERENCE_/$_DATABASE_/architecture_$_MODEL_TYPE_.json

_TRAIN_GAN_=1
_TRAIN_CLS_=1
_TRAIN_REG_=1
_SAVE_POINTS_="[926]"
_SAVE_=0

python train.py --dbname $_DATABASE_ --baseroot $_BASE_FOLDER_ --train_gan $_TRAIN_GAN_ --train_cls $_TRAIN_CLS_ --train_reg $_TRAIN_REG_ --architecture_file $_ARCHITECTURE_ --gpu_devices $_GPU_DEV_ --gpu_memory $_GPU_MEM_ --save_model $_SAVE_ --savepoints $_SAVE_POINTS_ --datadir $__DATA_FOLDER