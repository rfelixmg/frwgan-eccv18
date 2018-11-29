#!/usr/bin/env bash
# Dataset root
_DBDIR_=./data/
# Model type
_MODEL_TYPE_=cycle_wgan
# GPU assign
_GPU_DEV_="0"
# GPU memory use
_GPU_MEM_=0.9
# conference flag
_CONFERENCE_=sota
# DB type. (requires a folder at _DBDIR_)
_DATABASE_=cub

# Set up of base folder for running GAN
_BASE_FOLDER_=./experiments/$_CONFERENCE_/$_DATABASE_/$_MODEL_TYPE_/
mkdir $_BASE_FOLDER_ --parents

# Architecture file. Provided in the code
_ARCHITECTURE_=./src/$_CONFERENCE_/$_DATABASE_/architecture/$_MODEL_TYPE_.json

_TRAIN_GAN_=1

#Note: the classifier is not used to train, only to evaluate the fake samples, as an stop criteria.
_TRAIN_CLS_=1
_TRAIN_REG_=1
# Saving points for REG, CLS and GAN
_SAVE_POINTS_="[1,30,40,926]"
# Saving option for every epoch 0 switch off
_SAVE_=0

# Training routine for GAN
python train.py --dbname $_DATABASE_ --baseroot $_BASE_FOLDER_ --train_gan $_TRAIN_GAN_ --train_cls $_TRAIN_CLS_ --train_reg $_TRAIN_REG_ --architecture_file $_ARCHITECTURE_ --gpu_devices $_GPU_DEV_ --gpu_memory $_GPU_MEM_ --save_model $_SAVE_ --savepoints $_SAVE_POINTS_ --dataroot $_DBDIR_
