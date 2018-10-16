#!/usr/bin/env bash

_MODEL_TYPE_=rwgan

_GPU_DEV_="0"
_GPU_MEM_=0.9
_CONFERENCE_=eccv18
_DATABASE_=cub
_EXP_BATCH_=batch_001

_BASE_FOLDER_=~/$_CONFERENCE_/$_DATABASE_/$_MODEL_TYPE_/$_EXP_BATCH_/

_GAN_REF_=None ###INSERT_HERE_FOLDER_GENERATED_BY_TRAIN_GAN###
#example: _GAN_REF_=0001_TEST_161018_143113

_ARCHITECTURE_=$_BASE_FOLDER_/$_GAN_REF_/checkpoint/generator/
_GAN_FOLDER_=$_BASE_FOLDER_/$_GAN_REF_/

_OUTDIR_=$_GAN_FOLDER_/source/data

mkdir --parents $_OUTDIR_
rm $_OUTDIR_/*

_MERGE_=0
_DOMAIN_=unseen
_NUM_FEATURES_=200
_SAVENPY_=0

python -m routines.augment_features --dbname $_DATABASE_ --architecture_file $_ARCHITECTURE_ --outdir $_OUTDIR_ --merge $_MERGE_ --num_features $_NUM_FEATURES_ --domain $_DOMAIN_ --save_numpy $_SAVENPY_