#!/usr/bin/env bash

# Dataset root
_DBDIR_=./data/
# Model type
_MODEL_TYPE_=cycle_wgan

_GPU_DEV_="0"
_GPU_MEM_=0.9
_CONFERENCE_=sota
_DATABASE_=awa1

_BASE_FOLDER_=./experiments/$_CONFERENCE_/$_DATABASE_/$_MODEL_TYPE_/
_BASE_FOLDER_=$_BASE_FOLDER_/gan-model/

#Architecture and checkpoint for generator
_ARCHITECTURE_=$_BASE_FOLDER_/checkpoint/generator/

_OUTDIR_=$_BASE_FOLDER_/source/data
echo $_OUTDIR_
mkdir --parents $_OUTDIR_
rm $_OUTDIR_/*

_MERGE_=0
# domain to be generated, can be: unseen, seen or [unseen,seen]
_DOMAIN_=[unseen,seen]
# Number of features per class, can be either: Number or [N_U, N_S]
_NUM_FEATURES_=[1200,300]
_SAVENPY_=0

# Routine
python -m routines.augment_features --dbname $_DATABASE_ --architecture_file $_ARCHITECTURE_ --outdir $_OUTDIR_ --merge $_MERGE_ --num_features $_NUM_FEATURES_ --domain $_DOMAIN_ --save_numpy $_SAVENPY_ --dataroot $_DBDIR_







