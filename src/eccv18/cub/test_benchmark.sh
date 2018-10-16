#!/usr/bin/env bash

_GPU_DEV_="0"
_GPU_MEM_=0.9
_CONFERENCE_=eccv18
_DATABASE_=cub

_CLASSIFIER_REF_=~/eccv18/cub/rwgan/batch_001/benchmark/None ###INSERT_HERE_CLASSIFIER_FOLDER###
#example: _CLASSIFIER_REF_=~/eccv18/cub/rwgan/batch_001//benchmark/0000_TEST_161018_143518/

_LOAD_=$_CLASSIFIER_REF_/checkpoint/epoch_80_80/architecture.json #MAKE SURE EPOCH IS CORRECT#
_OUT_=$_CLASSIFIER_REF_/results/

python -m routines.tester --load_model $_LOAD_ --output $_OUT_ --dbname $_DATABASE_

