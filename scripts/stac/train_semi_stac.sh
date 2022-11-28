#!/usr/bin/env bash
export PYTHONPATH=./
CONFIG=$1
RESUME_WEIGHTS=$2

echo "Running STAC baseline"
python -W ignore train_semi.py --config $CONFIG --resume-weights $RESUME_WEIGHTS
echo "end"