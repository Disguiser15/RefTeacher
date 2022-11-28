#!/usr/bin/env bash
export PYTHONPATH=./
CONFIG=$1
EVAL_WEIGHTS=$2

echo "Evaluating RefTeacher"
python -W ignore test_semi.py --config $CONFIG --eval-weights $EVAL_WEIGHTS
echo "end"