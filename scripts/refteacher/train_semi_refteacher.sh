#!/usr/bin/env bash
export PYTHONPATH=./
CONFIG=$1

echo "Running RefTeacher"
python -W ignore train_semi_refteacher.py --config $CONFIG
echo "end"