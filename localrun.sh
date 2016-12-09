#!/usr/bin/env bash

input="lolita"
model="rnn"
out="output/"

python run.py $input $model data/$input.txt $out