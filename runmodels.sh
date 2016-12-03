#!/bin/bash 
source /course/cs2950k/tf_gpu_venv/bin/activate

datasets=( "finneganswake" "nightvale" "quentintarantino" "asongoffireandice" )

for d in "${datasets[@]}"
do
    python run.py --label $d --input_text data/$d.txt
done
