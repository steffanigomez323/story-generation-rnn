#!/bin/bash 
source /course/cs2950k/tf_gpu_venv/bin/activate

datasets=( "finneganswake" "nightvale" "quentintarantino" "smallshakespeare" )

for d in "${datasets[@]}"
do
    python run.py --label $d --input_text data/$d.txt
done

python run.py --label asongoffireandice --input_text data/asongoffireandice.txt --batch_size 25 --step_size 25 
