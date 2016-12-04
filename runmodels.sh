#!/bin/bash 
source /course/cs2950k/tf_gpu_venv/bin/activate

#datasets=( "finneganswake" "nightvale" "quentintarantino" "asongoffireandice" )

datasets=( "lolita" "xfilesseason1" )

for d in "${datasets[@]}"
do
    python run.py --label $d --input_text data/$d.txt
done
