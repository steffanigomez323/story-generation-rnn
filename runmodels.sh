#!/bin/bash 
source /course/cs2950k/tf_gpu_venv/bin/activate

datasets=( "xfilesalloriginal" "edgarallenpoe" "finneganswake" "lolita" "nightvale" "quentintarantino" "smallshakespeare" "asongoffireandice")

lstm=256
layer=3
batchsize=25
stepsize=25

for d in "${datasets[@]}"
do
    python run.py --label $d --input_text data/$d.txt --words False --lstm_size $lstm
    python run.py --label $d --input_text data/$d.txt --words False --num_layers $layer
    python run.py --label $d --input_text data/$d.txt --words False --batch_size $batchsize
    python run.py --label $d --input_text data/$d.txt --words False --step_size $stepsize
#    python run.py --label $d --input_text data/$d.txt --batch_size 25 --step_size 25
done

#python run.py --label asongoffireandice --input_text data/asongoffireandice.txt --batch_size 25 --step_size 25
