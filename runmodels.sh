#!/bin/bash

source venv/bin/activate
source /course/cs2950k/tf_gpu_venv/bin/activate

datasets=( "edgarallenpoe" "finneganswake" "lolita" "nightvale" "quentintarantino" "smallshakespeare" )

lstm=256
layer=3
batchsize=100
stepsize=100
numtopics=40
doclen=25
out="output/"

for d in "${datasets[@]}"
do
    python run.py $d lda data/$d.txt $out -num_topics $numtopics
    python run.py $d lda data/$d.txt $out -doclen $doclen

    python run.py $d rnn data/$d.txt $out -lstm_size $lstm
    python run.py $d rnn data/$d.txt $out -num_layers $layer
    python run.py $d rnn data/$d.txt $out -batch_size $batchsize
    python run.py $d rnn data/$d.txt $out -step_size $stepsize
done

#python run.py --label asongoffireandice --input_text data/asongoffireandice.txt --batch_size 25 --step_size 25
