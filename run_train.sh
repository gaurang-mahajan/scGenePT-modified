#!/bin/sh

python ./train.py --model-type=scgpt --num-epochs=20 --dataset=norman --device=cuda:0 --batch-size=16 --eval-batch-size=16

python ./train.py --model-type=scgenept_go_all_gpt_concat --num-epochs=20 --dataset=norman --device=cuda:0 --batch-size=16 --eval-batch-size=16
