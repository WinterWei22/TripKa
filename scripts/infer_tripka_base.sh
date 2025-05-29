#!/bin/bash

name=$1 # SP7
num=$2
conf_type=$3
cuda_device=$4
bs=$5
echo "dataset: $name, samples_num: $num, conf_type: $conf_type, cuda_idx: $cuda_device"

for((i=0;i<$num;i++));
do
       conf_type=$conf_type
       data_path="./tripka/examples/${conf_type}/${i}"
       cp tripka/examples/dict_charge.txt $data_path
       cp tripka/examples/dict.txt $data_path
       infer_task=$name
       results_path="results/${name}/${conf_type}/${i}"
       head_name='chembl_small'
       conf_size=1
       dict_name='dict.txt'
       charge_dict_name='dict_charge.txt'
       task_num=1
       batch_size=$bs
       model_path='checkpoint/tripka_confidence'
       loss_func="finetune_confidence_bce"
       only_polar=-1

       CUDA_VISIBLE_DEVICES=$cuda_device python ./tripka/infer.py --user-dir ./tripka ${data_path}  --task-name $infer_task --valid-subset $infer_task \
              --results-path $results_path  \
              --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
              --task tgt_pka --loss $loss_func --arch tgt_pka \
              --classification-head-name $head_name --num-classes $task_num \
              --dict-name $dict_name --charge-dict-name $charge_dict_name --conf-size $conf_size \
              --only-polar $only_polar  \
              --path $model_path/checkpoint_best.pt \
              --log-interval 50 --log-format simple --required-batch-size-multiple 1 \
              --tgt-config $model_path/model.yaml \
              --seed 42 --confidence
done
