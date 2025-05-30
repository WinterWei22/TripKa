data_path='./tripka/examples/MM/conf6'
MASTER_PORT=10152
task_name="dwar_small"
head_name='chembl_small'
weight_path='weights/tgt_pretrain_update'
n_gpu=1

# train params
seed=0
task_num=1
loss_func="finetune_mse"
dict_name='dict.txt'
charge_dict_name='dict_charge.txt'
only_polar=-1
conf_size=6
local_batch_size=2
lr=5e-5
bs=2
epoch=100
dropout=0.1
warmup=0.06
model_dir="weights/pka/tripka"

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
echo "params setting lr: $lr, bs: $bs, epoch: $epoch, dropout: $dropout, warmup: $warmup, seed: $seed"
update_freq=`expr $bs / $local_batch_size`
python -m torch.distributed.launch --use-env --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --task-name $task_name --user-dir ./tripka --train-subset train --valid-subset valid \
        --conf-size $conf_size \
        --num-workers 8 --ddp-backend=no_c10d \
        --dict-name $dict_name --charge-dict-name $charge_dict_name \
        --task tgt_pka --loss $loss_func --arch tgt_pka  \
        --classification-head-name $head_name --num-classes $task_num \
        --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
        --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size --pooler-dropout $dropout \
        --update-freq $update_freq --seed $seed \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
        --log-interval 100 --log-format simple \
        --finetune-from-model $weight_path/checkpoint_best.pt \
        --validate-interval 1 --keep-last-epochs 1 \
        --all-gather-list-size 102400 \
        --save-dir $model_dir --tensorboard-logdir $model_dir/tsb \
        --best-checkpoint-metric valid_rmse --patience 2000 \
        --only-polar $only_polar --split-mode random \
        --tgt-config  $weight_path/model.yaml \
