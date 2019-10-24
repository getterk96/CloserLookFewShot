#mkdir outputs
#mkdir records

method=baseline
gpu=4

mode=train
model=ResNet18

optimizer=SGD
entropy=0
num_classes=200
train_num_way=5
train_num_shot=5
test_num_way=5
test_num_shot=5
test_num_query=15

# stage-1
#model_id=$(date +"%Y%m%d_%H%M")
model_id=20191017_1621
train_dataset=miniImagenet
train_loss_type=None
train_temperature=30
train_margin=0
train_lr=0.1
test_start_epoch=0.75
resume=1
shake=0
shake_forward=1
shake_backward=1
shake_picture=1
aug=1
episodic=1
curriculum=0

# stage-2
exp_id=1
test_dataset=miniImagenet
test_loss_type=dist
test_temperature=5
test_margin=0
test_lr=0.05
test_epoch=600
fast_adapt=0

save_freq=25
start_epoch=0
stop_epoch=-1

vis_log=/home/gaojinghan/closer/${train_dataset}_vis_log_exp
checkpoint_dir=${train_num_way}w_${train_num_shot}s_${train_loss_type}_t${train_temperature}
if [ ${episodic} == 1 ]; then
  checkpoint_dir="${checkpoint_dir}_episodic"
  if [ ${entropy} == 1 ]; then
    checkpoint_dir="${checkpoint_dir}_entropy"
  fi
fi
if [ ${curriculum} == 1 ]; then
  checkpoint_dir="${checkpoint_dir}_curriculum_30-5"
fi
if [ ${shake} == 1 ]; then
  checkpoint_dir="${checkpoint_dir}_shake"
  if [ ${shake_forward} == 1 ]; then checkpoint_dir="${checkpoint_dir}_f"; fi
  if [ ${shake_backward} == 1 ]; then checkpoint_dir="${checkpoint_dir}_b"; fi
  if [ ${shake_picture} == 1 ]; then checkpoint_dir="${checkpoint_dir}_p"; fi
fi
if [ ${aug} == 1 ]; then
  checkpoint_dir="${checkpoint_dir}_aug"
fi
checkpoint_dir=${checkpoint_dir}_model-${model_id}
tag=${checkpoint_dir}_${test_loss_type}_t${test_temperature}
if [ ${fast_adapt} == 1 ]; then
  tag="${tag}_fast-adapt"
fi
tag=${tag}_exp-${exp_id}

export CUDA_VISIBLE_DEVICES=${gpu}

if [ ${mode} == 'train' ]; then
  echo -----back up methods/${method}.py-----
  now=$(date +"%Y%m%d_%T")
  cp methods/${method}.py methods/bak/${method}_${now}.py
  cp run.py methods/bak/run_${now}.py
fi

cmd="
    python run.py
    --mode ${mode}
    --train_dataset ${train_dataset}
    --test_dataset ${test_dataset}
    --model ${model}
    --optimizer ${optimizer}
    --num_classes ${num_classes}
    --train_num_way ${train_num_way}
    --train_num_shot ${train_num_shot}
    --test_num_way ${test_num_way}
    --test_num_shot ${test_num_shot}
    --test_num_query ${test_num_query}
    --train_loss_type ${train_loss_type}
    --test_loss_type ${test_loss_type}
    --train_temperature ${train_temperature}
    --test_temperature ${test_temperature}
    --train_margin ${train_margin}
    --test_margin ${test_margin}
    --train_lr ${train_lr}
    --test_lr ${test_lr}

    --save_freq ${save_freq}
    --start_epoch ${start_epoch}
    --test_start_epoch ${test_start_epoch}
    --test_epoch ${test_epoch}
    --stop_epoch ${stop_epoch}
    --vis_log ${vis_log}
    --tag ${tag}
    --checkpoint_dir ${checkpoint_dir}
    "

if [ ${resume} == 1 ]; then cmd="${cmd} --resume"; fi
if [ ${shake} == 1 ]; then cmd="${cmd} --shake"; fi
if [ ${aug} == 1 ]; then cmd="${cmd} --train_aug"; fi
if [ ${episodic} == 1 ]; then cmd="${cmd} --episodic"; fi
if [ ${curriculum} == 1 ]; then cmd="${cmd} --curriculum"; fi
if [ ${episodic} == 1 ] && [ ${entropy} == 1 ]; then cmd="${cmd} --entropy"; fi
if [ ${shake} == 1 ] && [ ${shake_forward} == 1 ]; then cmd="${cmd} --shake_forward"; fi
if [ ${shake} == 1 ] && [ ${shake_backward} == 1 ]; then cmd="${cmd} --shake_backward"; fi
if [ ${shake} == 1 ] && [ ${shake_picture} == 1 ]; then cmd="${cmd} --shake_picture"; fi
if [ ${fast_adapt} == 1 ]; then cmd="${cmd} --fast_adapt"; fi

if [ ${mode} == 'test' ]; then $cmd;
else
  nohup $cmd >> outputs/nohup_${train_dataset}_${test_dataset}_${tag}.output &
  train_pid=$!
  echo -----cmd------
  echo ${cmd}
  echo ----output----
  echo "tail -f outputs/nohup_${train_dataset}_${test_dataset}_${tag}.output"

  echo -----tag------
  echo Start task of ${train_dataset}_${test_dataset}_${tag}...
  echo $cmd >> records/${train_dataset}_${test_dataset}_${tag}_task.record
  echo gpu=${gpu} >> records/${train_dataset}_${test_dataset}_${tag}_task.record
  echo train_pid=${train_pid} >> records/${train_dataset}_${test_dataset}_${tag}_task.record
  cat records/${train_dataset}_${test_dataset}_${tag}_task.record
  echo -------------- >> records/${train_dataset}_${test_dataset}_${tag}_task.record
fi