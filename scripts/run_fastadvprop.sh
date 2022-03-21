GPU_ID=0,1,2,3
run_epochs=105
run_schedule='30 60 90 100'
warm=0
norm_layer=mixbn
train_batch=256
lr=0.1
attack_iter=1
attack_eps=1.0
model=resnet50

shared_params="--attack-iter ${attack_iter} --attack-epsilon ${attack_eps} --attack-step-size 1 -a ${model} --num_classes 1000 --data ~/data/ImageNet --train-batch ${train_batch} --lr ${lr} --epochs ${run_epochs} --schedule ${run_schedule} --gamma 0.1 --lr_schedule step --norm_layer ${norm_layer} --warm ${warm}"
postfix=_batch${train_batch}_lr${lr}

function run_fast_advprop() {
    multi_clean_strategy=4:1
    lr_strategy=shared:1,clean:1,adv:1
    loss_strategy=clean:1,attack:0.5,adv:0.5

    python imagenet.py ${shared_params} \
        --gpu-id ${GPU_ID} \
        -c exp/imagenet/epoch${run_epochs}/${model}-fastadvprop_mcs.${multi_clean_strategy}_ls.${lr_strategy}_los.${loss_strategy}_${postfix} \
        --attacker_type pgd \
        --multi_clean_strategy ${multi_clean_strategy} \
        --lr_strategy ${lr_strategy} --loss_strategy ${loss_strategy} \
        --reuse_mid_grad --attack_in_train \
        --original_attack \
        --exact_same_training_budget \
        --shuffle_before_train_adv
}

run_fast_advprop
