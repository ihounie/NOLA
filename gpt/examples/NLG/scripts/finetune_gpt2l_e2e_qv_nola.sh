PEFT=LORA
RANK=128 
LORA_ALPHA=2 
LR=0.01

python -m torch.distributed.launch --master_port 9092 --nproc_per_node=1 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --log_interval 50 \
    --eval_interval 5000 \
    --train_batch_size 4 \
    --grad_acc 4 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.lg \
    --init_checkpoint ./pretrained_checkpoints/gpt2-large-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --weight_decay 0.0 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 50000 \
    --label_smooth 0.1 \
    --lora_qv \
    --lora_rank $RANK \
    --lora_alpha $LORA_ALPHA \
    --lr $LR \
    --work_dir ./checkpoints/trained_models/GPT2_M/e2e_qv_${PEFT}_lr${LR}_rank${RANK}_alpha${LORA_ALPHA} \
    --random_seed 110
    
    


