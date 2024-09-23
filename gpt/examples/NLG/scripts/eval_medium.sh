PEFT=LORTA
RANK=8
LORA_ALPHA=8 
LR=0.005

python -m torch.distributed.launch --master_port 9091 --nproc_per_node=1 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 8 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./checkpoints/trained_models/GPT2_M/e2e_qv_${PEFT}_lr${LR}_rank${RANK}_alpha${LORA_ALPHA}/model.26290.pt \
    --platform local \
    --lora_qv \
    --lora_rank $RANK \
    --lora_alpha $LORA_ALPHA \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./checkpoints/trained_models/GPT2_M/e2e_qv${PEFT}_lr${LR}_rank${RANK}_alpha${LORA_ALPHA}/model.26290.pt \
    --output_file predict.26290.b10p08r4.jsonl
    
    
python src/gpt2_decode.py     --vocab ./vocab     --sample_file ./checkpoints/trained_models/GPT2_M/e2e_qv_${PEFT}_lr${LR}_rank${RANK}_alpha${LORA_ALPHA}/predict.26290.b10p08r4.jsonl     --input_file ./data/e2e/test_formatted.jsonl     --output_ref_file e2e_ref.txt     --output_pred_file e2e_pred_${PEFT}${LR}${RANK}${LORA_ALPHA}}.txt

python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred_${PEFT}${LR}${RANK}${LORA_ALPHA}.txt -p

