
CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
    --config configs/config_LogiQA_X_with_pretrained.yaml \
    --fine_tuning \
    # > logs/diffXformer_ICL.txt 2>&1



