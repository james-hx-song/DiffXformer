
CUDA_VISIBLE_DEVICES=1 \
python3 train.py \
    --config configs/config_MSMARCO_DX_with_pretrained.yaml \
    --fine_tuning \
    # > logs/diffXformer_ICL.txt 2>&1



