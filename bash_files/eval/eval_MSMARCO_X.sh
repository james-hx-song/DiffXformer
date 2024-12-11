
CUDA_VISIBLE_DEVICES=0 \
python3 eval.py \
    --config configs/config_MSMARCO_X_with_pretrained.yaml \
    --similarity \
    # > logs/diffXformer_ICL.txt 2>&1



