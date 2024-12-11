
CUDA_VISIBLE_DEVICES=1 \
python3 eval.py \
    --config configs/config_MSMARCO_X_with_pretrained.yaml \
    --similarity \
    --beam_search \
    # > logs/diffXformer_ICL.txt 2>&1



