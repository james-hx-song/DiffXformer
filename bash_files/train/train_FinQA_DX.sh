
CUDA_VISIBLE_DEVICES=1 \
python3 train.py \
    --config configs/config_FinQA_DX.yaml \
    # > logs/diffXformer_ICL.txt 2>&1


