 accelerate launch \
    --config_file "./configs/accelerate_config.yaml" \
    trainers/diff.py \
    --config_file "./configs/train_config.json"