date_time=$(python -c "from nntool.utils import get_current_time; print(get_current_time())")
export OUTPUT_PATH=outputs/train_3d_energy_atomic/$date_time
mkdir -p "$OUTPUT_PATH"

python -m scripts.3d.train_3d_energy_distributed amber_pbsa \
    --trainer.data-split-json-path checkpoints/amber_pbsa/data_split.json \
    --trainer.train-dataset-extra-config.neighbor-list-cutoff 15 \
    --trainer.eval-dataset-extra-config.neighbor-list-cutoff 15 \
    --trainer.train-num-steps 32000
