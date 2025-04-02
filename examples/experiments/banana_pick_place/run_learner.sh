export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8 && \
python ../../train_rlpd.py "$@" \
    --exp_name=banana_pick_place \
    --checkpoint_path=../../experiments/banana_pick_place/test3 \
    --demo_path=../../experiments/banana_pick_place/demo_data_for_sac/banana_pick_place_20_demos_2025-04-01_10-23-54.pkl \
    --learner \