export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python ../../train_rlpd.py "$@" \
    --exp_name=banana_pick_place \
    --checkpoint_path=../../experiments/banana_pick_place/test3 \
    --actor \