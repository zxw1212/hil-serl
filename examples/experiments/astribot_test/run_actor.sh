export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=astribot_test \
    --checkpoint_path=../../experiments/astribot_test/test1 \
    --actor \