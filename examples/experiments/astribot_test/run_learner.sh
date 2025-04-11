export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8 && \
python ../../train_rlpd.py "$@" \
    --exp_name=astribot_test \
    --checkpoint_path=../../experiments/astribot_test/test1 \
    --demo_path=../../experiments/astribot_test/demo_data/astribot_test_10_demos_2025-04-11_14-37-06.pkl \
    --learner \