export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7 && \
python ../../train_rlpd.py "$@" \
    --exp_name=astribot_test \
    --checkpoint_path=../../experiments/astribot_test/test1 \
    --demo_path=../../experiments/astribot_test/demo_data_with_bc/astribot_test_10_demos_2025-04-15_11-26-28.pkl \
    --learner \