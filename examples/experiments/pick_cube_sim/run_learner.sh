export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd_sim.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=test3 \
    --demo_path=../../../demo_data/pick_cube_sim_20_demos_2025-02-26_11-36-30_with_classifier.pkl \
    --learner \
