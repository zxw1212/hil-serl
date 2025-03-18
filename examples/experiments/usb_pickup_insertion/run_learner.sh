export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8 && \
python ../../train_rlpd.py "$@" \
    --exp_name=usb_pickup_insertion \
    --checkpoint_path=../../experiments/usb_pickup_insertion/test3 \
    --demo_path=../../experiments/usb_pickup_insertion/demo_data/usb_pickup_insertion_10_demos_2025-03-14_10-15-57.pkl \
    --learner \