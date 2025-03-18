export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python ../../train_rlpd.py "$@" \
    --exp_name=usb_pickup_insertion \
    --checkpoint_path=../../experiments/usb_pickup_insertion/test3 \
    --actor \