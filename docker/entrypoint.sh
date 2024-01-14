#!/bin/bash
wandb login 8d9f1df280c9e9676b3dcb4435edcfb6f2920d1b
mkdir -p /root/.cache/torch/kernels
chmod 777 /root/.cache/torch/kernels

# If no command is specified, start a shell
if [ -z "$1" ]; then
    exec bash
else
    exec "$@"
fi
