ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# fetch the key refer to https://forums.developer.nvidia.com/t/18-04-cuda-docker-image-is-broken/212892/9
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 32
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="(dirname(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MIM
RUN pip install openmim

# Install MMPretrain
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmpretrain.git
WORKDIR ./mmpretrain
RUN mim install --no-cache-dir -e .

RUN pip install wandb && pip install tensorboard && pip install torch-model-archiver==0.6.1

## Below shouldn't be needed to package multitask models with newer versions of mmpretrain.
# COPY extra/mmpretrainMulti2torchserve.py tools/torchserve/mmpretrainMulti2torchserve.py
# COPY extra/mmpretrain_multitask_handler.py tools/torchserve/mmpretrain_multitask_handler.py
# COPY extra/multiTask_image_classification.py mmpretrain/apis/multiTask_image_classification.py
# COPY extra/baseMulti.py mmpretrain/apis/baseMulti.py
# COPY extra/modelMulti.py mmpretrain/apis/modelMulti.py
# COPY extra/transform_utils.py mmpretrain/datasets/transforms/modelMulti.py


COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]