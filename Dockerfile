FROM nvcr.io/nvidia/pytorch:22.10-py3

# ENV Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/include/"

# Local libraries
WORKDIR /project

COPY ./cinnamon_core /cinnamon_core
RUN pip3 install -r /cinnamon_core/requirements.txt

COPY ./cinnamon_generic /cinnamon_generic
RUN pip3 install -r /cinnamon_generic/requirements.txt

COPY ./cinnamon_th /cinnamon_th

COPY ./bert-natural-explanations /bert-natural-explanations
RUN pip3 install -r /bert-natural-explanations/requirements.txt
WORKDIR /bert-natural-explanations

ENV NUMBA_CACHE_DIR="/bert-natural-explanations/numba/"
ENV PYTHONPATH="$PYTHONPATH:/bert-natural-explanations/:/cinnamon_core/:/cinnamon_generic/:/cinnamon_th/"
ENV WANDB_API_KEY=""
ENV WANDB_DOCKER="nle"

# Project-specific dependencies
RUN pip3 install -r requirements.txt