# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    TRANSFORMERS_CACHE=/models \ 
    CUDA_HOME=/usr/local/cuda

# Install Miniconda for Conda management
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /miniconda && \
    rm /miniconda.sh && \
    /miniconda/bin/conda clean -tip

# Add Miniconda to PATH
ENV PATH=/miniconda/bin:$PATH

# Install necessary OS packages
RUN apt-get install -y git vim build-essential procps net-tools tini ffmpeg libsm6 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt from the build context
COPY ./webui/requirements.txt /tmp/
COPY ./webui/requirements-ui.txt /tmp/

# Create the llava conda environment
RUN conda create -n bakllava_int python=3.10 -y && \
    echo "source activate bakllava_int" > ~/.bashrc && \
    /bin/bash -c "source activate bakllava_int && \
    pip install --upgrade pip && \
    pip install torch==2.0.1+cu117 --find-links https://download.pytorch.org/whl/cu117/torch_stable.html && \
    pip install --find-links https://download.pytorch.org/whl/cu117/torch_stable.html torchvision==0.15.2+cu117 && \
    pip install -r /tmp/requirements.txt && \ 
    pip install -r /tmp/requirements-ui.txt"

COPY . /workspace
# Set working directory
WORKDIR /workspace
# Make the docker entrypoint scripts executable
RUN chmod +x /workspace/docker/*.sh

# Set the entrypoint to use tini
CMD ["tini", "--", "tail", "-f", "/dev/null"]