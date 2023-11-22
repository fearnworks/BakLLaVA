# path/filename: Dockerfile

# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

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
RUN apt-get install -y git vim build-essential procps net-tools tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


COPY ./Grounded-Segment-Anything /Grounded-Segment-Anything

# Create the llava conda environment with Python 3.10
RUN conda create -n llava python=3.10 -y && \
    echo "source activate llava" > ~/.bashrc

RUN /bin/bash -c "source activate llava && \
    cd /Grounded-Segment-Anything && \
    pip install -r requirements.txt && \
    cd GroundingDINO && \
    pip install -e . && \
    cd ../segment_anything && \
    pip install -e ."

COPY requirements.txt /tmp/
RUN /bin/bash -c "source activate llava && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /tmp/requirements.txt"

COPY . /workspace
# Set working directory

RUN /bin/bash -c "source activate llava && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install torch && \ 
    python3 -m pip install -e /workspace"

WORKDIR /workspace
RUN chmod +x /workspace/docker/*.sh

# Set the entrypoint
 CMD ["tini","--","tail", "-f", "/dev/null"]