FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV CUDA_HOME /usr/local/cuda
ENV CUDA_MODULE_LOADING=LAZY

RUN mkdir tmp
# Install system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt into the image
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip3 install -r /tmp/requirements.txt

# Set CUDA_LAUNCH_BLOCKING environment variable
ENV CUDA_LAUNCH_BLOCKING=1

# Create a workspace directory
RUN mkdir workspace
COPY ./ /workspace
