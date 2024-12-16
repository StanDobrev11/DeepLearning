# Use the TensorFlow GPU image as the base
FROM tensorflow/tensorflow:2.10.1-gpu

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies for OpenAI Gym rendering
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    x11-utils \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements.txt to the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Jupyter Lab's default port
EXPOSE 8888

# Command to start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
