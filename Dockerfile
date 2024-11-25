# Use the official Miniconda image as the base
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the environment file for data science tools
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy && \
    conda init bash

# Activate the environment
ENV PATH /opt/conda/envs/venv/bin:$PATH

# Expose the Jupyter Lab default port
EXPOSE 8888

# Start Jupyter Lab as the default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
