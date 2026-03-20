FROM dgx-spark-base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
# torch and torchvision are already provided by the base image
COPY requirements.txt .
RUN pip install --no-cache-dir \
    diffusers \
    einops \
    gradio==6.1.0 \
    jaxtyping \
    numpy \
    omegaconf \
    peft \
    Pillow \
    transformers \
    spaces

# Copy application source
COPY . .

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
