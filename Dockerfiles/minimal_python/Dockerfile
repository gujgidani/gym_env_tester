# Builder stage
FROM python:3.10-slim-bullseye AS builder

WORKDIR /home/app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade "numpy<2.0" tflite-runtime pymodbus

# Copy application code
COPY nn_interface.py .
COPY model_float16.tflite .
COPY model_float32.tflite .

# Final stage
FROM python:3.10-slim-bullseye

WORKDIR /home/app

# Copy only necessary files from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /home/app /home/app

#CMD ["python", "nn_interface.py"]
#Make it interactive for debuging
CMD ["bash"]