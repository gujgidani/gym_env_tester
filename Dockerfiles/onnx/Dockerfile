# Builder stage
FROM python:3.10-slim-bullseye AS builder

WORKDIR /home/app

# Clean the local repository cache and update the package list
RUN apt-get update && apt-get install -y \
    build-essential git\
    && rm -rf /var/lib/apt/lists/*


RUN git clone --recursive https://github.com/Microsoft/onnxruntime.git
WORKDIR /home/app/onnxruntime

RUN python3 -m pip install cmake && \
    which cmake

RUN ./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --allow_running_as_root

COPY model.onnx .


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