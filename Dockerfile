FROM python
LABEL authors="wagnertamas"

#install numpy and onnx
RUN pip install onnxruntime numpy opcua

#copy onnx model
COPY model.onnx /model.onnx
COPY onnx_main.py /onnx_main.py

#run the onnx model
CMD ["python", "onnx_main.py"]