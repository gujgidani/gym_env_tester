import onnxruntime as ort
import numpy as np
import opcua

# Load the ONNX model
onnx_model_path = "model-4.onnx"
session = ort.InferenceSession(onnx_model_path)

# Prepare two input tensors of shape (1,6)
input_tensor_1 = np.random.rand(1,6).astype(np.float32)  # Example data
#make random numbers between 20 and 150 in (1,6) shape
input_tensor_2 = np.random.randint(20,150, size=(1,6)).astype(np.float32)  # Example data
#input_tensor_2 = np.random.rand(1,6).astype(np.float32)  # Example data

print(input_tensor_1, input_tensor_2)

# Get the input names
input_name_1 = session.get_inputs()[0].name
input_name_2 = session.get_inputs()[1].name

# Assuming the model has two inputs and you've prepared input_tensor_1 and input_tensor_2
input_dict = {input_name_1: input_tensor_1, input_name_2: input_tensor_2}

# Get the name of the output
output_name = [output.name for output in session.get_outputs()]

# Run inference
outputs = session.run(output_name, input_dict)

# Process the output
# outputs will be a list of numpy arrays containing the results
print(outputs[0][0])
#choose the maximum values's index
print(np.argmax(outputs[0][0]))