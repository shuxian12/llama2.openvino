# llama_openvino
This sample shows how to implement a llama-based model with OpenVINO runtime.

**Please notice this repository is only for a functional test, and you can try to quantize the model to further optimize the performance of it**

## How to run it?
1. Install the requirements:

    ```$pip install -r requirements.txt```


### Option 1: OpenVINO IR pipeline
2. Run [Optimum-Intel OpenVINO pipeline](https://huggingface.co/docs/optimum/intel/inference) and export the IR model

    ```$cd ir_pipeline```

    ```$python3 generate_op.py -m "{HuggingFace model id}" -p "what is openvino ?" -d "CPU"``` 

3. (Optional) Run restructured pipeline:

    ```$python3 generate_ir.py -m "{HuggingFace model id}" -p "what is openvino ?" -d "CPU"```

### Option 2: ONNX pipeline, directly load a merged ONNX model to OpenVINO runtime

2. Export the ONNX model from HuggingFace pipeline:

    ```$cd onnx_pipeline```

    ```$optimum-cli export onnx --model {HuggingFace model id} ./onnx_model/```

    For example: optimum-cli export onnx --model xxx/llama-7b-hf ./llama_model/"

    ***please follow the Licence on HuggingFace and get the approval from Meta before downloading llama checkpoints***

3. Run restructured pipeline:

    ```$python3 generate_onnx.py -m  "{HuggingFace model id}" -p "what is openvino ?" -d "CPU"```


### Option 3: Interactive demo with Gradio

2. Run interactive demo

    ```$python3 gradio_demo.py -m "{HuggingFace model id}" ```