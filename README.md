# llama_openvino
This sample shows how to implement a llama-based model with OpenVINO runtime.

- ***Please follow the Licence on HuggingFace and get the approval from Meta before downloading llama checkpoints***

- ***Please notice this repository is only for a functional test and personal study, and you can try to quantize the model to further optimize the performance of it***

## How to run it?
1. Install the requirements:

    ```$pip install -r requirements.txt```


### Option 1: OpenVINO IR pipeline
2. Run [Optimum-Intel OpenVINO pipeline](https://huggingface.co/docs/optimum/intel/inference) and export the IR model

    ```$cd ir_pipeline```

    ```$python3 generate_op.py -m "decapoda-research/llama-7b-hf" -p "what is openvino ?" -d "CPU"``` 

3. (Optional) Run restructured pipeline:

    ```$python3 generate_ir.py -m "decapoda-research/llama-7b-hf" -p "what is openvino ?" -d "CPU"```

### Option 2: ONNX pipeline, directly load a merged ONNX model to OpenVINO runtime

2. Export the ONNX model from HuggingFace pipeline:

    ```$cd onnx_pipeline```

    ```$optimum-cli export onnx --model decapoda-research/llama-7b-hf ./onnx_model/```

3. Run restructured pipeline:

    ```$python3 generate_onnx.py -m  "decapoda-research/llama-7b-hf" -p "what is openvino ?" -d "CPU"```


### Option 3: Interactive demo with Gradio

2. Run interactive demo

    ```$python3 gradio_demo.py -m "decapoda-research/llama-7b-hf" ```
