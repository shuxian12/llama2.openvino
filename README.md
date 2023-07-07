# llama_openvino
This sample shows how to implement a llama-based model with OpenVINO runtime.

**Please notice this repository is only for a functional test, and you can try to quantize the model to further optimize the performance of it**

## How to run it?
1. Install the requirements:

    ```$pip install -r requirements.txt```

2. Export the ONNX model from HuggingFace pipeline:

    ```$optimum-cli export onnx --model {HuggingFace model id} ./onnx_model/```

    For example: optimum-cli export onnx --model xxx/llama-7b-hf ./llama_model/"

    ***please follow the Licence on HuggingFace and get the approval from Meta before downloading llama checkpoints***

4. Run restructured native OpenVINO pipeline:

    ```$python3 generate_ov.py -m  "{HuggingFace model id}" -p "what is openvino ?" ```

4. (Optional) Run [Optimum-Intel OpenVINO pipeline](https://huggingface.co/docs/optimum/intel/inference)

    ```$python3 generate_op.py -m "{HuggingFace model id}" -p "what is openvino ?" ```