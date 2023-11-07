# llama2.openvino
This sample shows how to implement a llama-based model with OpenVINO runtime.

<img width="947" alt="MicrosoftTeams-image (2)" src="https://github.com/OpenVINO-dev-contest/llama2.openvino/assets/91237924/c210507f-1fb2-4c68-a8d9-dae945df07d3">


- Please follow the Licence on HuggingFace and get the approval from Meta before downloading llama checkpoints, for more [information](https://huggingface.co/meta-llama/Llama-2-7b-hf)

- Please notice this repository is only for a functional test and personal study, and you can try to quantize the model to further optimize the performance of it

## Requirements

- Linux, Windows
- Python >= 3.8.0
- CPU or GPU compatible with OpenVINO.
- RAM: >=32GB
- vRAM: >=8GB

## Install the requirements

    $ python3 -m venv openvino_env

    $ source openvino_env/bin/activate

    $ python3 -m pip install --upgrade pip
    
    $ pip install wheel setuptools
    
    $ pip install -r requirements.txt


## Deployment Method 1: OpenVINO IR pipeline, export IR model from HF Optimum-Intel or Transformers

**1. Export IR model**

    from Transformers:

    $ python3 export_ir.py -m 'meta-llama/Llama-2-7b-hf'

    or from Optimum-Intel:

    $ python3 export_op.py -m 'meta-llama/Llama-2-7b-hf'

**1.1.  (Optional) Export IR model with int8 weight**

    $ python3 export_ir.py -m 'meta-llama/Llama-2-7b-hf' -cw=True

**2. Run [Optimum-Intel OpenVINO pipeline](https://huggingface.co/docs/optimum/intel/inference)**

    $ python3 ir_pipeline/generate_op.py -m "./ir_model" -p "what is openvino ?" -d "CPU"

**3. (Optional) Run restructured pipeline**:

    $ python3 ir_pipeline/generate_ir.py -m "./ir_model" -p "what is openvino ?" -d "CPU"


## (Optional) Deployment Method 2: ONNX pipeline, export ONNX model from HF Optimum
- Please notice the step below will leadd large memory consumption, you have make sure your server should be with >256GB RAM

**1. Export the ONNX model from HuggingFace Optimum and convert it to OpenVINO IR**:

    $ cd onnx_pipeline

    $ optimum-cli export onnx --model meta-llama/Llama-2-7b-hf ./onnx_model/

    $ mkdir ir_model

    $ mo -m ./onnx_model/decoder_model_merged.onnx -o ./ir_model/ --compress_to_fp16

    $ rm ./onnx_model/ -rf

**2. Run restructured pipeline**:

    $ python3 generate_onnx.py -m  "meta-llama/Llama-2-7b-hf" -p "what is openvino ?" -d "CPU"


## Interactive demo

**1. Run interactive Q&A demo with Gradio**:

    $ python3 demo/qa_gradio.py -m "./ir_model" 

**2. or chatbot demo with Streamlit**:

    $ python3 export_ir.py -m 'meta-llama/Llama-2-7b-chat-hf' -o './ir_model_chat' # "-cw=True" to get model with int8 weight

    $ streamlit run demo/chat_streamlit.py -- -m './ir_model_chat'
