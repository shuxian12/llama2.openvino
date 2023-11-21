# llama2.openvino
This sample shows how to implement a llama-based model with OpenVINO runtime.

<img width="947" alt="MicrosoftTeams-image (2)" src="https://github.com/OpenVINO-dev-contest/llama2.openvino/assets/91237924/c210507f-1fb2-4c68-a8d9-dae945df07d3">


- Please follow the Licence on HuggingFace and get the approval from Meta before downloading llama checkpoints, for more [information](https://huggingface.co/meta-llama/Llama-2-7b-hf)

- Please notice this repository is only for a functional test and personal study.

## Requirements

- Linux, Windows
- Python >= 3.9.0
- CPU or GPU compatible with OpenVINO.
- RAM: >=16GB
- vRAM: >=8GB

## Install the requirements

    $ python3 -m venv openvino_env

    $ source openvino_env/bin/activate

    $ python3 -m pip install --upgrade pip
    
    $ pip install wheel setuptools
    
    $ pip install -r requirements.txt


## Q&A Pipeline

**1. Export IR model**

from Transformers:

    $ python3 export_ir.py -m 'meta-llama/Llama-2-7b-hf'

or from Optimum-Intel:

    $ python3 export_op.py -m 'meta-llama/Llama-2-7b-hf'

or for #GPTQ model:

    $ python3 export_op.py -m 'TheBloke/Llama-2-7B-Chat-GPTQ'

**1.1.  (Optional) quantize local IR model with #int8 or #int4 weight**

    $ python3 quantize.py -m 'ir_model' -p 'int4'

For more information on quantization configuration, please refer to [weight compression](https://github.com/openvinotoolkit/nncf/blob/release_v270/docs/compression_algorithms/CompressWeights.md)

**2.  Run pipeline**

[Optimum-Intel OpenVINO pipeline](https://huggingface.co/docs/optimum/intel/inference):

    $ python3 ir_pipeline/generate_op.py -m "./ir_model" -p "what is openvino ?" -d "CPU"

or Restructured pipeline:

    $ python3 ir_pipeline/generate_ir.py -m "./ir_model" -p "what is openvino ?" -d "CPU"

## Interactive demo

**1. Run interactive Q&A demo with Gradio**:

    $ python3 demo/qa_gradio.py -m "./ir_model" 

**2. or chatbot demo with Streamlit**:

    $ python3 export_op.py -m 'meta-llama/Llama-2-7b-chat-hf' -o './ir_model_chat'

    $ streamlit run demo/chat_streamlit.py -- -m './ir_model_chat'
