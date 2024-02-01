#!/bin/bash
# source /home/llm/LLM/openvino_env/bin/activate

DEVICE=CPU
WORK_PATH=/opt/intel/openvino_2023.3.0.13775/llama2.openvino/
cd $WORK_PATH

MODEL_PATH=/opt/intel/openvino_2023.3.0.13775/Model/llama-2-chat-7b
PROMPT="what is large language model (LLM)? please reply under 100 words"
python3 pipeline/generate_ir.py -m $MODEL_PATH -p "$PROMPT" -d $DEVICE
