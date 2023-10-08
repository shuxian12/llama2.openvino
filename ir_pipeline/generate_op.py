from transformers import LlamaTokenizer
from optimum.intel.openvino import OVModelForCausalLM
import time
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-h',
                    '--help',
                    action='help',
                    help='Show this help message and exit.')
parser.add_argument('-m',
                    '--model_id',
                    required=True,
                    type=str,
                    help='Required. hugging face model id or local model path')
parser.add_argument('-p',
                    '--prompt',
                    required=True,
                    type=str,
                    help='Required. prompt sentence')
parser.add_argument('-l',
                    '--max_sequence_length',
                    default=128,
                    required=False,
                    type=int,
                    help='maximun lengh of output')
parser.add_argument('-d',
                    '--device',
                    default='CPU',
                    required=False,
                    type=str,
                    help='device for inference')
args = parser.parse_args()

print(" --- load tokenizer --- ")
tokenizer = LlamaTokenizer.from_pretrained(args.model_id)

try:
    print(" --- use local model --- ")
    model = OVModelForCausalLM.from_pretrained(args.model_id, compile=False, device=args.device)
except:
    print(" --- use remote model --- ")
    model = OVModelForCausalLM.from_pretrained(args.model_id, compile=False, device=args.device, export=True)
model.compile()

inputs = tokenizer(args.prompt, return_tensors="pt")

print(" --- start generating --- ")
start = time.perf_counter()
generate_ids = model.generate(inputs.input_ids,
                                 max_length=args.max_sequence_length)
end = time.perf_counter()

print(" --- text decoding --- ")
output_text = tokenizer.batch_decode(generate_ids,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
print(f"Generation took {end - start:.3f} s on {args.device}")
print(f"Response: {output_text}")
