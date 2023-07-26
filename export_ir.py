from transformers import LlamaTokenizer
from optimum.intel.openvino import OVModelForCausalLM
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_id',
                        required=True,
                        type=str,
                        help='Required. hugging face model id')
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        type=str,
                        help='Required. path to save the ir model')

    args = parser.parse_args()

    model_path = Path(args.output)

    ov_model = OVModelForCausalLM.from_pretrained(args.model_id,
                                                  compile=False,
                                                  from_transformers=True)
    ov_model.save_pretrained(model_path)