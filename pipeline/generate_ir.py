from transformers import AutoTokenizer
from openvino.runtime import Core, Tensor
from pathlib import Path
import numpy as np
import argparse
import time
import sys
utils_file_path = Path('.')
sys.path.append(str(utils_file_path))
from utils.memory_profile import MemConsumption


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    summation = e_x.sum(axis=-1, keepdims=True)
    return e_x / summation


def process_logits(cur_length, scores, eos_token_id, min_length=0):
    if cur_length < min_length:
        scores[:, eos_token_id] = -float("inf")
    return scores


def get_top_k_logits(scores, top_k):
    filter_value = -float("inf")
    top_k = min(max(top_k, 1), scores.shape[-1])
    top_k_scores = -np.sort(-scores)[:, :top_k]
    indices_to_remove = scores < np.min(top_k_scores)
    filtred_scores = np.ma.array(scores,
                                 mask=indices_to_remove,
                                 fill_value=filter_value).filled()
    return filtred_scores


def generate_sequence(sampling, input_ids, attention_mask, eos_token_id,
                      max_sequence_length):
    past_key_values = None
    prompt_len = len(input_ids[0])
    count = 0
    other_latency = 0
    position_ids = None
    if "position_ids" in input_names:
        position_ids = np.arange(0, input_ids.shape[1], dtype=np.int64)
        position_ids = np.expand_dims(position_ids,axis=0)
    while True:
        inputs = {}
        if past_key_values is not None:
            inputs = dict(zip(key_value_input_names, past_key_values))
            inputs["input_ids"] = input_ids[:, -1:]
            cur_input_len = 1
        else:
            inputs["input_ids"] = input_ids
            shape_input_ids = input_ids.shape
            num_attention_heads = 1
            for input_name in key_value_input_names:
                model_inputs = model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = shape_input_ids[0] * num_attention_heads
                if shape[2].is_dynamic:
                    shape[2] = 0
                if shape[1].is_dynamic:
                    shape[1] = 0
                inputs[input_name] = Tensor(model_inputs.get_element_type(),
                                            shape.get_shape())
        cur_input_len = len(inputs["input_ids"][0])
        if "attention_mask" in input_names and attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        if position_ids is not None:
            inputs["position_ids"] = position_ids
            position_ids = np.copy(position_ids[..., -1:]) + 1
        before = time.perf_counter()
        request.start_async(inputs, share_inputs=True)
        request.wait()
        after = time.perf_counter()
        if count == 0:
            first_latency = after - before
        else:
            other_latency += after - before
        count += 1
        logits = request.get_tensor("logits").data
        past_key_values = tuple(
            request.get_tensor(key).data for key in key_value_output_names)
        next_token_logits = logits[:, cur_input_len - 1, :]
        # pre-process distribution
        next_token_scores = process_logits(len(input_ids[0]),
                                           next_token_logits, eos_token_id)
        top_k = 20
        next_token_scores = get_top_k_logits(next_token_scores, top_k)
        # get next token id
        if sampling:
            probs = softmax(next_token_scores)
            next_tokens = np.random.choice(probs.shape[-1],
                                           1,
                                           p=probs[0],
                                           replace=True)
        else:
            next_tokens = np.argmax(next_token_scores, axis=-1)
        # break the loop if max length or end of text token is reached
        if (len(input_ids[0]) - prompt_len
                ) == max_sequence_length or next_tokens == eos_token_id:
            break
        else:
            input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
            attention_mask = np.concatenate(
                (attention_mask, [[1] * len(next_tokens)]), axis=-1)
    return input_ids, count, (first_latency, other_latency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_path',
                        required=True,
                        type=str,
                        help='Required. path of IR model and tokenizer')
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
    parser.add_argument('-s',
                        '--sampling',
                        default=True,
                        required=False,
                        type=bool,
                        help='sampling or not')
    args = parser.parse_args()

    num_pkv = 2
    core = Core()
    mem_consumption = MemConsumption()
    ir_model_path = Path(args.model_path)
    ir_model = ir_model_path / "openvino_model.xml"

    print(" --- load tokenizer --- ")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    inputs = tokenizer(args.prompt, return_tensors="np",
                       add_special_tokens=False)
    input_len = len(inputs["input_ids"][0])

    print(" --- reading model --- ")
    # read the model and corresponding weights from file
    model = core.read_model(ir_model)
    input_names = {
        key.get_any_name(): idx
        for idx, key in enumerate(model.inputs)
    }
    output_names = {
        key.get_any_name(): idx
        for idx, key in enumerate(model.outputs)
    }
    key_value_input_names = [key for key in input_names if "key_values" in key]
    key_value_output_names = [key for key in output_names if "present" in key]

    max_rss_mem_consumption = ''
    max_shared_mem_consumption = ''
    mem_consumption.start_collect_mem_consumption_thread()
    mem_consumption.start_collect_memory_consumption()

    print(" --- model compiling --- ")
    # compile the model for CPU devices
    request = core.compile_model(
        model=model, device_name=args.device).create_infer_request()

    print(" --- start generating --- ")
    start = time.perf_counter()
    output_ids, num_tokens, latencies = generate_sequence(
        args.sampling,
        inputs["input_ids"],
        inputs["attention_mask"],
        eos_token_id=tokenizer.eos_token_id,
        max_sequence_length=args.max_sequence_length,
    )
    end = time.perf_counter()
    mem_consumption.end_collect_momory_consumption()
    max_rss_mem_consumption, max_shared_mem_consumption = mem_consumption.get_max_memory_consumption()
    mem_consumption.clear_max_memory_consumption()
    output_text = " "
    # Convert IDs to words and make the sentence from it

    print(" --- text decoding --- ")
    output_text = tokenizer.batch_decode(output_ids,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]
    print(f"Response: {output_text}")

    print(" --- Benchmarking --- ")
    print(f"Input length: {input_len} tokens")
    print(
        f"Generated {num_tokens} tokens in {end - start:.2f} s on {args.device}")
    print(
        f"Maximum rss memory counsuption: {max_rss_mem_consumption:.2f} MB, Maximum shared memory counsuption: {max_shared_mem_consumption:.2f}  MB")
    print(
        f"First inference latency: {1000*latencies[0]:.2f} ms/token, Other inference latency {1000*latencies[1]/(num_tokens-1):.2f} ms/token in average")
    mem_consumption.end_collect_mem_consumption_thread()
