from transformers import LlamaTokenizer
from openvino.runtime import Core
import numpy as np
import argparse
import time

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
    filtred_scores = np.ma.array(scores, mask=indices_to_remove,
                                 fill_value=filter_value).filled()
    return filtred_scores

def generate_sequence(input_ids, attention_mask, eos_token_id, max_sequence_length=128,
                      dynamic_shapes=True):
    while True:
        cur_input_len = len(input_ids[0])
        if not dynamic_shapes:
            pad_len = max_sequence_length - cur_input_len
            model_input_ids = np.concatenate((input_ids, [[eos_token_id] * pad_len]), axis=-1)
            model_input_attention_mask = np.concatenate((attention_mask, [[0] * pad_len]), axis=-1)
            
        else:
            model_input_ids = input_ids
            model_input_attention_mask = attention_mask
        outputs = compiled_model({"input_ids": model_input_ids, "attention_mask": model_input_attention_mask})[output_key]
        next_token_logits = outputs[:, cur_input_len - 1, :]
        # pre-process distribution
        next_token_scores = process_logits(cur_input_len,
                                           next_token_logits, eos_token_id)
        top_k = 20
        next_token_scores = get_top_k_logits(next_token_scores, top_k)
        # get next token id
        next_tokens  = np.argmax(next_token_scores, axis=-1)
        # break the loop if max length or end of text token is reached
        if cur_input_len == max_sequence_length or next_tokens == eos_token_id:
            break
        else:
            input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
            attention_mask = np.concatenate((attention_mask, [[1] * len(next_tokens)]), axis=-1)
    return input_ids


if __name__ == "__main__":
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
  parser.add_argument('-m', '--ov_model_path', required=True, type=str,
                      help='Required. openvino model path')
  parser.add_argument('-t', '--tokenizer_path', required=True, type=str,
                      help='Required. tokenizer path')
  parser.add_argument('-p', '--prompt', required=True, type=str,
                      help='Required. prompt sentence')
  parser.add_argument('-l', '--max_sequence_length', default=128, required=False, type=int,
                      help='Required. maximun lengh of output')
  parser.add_argument('-d', '--dynamic_shape', default=True, required=False, type=bool,
                      help='Required. prompt sentence')
  args = parser.parse_args()
  
  core = Core()
  # read the model and corresponding weights from file
  model = core.read_model(args.ov_model_path)
  if args.dynamic_shape == False:
    model.reshape({0: [1, args.max_sequence_length], 1: [1, args.max_sequence_length]})

  # compile the model for CPU devices
  compiled_model = core.compile_model(model=model, device_name="CPU")

  # get output tensors
  output_key = compiled_model.output(0)
  
  tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
  inputs = tokenizer(args.prompt, return_tensors="np")
  
  start = time.perf_counter()
  output_ids = generate_sequence(inputs["input_ids"], inputs["attention_mask"], eos_token_id=tokenizer.eos_token_id, max_sequence_length=args.max_sequence_length,  dynamic_shapes=args.dynamic_shape)
  end = time.perf_counter()
  output_text = " "
  # Convert IDs to words and make the sentence from it
  output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  print(f"Generation took {end - start:.3f} s")
  print("Question: ", args.prompt)
  print()
  print(f"Answer:{output_text}")