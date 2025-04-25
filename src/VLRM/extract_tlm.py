import os
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          MllamaForConditionalGeneration)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--tlm_id', type=str, required=True)
    parser.add_argument('--vlm_id', type=str, default="")
    parser.add_argument('--seq_id', type=str, default="")
    parser.add_argument('--output_path', type=str, default="models/merged")
    parser.add_argument('--source', type=str, default="vlm")

    args = parser.parse_args()

    return args


def replace_tlm_with_vlm(vlm_id, tlm_id, output_path):
    vlm = MllamaForConditionalGeneration.from_pretrained(
        vlm_id,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    tlm = AutoModelForCausalLM.from_pretrained(
        tlm_id,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    tlm.resize_token_embeddings(vlm.language_model.get_input_embeddings().num_embeddings)

    vlm_state_dict = vlm.state_dict()
    tlm_state_dict = tlm.state_dict()

    cross_attention_layers = vlm.config.text_config.cross_attention_layers

    for tlm_param_name in tlm_state_dict.keys():
        if "layers." in tlm_param_name:
            layer_idx = int(tlm_param_name.split("layers.")[1].split(".")[0])  # Extract layer index
            vlm_layer_idx = layer_idx
            for cross_attention_layer in sorted(cross_attention_layers):
                if vlm_layer_idx >= cross_attention_layer:
                    vlm_layer_idx += 1
                else:
                    break
            tlm_state_dict[tlm_param_name] = vlm_state_dict[
                "language_model." + tlm_param_name.replace(f"layers.{layer_idx}.", f"layers.{vlm_layer_idx}.")
            ]
        elif "lm_head" in tlm_param_name:
            vlm_lm_head_size = vlm.config.text_config.vocab_size
            tlm_state_dict[tlm_param_name][:vlm_lm_head_size,:] = vlm_state_dict["language_model." + tlm_param_name]
        else:
            tlm_state_dict[tlm_param_name] = vlm_state_dict["language_model." + tlm_param_name]

    tlm.load_state_dict(tlm_state_dict)
    tlm.save_pretrained(output_path)

    tokenizer_vlm = AutoTokenizer.from_pretrained(vlm_id)
    tokenizer_vlm.save_pretrained(output_path)


def replace_tlm_with_seq(seq_id, tlm_id, output_path):
    seq = AutoModelForSequenceClassification.from_pretrained(
        seq_id,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    tlm = AutoModelForCausalLM.from_pretrained(
        tlm_id,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    tlm.resize_token_embeddings(seq.get_input_embeddings().num_embeddings)

    seq_state_dict = seq.state_dict()
    tlm_state_dict = tlm.state_dict()

    for tlm_param_name in tlm_state_dict.keys():
        if "lm_head" not in tlm_param_name:
            tlm_state_dict[tlm_param_name] = seq_state_dict[tlm_param_name]

    tlm.load_state_dict(tlm_state_dict)
    tlm.save_pretrained(output_path)

    tokenizer_seq = AutoTokenizer.from_pretrained(seq_id)
    tokenizer_seq.save_pretrained(output_path)


def print_debug(tlm_id, output_path):
    tlm = AutoModelForCausalLM.from_pretrained(
        tlm_id,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    out = AutoModelForCausalLM.from_pretrained(
        output_path,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    for name_tlm, param_tlm in tqdm(tlm.named_parameters()):
        for name_out, param_out in out.named_parameters():
            if name_tlm == name_out:
                if not torch.equal(param_tlm, param_out):
                    try:
                        difference = param_tlm - param_out
                        max_diff = torch.max(torch.abs(difference)).item()
                    except:
                        max_diff = 0
                    print(
                        f"Parameter weights do not match: {name_tlm}"
                        f"Max absolute difference: {max_diff}"
                    )

def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    if args.source == "vlm":
        replace_tlm_with_vlm(args.vlm_id, args.tlm_id, args.output_path)
    elif args.source == "seq":
        replace_tlm_with_seq(args.seq_id, args.tlm_id, args.output_path)
    else:
        raise NotImplementedError(f"Cannot support source == {args.source}")

    #print_debug(args.tlm_id, args.output_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
