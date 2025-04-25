import os
from argparse import ArgumentParser

import torch
from modeling_mllama_cls import MllamaForSequenceClassification
from tqdm import tqdm
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoProcessor,
                          AutoTokenizer, MllamaForConditionalGeneration)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--tlm_id', type=str, default="")
    parser.add_argument('--seq_id', type=str, default="")
    parser.add_argument('--vlseq_id', type=str, default="")
    parser.add_argument('--output_path', type=str, default="models/merged")

    args = parser.parse_args()

    return args


def map_tlm_to_vlseq(vlseq_id, tlm_id, output_path):
    vlseq = MllamaForSequenceClassification.from_pretrained(
        vlseq_id,
        num_labels=1,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    tlm = AutoModelForCausalLM.from_pretrained(
        tlm_id,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

    vlseq_state_dict = vlseq.state_dict()
    tlm_state_dict = tlm.state_dict()

    cross_attention_layers = vlseq.config.text_config.cross_attention_layers

    for tlm_param_name in tlm_state_dict.keys():
        if "layers." in tlm_param_name:
            layer_idx = int(tlm_param_name.split("layers.")[1].split(".")[0])  # Extract layer index
            vlseq_layer_idx = layer_idx
            for cross_attention_layer in sorted(cross_attention_layers):
                if vlseq_layer_idx >= cross_attention_layer:
                    vlseq_layer_idx += 1
                else:
                    break
            vlseq_param_name = "language_model." + tlm_param_name.replace(f"layers.{layer_idx}.", f"layers.{vlseq_layer_idx}.")
            vlseq_state_dict[vlseq_param_name] = tlm_state_dict[tlm_param_name]
        elif "lm_head" in tlm_param_name:
            continue
        else:
            vlseq_param_name = "language_model." + tlm_param_name
            vlseq_state_dict[vlseq_param_name] = tlm_state_dict[tlm_param_name]

    vlseq.load_state_dict(vlseq_state_dict)

    
    processor_vlseq = AutoProcessor.from_pretrained(vlseq_id)
    processor_vlseq.save_pretrained(output_path)
    
    tokenizer_tlm = AutoTokenizer.from_pretrained(tlm_id)
    tokenizer_tlm.save_pretrained(output_path)

    return vlseq


def map_score_to_vlseq(vlseq, seq_id, output_path):
    seq = AutoModelForSequenceClassification.from_pretrained(
        seq_id,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    vlseq.score = seq.score

    vlseq.save_pretrained(output_path)


def print_debug_vlseq(vlseq_id, output_path):
    vlseq = MllamaForSequenceClassification.from_pretrained(
        vlseq_id,
        num_labels=1,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    out = MllamaForSequenceClassification.from_pretrained(
        output_path,
        num_labels=1,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    for name_vlseq, param_vlseq in tqdm(vlseq.named_parameters()):
        for name_out, param_out in out.named_parameters():
            if name_vlseq == name_out:
                if not torch.equal(param_vlseq, param_out):
                    try:
                        difference = param_vlseq - param_out
                        max_diff = torch.max(torch.abs(difference)).item()
                    except:
                        max_diff = 0
                    print(
                        f"Parameter weights do not match: {name_vlseq}"
                        f"Max absolute difference: {max_diff}"
                    )


def print_debug_seq(seq_id, output_path):
    seq = AutoModelForSequenceClassification.from_pretrained(
        seq_id,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    out = MllamaForSequenceClassification.from_pretrained(
        output_path,
        num_labels=1,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    assert torch.equal(seq.score.weight, out.score.weight)


def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    vlseq = map_tlm_to_vlseq(args.vlseq_id, args.tlm_id, args.output_path)
    map_score_to_vlseq(vlseq, args.seq_id, args.output_path)

    #print_debug_vlseq(args.vlseq_id, args.output_path)
    #print_debug_seq(args.seq_id, args.output_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
