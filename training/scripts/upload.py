import torch
import argparse
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_model_id",
        type=str,
        default="nq_no_ret/checkpoint-625",
    )
    parser.add_argument(
        "--hf_model_id",
        type=str,
        default="",
        help="The location of the model on the HF hub",
    )
    return parser.parse_args()


def upload_model(args):
    """
    :param args:
    :return:
    """
    peft_model_id = args.local_model_id
    model_id_load = args.hf_model_id

    # load model
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id)

    # push to hub
    # tokenizer
    tokenizer.push_to_hub(model_id_load, use_auth_token=True)
    # safetensors
    model.push_to_hub(model_id_load, use_auth_token=True, safe_serialization=True)
    # torch tensors
    model.push_to_hub(model_id_load, use_auth_token=True)


if __name__ == "__main__":
    """ """
    args = parse_args()
    upload_model(args)
