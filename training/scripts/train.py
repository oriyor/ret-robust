import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from data_collator import DataCollatorSelfAsk
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="meta-llama/Llama-2-13b-hf",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Ori/llama-2-13b-peft-nq-no-ret",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="nq_no_ret",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    return parser.parse_args()


def train_model(args):
    """
    set up and train using HF trainer
    based on example code in HF
    """
    # get args
    dataset_name = args.dataset_name
    model_name = args.base_model_name
    output_dir = args.output_dir
    seed = args.seed

    # load dataset
    train_dataset = load_dataset(dataset_name, split="train")

    # setup 4 bit training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        use_auth_token=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    # default configs
    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    num_train_epochs = 5
    gradient_accumulation_steps = 1
    optim = "paged_adamw_32bit"
    save_strategy = "epoch"
    learning_rate = 2e-4
    lr_scheduler_type = "linear"
    warmup_ratio = 0.03
    logging_steps = 25
    prediction_loss_only = True
    eval_steps = 0.2
    bf16 = True

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        seed=seed,
        num_train_epochs=num_train_epochs,
        auto_find_batch_size=4,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_strategy=save_strategy,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        logging_strategy="epoch",
        logging_steps=logging_steps,
        prediction_loss_only=prediction_loss_only,
        eval_steps=eval_steps,
        bf16=bf16,
    )
    max_seq_length = 4096

    # init the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        data_collator=DataCollatorSelfAsk(
            tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf"),
            mlm=False,
        ),
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    # train!
    trainer.train()


if __name__ == "__main__":
    """ """
    args = parse_args()
    train_model(args)
