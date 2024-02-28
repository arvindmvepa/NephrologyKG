import torch
from datasets import load_dataset
import transformers
from transformers import DataCollatorForLanguageModeling
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)


def train_model(model, tokenizer, data, save_model_name="neph_model"):
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    OUTPUT_DIR = "experiments"
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=150,
        learning_rate=2e-4,
        fp16=True,
        save_total_limit=3,
        logging_steps=1,
        output_dir=OUTPUT_DIR,
        max_steps=150,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="tensorboard",
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=training_args,
        data_collator=data_collator,
    )
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(save_model_name)


def load_tokenizer_from_huggingface(tokenizer_name="HuggingFaceH4/zephyr-7b-beta"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_llm_from_huggingface(model_name="HuggingFaceH4/zephyr-7b-beta", use_quantization=False, r=16, lora_alpha=32,
                              target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
                              lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"):
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    print(model)

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )

    model = get_peft_model(model, config)
    return model


def load_dataset_from_file(data_path):
    data = load_dataset("csv", data_files=data_path)
    data = data['train']
    data = data.train_test_split(test_size=0.2)
    return data

def process_dataset(data, tokenizer, block_size=512):
    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["text"]])
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    tokenized_data = data.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=data["train"].column_names,
    )
    lm_dataset = tokenized_data.map(group_texts, batched=True, num_proc=4)
    return lm_dataset


if __name__ == '__main__':
    block_size = 512
    #data_path = f"input_target_pairs_zephyr7bbetatk_toklen_{block_size}_clean_no_trunc_1target.csv"
    data_path = "neph.csv"
    data = load_dataset_from_file(data_path)
    tokenizer = load_tokenizer_from_huggingface()
    processed_data = process_dataset(data, tokenizer, block_size=block_size)
    model = load_llm_from_huggingface()
    train_model(model, tokenizer, processed_data)
