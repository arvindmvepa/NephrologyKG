from datasets import list_datasets, load_dataset
import pandas as pd
import json
import os
from pprint import pprint

import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
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
)


def dataset_preprocess():
    # download dataset
    dataset = load_dataset('web_nlg', 'release_v3.0_en')

    data_test = dataset['test']
    data_train = dataset['train']

    # combine both the training and test
    df_train = pd.DataFrame(data_train)
    df_test = pd.DataFrame(data_test)
    df_all = df_train.append(df_test)

    # extract to formate easier to use
    text = []
    triple = []
    sentence = ''
    for i in range(len(df_all)):
        triple.append(
            str(df_all['modified_triple_sets'].iloc[i]['mtriple_set'][0]).replace('_', ' ').replace("'", "").replace(
                '"', '').replace('|', '-->'))
        text.append(str(df_all['lex'].iloc[i]['text']).replace('_', ' ').replace("[", '').replace(']', '').replace("'",
                                                                                                                   "").replace(
            '"', ''))

    # save to data frame
    dataset_df = pd.DataFrame()
    dataset_df['text'] = text
    dataset_df['triple'] = triple

    # save this to a CSV file
    dataset_df.to_csv('Web_NLG.csv', index=False)


def load_dataset():
    data = load_dataset("csv", data_files="Web_NLG.csv")
    def generate_prompt(data_point):
        return f"""
    INPUT: {data_point["text"]}
    OUTPUT: {data_point["triple"]}
    """.strip()

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
        return tokenized_full_prompt

    data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    print(data)
    return data


def load_llm_from_huggingface():
    MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
    TOKENIZER_NAME = "HuggingFaceH4/zephyr-7b-beta"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    print(model)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    return model, tokenizer


def train_llm(model, data):
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
        train_dataset=data,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained("trained-model")


def load_llm():
    PEFT_MODEL = "trained-model"

    config = PeftConfig.from_pretrained(PEFT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        # quantization_config=bnb_config,
        device_map="cuda:0",
        # trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, PEFT_MODEL)
    return model, tokenizer

def test1_text_gen(model, tokenizer, device="cuda:0"):
    prompt = f"""
    how can i clean here?
    """.strip()
    print(prompt)

    generation_config = model.generation_config
    generation_config.max_new_tokens = 200
    generation_config.temperature = 0.7
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def test2_text_gen(model, tokenizer, device="cuda:0"):
    generation_config = model.generation_config
    generation_config.max_new_tokens = 256
    generation_config.temperature = 0.4
    generation_config.top_p = 0.2
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    def generate_response(question: str) -> str:
        prompt = f"""
    [S]
    INPUT: {question}
    OUTPUT: 
    """.strip()
        encoding = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                generation_config=generation_config,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        assistant_start = "OUTPUT:"
        response_start = response.find(assistant_start)
        return response[response_start + len(assistant_start):].strip()

    prompt = """ 
    Bangkok It is the capital and most populous city of Thailand. It is the center of government, education, transportation.
    Finance, banking, commerce, communications and national prosperity. Located on the Chao Phraya River Delta. 
    The Chao Phraya River flows through it and divides the city into 2 sides: the Phra Nakhon side and the Thonburi side. 
    Bangkok has a total area of 1,568.737 sq. km. with a registered population of more than 6 million people, making Bangkok classified as a Primate City. 
    It is said that Bangkok used to be "The greatest city in the world" because in the year 2000
     """
    print(generate_response(prompt))

if __name__ == '__main__':
    device = "cuda:0"
    model, tokenizer = load_llm_from_huggingface()
    test1_text_gen(model, tokenizer, device=device)
    dataset_preprocess()
    data = load_dataset()
    train_llm(model, data)
    model, tokenizer = load_llm()
    test2_text_gen(model, tokenizer, device=device)