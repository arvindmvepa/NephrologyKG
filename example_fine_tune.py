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


def load_llm():
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

def test_text_gen(model, tokenizer):
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

    device = "cuda:0"

    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == '__main__':
    model, tokenizer = load_llm()
    test_text_gen(model, tokenizer)