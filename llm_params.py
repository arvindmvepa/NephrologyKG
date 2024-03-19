from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    LlamaForCausalLM,
    LlamaTokenizer
)


llm_dict = {"llama": {"model": LlamaForCausalLM, "tokenizer": LlamaTokenizer},
            "default": {"model": AutoModelForCausalLM, "tokenizer": AutoTokenizer}}