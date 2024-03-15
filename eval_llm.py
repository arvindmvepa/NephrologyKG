from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    PeftConfig,
    PeftModel
)
import csv
import torch


torch.set_printoptions(threshold=10_000)
decoding_strats= {"greedy": {"num_beams": 1, "do_sample": False},
                  "contrast": {"penalty_alpha": 0.6, "top_k": 4},
                  "beam": {"num_beams": 4, "do_sample": False},
                  "top_k": {"do_sample": True, "top_k": 50}}


def eval_llm(model_name, save_file, questions=[], prompt="", max_new_tokens=1000, decoding_strat="greedy",
             used_lora=True, debug=False):
    if used_lora:
        config = PeftConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = PeftModel.from_pretrained(model, model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    decoding_kwargs = decoding_strats[decoding_strat]
    content = []
    for question in questions:
        question = prompt + question
        inputs = tokenizer(question, return_tensors="pt").to("cuda")
        generated_ids = model.generate(**inputs, num_return_sequences=1, max_new_tokens=max_new_tokens, **decoding_kwargs)
        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        answer = answer.split(question)[-1]
        if debug:
            content.append((question, answer, generated_ids))
        else:
            content.append((question, answer))
        print(f"question: {question}")
        print(f"answer: {answer}")
        if debug:
            print(f"generated_ids: {generated_ids}")
        print()

    with open(save_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if debug:
            writer.writerow(["question", "answer", "generated_ids"])
        else:
            writer.writerow(["question", "answer"])
        for line in content:
            writer.writerow(line)


if __name__ == '__main__':
    block_size = 512
    fp16 = True
    optimizer = "adamw_torch_fused"
    per_device_train_batch_size=8
    save_eval_steps=1000
    data_path = "neph.csv"
    num_train_epochs = 10
    model_name = f"neph_blocksize{block_size}_optm{optimizer}_fp16{fp16}_bs{per_device_train_batch_size}_epochs{num_train_epochs}_v2"
    #model_name = "HuggingFaceH4/zephyr-7b-beta"
    decoding_strat = "beam"
    used_lora = True
    tag = "_v4"
    prompt= "Extract all the entities from the ensuing paragraph. Please provide them in a list format: "
    questions = [# q1
                 "Glomerular hypertrophy may be marker of FSGS. Glomerular enlarge-\n"
                 "ment precedes overt glomerulosclerosis in FSGS (19). Patients with abnor-\n"
                 "mal glomerular growth on initial biopsies that did not show overt sclerotic\n"
                 "lesions  subsequently  developed  overt  glomerulosclerosis,  documented  in\n"
                 "later  biopsies.  A  cutoff  of  glomerular area  larger  than  50%  more  than\n"
                 "normal for age indicated increased risk for progression. Of note, glomeruli\n"
                 "grow until approximately age 18 years, so age-matched controls must be\n"
                 "used  in  the  pediatric  population.  Since  tissue  processing  methods  may\n"
                 "inﬂuence the size of structures in tissue, it is imperative that each labora-\n"
                 "tory determines normal ranges for this parameter.",
                 # q2
                 "Optimal management of patients\n"
                 "with chronic kidney disease (CKD) requires appropriate\n"
                 "interpretation and use of the markers and stages of CKD, early disease recognition, and\n"
                 "collaboration between primary care physicians and nephrologists.Because multiple terms have\n"
                 "been applied to chronic kidney disease (CKD), eg, chronic renal insufficiency, chronic renal\n"
                 "disease, and chronic renal failure, the National Kidney Foundation Kidney Disease Outcomes\n"
                 "Quality Initiative™ (NKF KDOQI™) has defined the all-encompassing term, CKD.Using kidney\n"
                 "rather than renal improves understanding by patients, families, healthcare workers, and the lay\n"
                 "public.This term includes the continuum of kidney dysfunction from mild kidney damage to\n"
                 "kidney failure, and it also includes the term, end-stage renal disease (ESRD)."
                 ]
    eval_llm(model_name, model_name.replace('/','_') + f"_{decoding_strat}_entities" + tag + ".csv",
             decoding_strat=decoding_strat, questions=questions, prompt=prompt, used_lora=used_lora, debug=False)



