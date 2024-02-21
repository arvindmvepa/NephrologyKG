import os
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

# Specify the tokenizer for LLAMA 2 (adjust as necessary)
tokenizer = AutoTokenizer.from_pretrained("gpt2")


def clean_text(text):
    """
    Clean and preprocess text by removing excessive whitespaces and optionally
    other non-standard formatting.
    """
    # Remove leading and trailing whitespaces
    text = text.strip()
    # Replace multiple whitespace characters with a single space
    text = ' '.join(text.split())
    return text


def prepare_data_and_save_to_csv(directory, output_csv_path, chunk_size=1024, clean=True):
    data = []
    unique_id = 0

    for filename in tqdm(sorted(os.listdir(directory))):
        if filename.endswith(".txt"):  # Adjust this condition based on your file types
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                if clean:
                    text = clean_text(text)

                # Tokenize the text and split into chunks
                tokens = tokenizer.encode(text, truncation=False, max_length=None)
                for i in range(0, len(tokens)):
                    chunk_tokens = tokens[i:i + chunk_size + 1]  # +1 to include the target token
                    if len(chunk_tokens) == chunk_size + 1:
                        input_ids = chunk_tokens[:-1]
                        label = chunk_tokens[-1]
                        input_text = tokenizer.decode(input_ids)
                        target_text = tokenizer.decode(label)
                        data.append([unique_id, input_text, target_text])
                        unique_id += 1

    # Convert the list to a DataFrame
    df = pd.DataFrame(data, columns=["ID", "Input", "Target"])

    # Save to CSV
    df.to_csv(output_csv_path, index=False)


# Example usage
chunk_size = 512
directory = "textbook_txt_files"
output_csv_path = f"input_target_pairs_gp2tk_toklen_{chunk_size}_clean_no_trunc_1target.csv"
prepare_data_and_save_to_csv(directory, output_csv_path, chunk_size=512)