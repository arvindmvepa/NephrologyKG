import os
import pandas as pd
#from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer_name="HuggingFaceH4/zephyr-7b-beta"
#tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


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
    chunk_and_target_size = chunk_size + 1  # +1 for the target token

    for filename in tqdm(sorted(os.listdir(directory))):
        if filename.endswith(".txt"):  # Adjust this condition based on your file types
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                if clean:
                    text = clean_text(text)

                # Tokenize the text and split into chunks
                tokens = tokenizer.encode(text, truncation=False, max_length=None)
                for i in range(0, len(tokens), chunk_and_target_size):
                    chunk_tokens = tokens[i:i + chunk_and_target_size]  # +1 to include the target token
                    if len(chunk_tokens) == chunk_and_target_size:
                        input_ids = chunk_tokens[:-1]
                        input_text = tokenizer.decode(input_ids)
                        data.append([input_text])
                        unique_id += 1

    # Convert the list to a DataFrame
    df = pd.DataFrame(data, columns=["text"])

    # Save to CSV
    df.to_csv(output_csv_path, index=False)


def standard_save_to_csv(directory, output_csv_path, clean=True):
    data = []

    for filename in tqdm(sorted(os.listdir(directory))):
        if filename.endswith(".txt"):  # Adjust this condition based on your file types
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                if clean:
                    text = clean_text(text)
                if text:
                    data.append([text])

    # Convert the list to a DataFrame
    df = pd.DataFrame(data, columns=["text"])

    # Save to CSV
    df.to_csv(output_csv_path, index=False)

if __name__ == '__main__':
    chunk_size = 512
    directory = r"/Users/arvin/Documents/ucla research/nephrology nlp/textbook_txt_files_v2"
    #output_csv_path = f"input_target_pairs_zephyr7bbetatk_toklen_{chunk_size}_clean_no_trunc_1target.csv"
    #prepare_data_and_save_to_csv(directory, output_csv_path, chunk_size=512)
    output_csv_path = f"neph_v2.csv"
    standard_save_to_csv(directory, output_csv_path)