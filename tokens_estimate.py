import os
import pandas as pd
import tiktoken

encoder = tiktoken.encoding_for_model('gpt-4')

def count_tokens(file_path):

    # Read CSV file
    df = pd.read_csv(file_path)

    # Join all text in the dataframe into a single string
    text = ' '.join(df.values.flatten().astype(str))

    # Tokenize and return the number of tokens
    tokens = list(encoder.encode(text))
    return len(tokens)


def count_tokens_in_dir(dir_path):
    # Initialize total token count
    total_tokens = 0
    # Iterate over all CSV files in the directory
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(dir_path, file_name)
            tokens = count_tokens(file_path)
            print(f"Filename {file_name} - tokens: {tokens}")
            total_tokens += tokens

    return total_tokens


# Usage
dir_path = './data_mmlu/test'
n_tokens = count_tokens_in_dir(dir_path)
print(f"Total GPT-4 tokens in the CSV files: {n_tokens}")
print(f"Total price for GPT-4: {n_tokens/1000 * 0.03}")
print(f"Total price for GPT-35: {n_tokens/1000 * 0.002}")
