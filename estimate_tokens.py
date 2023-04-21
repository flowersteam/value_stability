import tiktoken
import argparse
import os

parser = argparse.ArgumentParser(description="Estimate the number of tokens in a text file using OpenAI's API.")
parser.add_argument("file_path", metavar="file_path", type=str, help="The path to the text file to process.")
args = parser.parse_args()

if not os.path.exists(args.file_path):
    print("File not found.")
    exit()


with open(args.file_path, "r") as f:
    text = f.read()


encoder = tiktoken.encoding_for_model('gpt-3.5-turbo-0301')
assert encoder == tiktoken.encoding_for_model('gpt-4-0314')

n_tokens = len(encoder.encode(text))

print("total GPT tokens used: {}".format(n_tokens))
print(f"\tgpt-4 ~ {0.04 * n_tokens / 1000:.4f} dollars")
print(f"\tgpt-3.5 ~ {0.002 * n_tokens / 1000:.4f} dollars")
print(f"\tdavinci ~ {0.02 * n_tokens / 1000:.4f} dollars")
print(f"\tcurie ~ {0.002 * n_tokens / 1000:.4f} dollars")
print(f"\tbabagge ~ {0.0005 * n_tokens / 1000:.4f} dollars")
print(f"\tada ~ {0.0004 * n_tokens / 1000:.4f} dollars")
