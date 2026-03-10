import json
import tiktoken

# clear bad subreddits
file_path = '/home/flowers-user/Documents/projects/SocialLLM/corpus-webis-tldr-17.json'

with open('contexts/subreddits_save.json') as f:
    subreddits_safe = json.load(f)

tokenizer = tiktoken.get_encoding("cl100k_base")


dataset = []
with open(file_path, 'r') as file:
    for line_i, line in enumerate(file):
        if line_i % 50_000 == 0:
            print("Line i: ", line_i)

        entry = json.loads(line)

        if "subreddit" not in entry:
            continue

        if subreddits_safe.get(entry['subreddit'], True):
            entry["n_tokens"] = len(tokenizer.encode(entry["content"]))
            dataset.append(entry)

# save
clear_file_path = '/home/flowers-user/Documents/projects/SocialLLM/clear-corpus-webis-tldr-17.json'
print("Clear dataset len:", len(dataset))

with open(clear_file_path, 'w') as outfile:
    for entry_i, entry in enumerate(dataset):
        if entry_i % 100_000 == 0:
            print("Entry i: ", entry_i)

        json.dump(entry, outfile)
        outfile.write('\n')

print(f"Saved to {clear_file_path}")
