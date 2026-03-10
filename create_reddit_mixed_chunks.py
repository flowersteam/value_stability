import pandas as pd
import os
import json
import tiktoken
import numpy as np
from termcolor import cprint

import praw

def is_subreddit_safe(subreddit):
    try:
        if reddit.subreddit(subreddit).over18:
            return False
    except:
        return False

    return True


def check_subreddit(subreddit):
    return subreddit, is_subreddit_safe(subreddit)


if __name__ == "__main__":

    user_agent = "filter_subs/0.1 by grggrggrggrg"
    reddit = praw.Reddit(
        client_id=os.environ.get('REDDIT_CLIENT_ID'),
        client_secret=os.environ.get('REDDIT_CLIENT_SECRET'),
        user_agent=user_agent
    )

    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")

    # this has good data, but also many bad subreddits, which we should exclude
    file_path = '/home/flowers-user/Documents/projects/SocialLLM/clear-corpus-webis-tldr-17.json'

    # List to hold the entries
    chunks_data = [[], [], [], [], []]

    chunk_size = 100

    tokenizer = tiktoken.get_encoding("cl100k_base")

    # for save subreddits -> load previous
    with open('contexts/subreddits_save.json') as f:
        subreddits_safe = json.load(f)

    save_chunks_dir = "contexts/mixed_v2_reddit_chunks"
    os.makedirs(save_chunks_dir, exist_ok=True)

    # load reddit posts
    with open(file_path, 'r') as file:
        for line_i, line in enumerate(file):
            if line_i % 100_000 == 0:
                print("Line i: ", line_i)
                print("chunk sizes:", [len(cd) for cd in chunks_data])

            entry = json.loads(line)

            # if it's not safe -> skip, if it's safe or we don't know -> keep
            if not subreddits_safe.get(entry['subreddit'], True):
                continue

            # Mixed - find the biggest bucket
            # chunk_limits = [7000, 5000, 4000, 3000, 1000]
            chunk_limits = [7000, 5000, 4000, 2500, 500]

            # laksi ? sa 500
            chunk_i = None
            for chunk_limit_i, chunk_limit in enumerate(chunk_limits):
                if entry['n_tokens'] < chunk_limit:
                    chunk_i = chunk_limit_i
                else:
                    break

            if chunk_i is not None:
                # put the entry in the correct bucket
                chunks_data[chunk_i].append({
                    "content": entry["content"],
                    "subreddit": entry["subreddit"],
                    "n_tokens": entry["n_tokens"]
                })

    # parse and save posts
    added_subreddits = []
    for chunk_i, data in enumerate(chunks_data):
        df = pd.DataFrame(data)

        # longest post for each subreddit
        idx = df.groupby('subreddit')['n_tokens'].nlargest(1).index.levels[1]
        df_longest_per_subreddit = df.loc[idx]

        # check if subreddits are safe, check for new subreddits
        subreddits = list(df_longest_per_subreddit['subreddit'].unique())
        for sub_i, subreddit in enumerate(subreddits):
            if sub_i % 100 == 0:
                print(f"Subreddit {sub_i}/{len(subreddits)}")

            if subreddit not in subreddits_safe:
                subreddits_safe[subreddit] = is_subreddit_safe(subreddit)

            if sub_i % 1000 == 0:
                # save new version of subreddit black/white list
                with open('contexts/subreddits_save.json', 'w') as f:
                    json.dump(subreddits_safe, f)
                    print("New version of subreddits saved")

        # save new version of subreddit black/white list
        with open('contexts/subreddits_save.json', 'w') as f:
            json.dump(subreddits_safe, f)
            print("New version of subreddits saved")

        # keep only the safe subreddits
        df_longest_per_subreddit = df_longest_per_subreddit[
            df_longest_per_subreddit.apply(lambda x: subreddits_safe[x['subreddit']], axis=1)
        ]
        # keep only the subreddits that were not previously added
        df_longest_per_subreddit = df_longest_per_subreddit[
            df_longest_per_subreddit.apply(lambda x: x['subreddit'] not in added_subreddits, axis=1)
        ]

        print("Unique subreddits:", len(df_longest_per_subreddit))

        if len(df_longest_per_subreddit) < chunk_size:
            cprint("Not enough entries left", "red")

        # take the chunk_size longest posts
        df_longest = df_longest_per_subreddit.nlargest(chunk_size, "n_tokens")
        df_longest = df_longest.sample(frac=1)
        chunk_unique_subreddits = list(df_longest['subreddit'])

        # add subreddits to list of added subreddits
        added_subreddits.extend(chunk_unique_subreddits)

        # assert no duplicate subreddits
        assert len(added_subreddits) == len(set(added_subreddits))

        # log lengths data
        n_toks = list(df_longest['n_tokens'])
        print(f"chunk {chunk_i} (limit {chunk_limits[chunk_i]}")
        print("Mean:", np.mean(n_toks))
        print("Min:", min(n_toks))
        print("Max:", max(n_toks))

        # save chunk
        outfile = f'{save_chunks_dir}/chunk_{chunk_i}.jsonl'
        df_longest.to_json(outfile, orient='records', lines=True)
        print(f"Chunk {chunk_i} saved to {outfile}.")
